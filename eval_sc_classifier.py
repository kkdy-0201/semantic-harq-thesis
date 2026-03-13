import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.config import load_config, set_seed, get_device, ensure_dir
from models.sc_cnn import SCEncoder, SCDecoder
from models.downstream_resnet import build_resnet50_cifar10
from channel.semantic_channel import semantic_channel_real
from losses.reconstruction import psnr


def preprocess_for_resnet(x: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def eval_clean_classifier(classifier, dl, device):
    classifier.eval()
    correct = 0
    total = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        logits = classifier(preprocess_for_resnet(x))
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def eval_sc_plus_classifier(encoder, decoder, classifier, dl, device, snr_db, channel_cfg):
    encoder.eval()
    decoder.eval()
    classifier.eval()

    correct = 0
    total = 0
    psnr_sum = 0.0
    batches = 0

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        z = encoder(x)
        z_noisy = semantic_channel_real(
            z,
            snr_db=snr_db,
            pt=float(channel_cfg["pt"]),
            eps=float(channel_cfg["eps"]),
        )
        x_hat = decoder(z_noisy).clamp(0.0, 1.0)

        logits = classifier(preprocess_for_resnet(x_hat))
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()
        psnr_sum += psnr(x_hat, x)
        batches += 1

    return correct / max(total, 1), psnr_sum / max(batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg)

    data_cfg = cfg["dataset"]
    model_cfg = cfg["sc_model"]
    channel_cfg = cfg["channel"]
    eval_cfg = cfg["eval"]

    ensure_dir(eval_cfg["save_dir"])

    tfm_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    ds_test = datasets.CIFAR10(
        root=data_cfg["root"],
        train=False,
        download=bool(data_cfg["download"]),
        transform=tfm_test,
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=int(eval_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(eval_cfg["num_workers"]),
        pin_memory=(device == "cuda"),
    )

    encoder = SCEncoder(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
    ).to(device)
    decoder = SCDecoder(
        out_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
    ).to(device)

    encoder.load_state_dict(torch.load("./checkpoints_sc/sc_encoder.pth", map_location=device))
    decoder.load_state_dict(torch.load("./checkpoints_sc/sc_decoder.pth", map_location=device))

    classifier = build_resnet50_cifar10(
        pretrained=False,
        num_classes=int(data_cfg["num_classes"])
    ).to(device)
    classifier.load_state_dict(torch.load("./checkpoints_downstream/resnet50_cifar10.pth", map_location=device))

    clean_acc = eval_clean_classifier(classifier, dl_test, device)
    print(f"Clean classifier acc on original images: {clean_acc:.4f}")

    for snr_db in eval_cfg["snrs"]:
        acc, avg_psnr = eval_sc_plus_classifier(
            encoder, decoder, classifier, dl_test, device, float(snr_db), channel_cfg
        )
        print(f"SNR={snr_db:>4} dB | cls_acc_on_xhat={acc:.4f} | avg_psnr={avg_psnr:.3f}")


if __name__ == "__main__":
    main()