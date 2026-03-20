import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

from utils.config import load_config, set_seed, get_device, ensure_dir
from models.sc_unet import SCUNetEncoder, SCUNetDecoder
from channel.semantic_channel import semantic_channel_real
from losses.reconstruction import psnr


def preprocess_for_resnet(x: torch.Tensor):
    """
    按官方权重推荐预处理：
    resize + crop + normalize
    """
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    return preprocess(x)


@torch.no_grad()
def extract_features_and_logits(model, x):
    """
    返回:
    - logits
    - penultimate feature
    """
    feat = None

    def hook_fn(module, inp, out):
        nonlocal feat
        feat = out.flatten(1)

    handle = model.avgpool.register_forward_hook(hook_fn)
    logits = model(x)
    handle.remove()

    return logits, feat


@torch.no_grad()
def evaluate_downstream_consistency(encoder, decoder, model, dl, device, snr_db, channel_cfg):
    encoder.eval()
    decoder.eval()
    model.eval()

    total = 0
    top1_same = 0
    top5_same = 0
    feat_cos_sum = 0.0
    psnr_sum = 0.0
    batches = 0

    for x, _ in dl:
        x = x.to(device)

        # 原图输出
        x_for_resnet = preprocess_for_resnet(x)
        logits_clean, feat_clean = extract_features_and_logits(model, x_for_resnet)

        # SC重构图
        z, skips = encoder(x)
        z_noisy = semantic_channel_real(
            z,
            snr_db=snr_db,
            pt=float(channel_cfg["pt"]),
            eps=float(channel_cfg["eps"])
        )
        x_hat = decoder(z_noisy, skips).clamp(0.0, 1.0)

        xhat_for_resnet = preprocess_for_resnet(x_hat)
        logits_hat, feat_hat = extract_features_and_logits(model, xhat_for_resnet)

        pred1_clean = logits_clean.argmax(dim=1)
        pred1_hat = logits_hat.argmax(dim=1)
        top1_same += (pred1_clean == pred1_hat).sum().item()

        top5_clean = torch.topk(logits_clean, k=5, dim=1).indices
        top5_hat = torch.topk(logits_hat, k=5, dim=1).indices
        same5 = []
        for i in range(x.size(0)):
            a = set(top5_clean[i].tolist())
            b = set(top5_hat[i].tolist())
            same5.append(int(len(a.intersection(b)) > 0))
        top5_same += sum(same5)

        feat_cos = F.cosine_similarity(feat_clean, feat_hat, dim=1)
        feat_cos_sum += feat_cos.mean().item()

        psnr_sum += psnr(x_hat, x)
        batches += 1
        total += x.size(0)

    return {
        "top1_consistency": top1_same / max(total, 1),
        "top5_overlap_rate": top5_same / max(total, 1),
        "feature_cosine": feat_cos_sum / max(batches, 1),
        "avg_psnr": psnr_sum / max(batches, 1)
    }


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
        transform=tfm_test
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=int(eval_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(eval_cfg["num_workers"]),
        pin_memory=(device == "cuda")
    )

    encoder = SCUNetEncoder(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        latent_channels=int(model_cfg["latent_channels"])
    ).to(device)

    decoder = SCUNetDecoder(
        out_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        latent_channels=int(model_cfg["latent_channels"])
    ).to(device)

    encoder.load_state_dict(torch.load("./checkpoints_sc/sc_encoder.pth", map_location=device))
    decoder.load_state_dict(torch.load("./checkpoints_sc/sc_decoder.pth", map_location=device))

    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    model.eval()

    print("Loaded fixed downstream model: ResNet50_Weights.DEFAULT")

    for snr_db in eval_cfg["snrs"]:
        metrics = evaluate_downstream_consistency(
            encoder=encoder,
            decoder=decoder,
            model=model,
            dl=dl_test,
            device=device,
            snr_db=float(snr_db),
            channel_cfg=channel_cfg
        )

        print(
            f"SNR={snr_db:>4} dB | "
            f"top1_consistency={metrics['top1_consistency']:.4f} | "
            f"top5_overlap_rate={metrics['top5_overlap_rate']:.4f} | "
            f"feature_cosine={metrics['feature_cosine']:.4f} | "
            f"avg_psnr={metrics['avg_psnr']:.3f}"
        )


if __name__ == "__main__":
    main()