import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.config import load_config, set_seed, ensure_dir, get_device
from models.sc_cnn import SCEncoder, SCDecoder
from channel.semantic_channel import semantic_channel_real, sample_snr_db
from losses.reconstruction import ReconstructionLoss, psnr


@torch.no_grad()
def evaluate_sc(encoder, decoder, dl, device, channel_cfg):
    encoder.eval()
    decoder.eval()

    psnr_sum = 0.0
    n = 0

    for x, _ in dl:
        x = x.to(device)
        snr_db = float(channel_cfg["snr_db"])

        z = encoder(x)
        z_noisy = semantic_channel_real(
            z,
            snr_db=snr_db,
            pt=float(channel_cfg["pt"]),
            eps=float(channel_cfg["eps"]),
        )
        x_hat = decoder(z_noisy)

        psnr_sum += psnr(x_hat, x)
        n += 1

    return psnr_sum / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg)

    train_cfg = cfg["train_sc"]
    data_cfg = cfg["dataset"]
    model_cfg = cfg["sc_model"]
    channel_cfg = cfg["channel"]

    ensure_dir(train_cfg["save_dir"])
    print("Device:", device)

    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    ds_train = datasets.CIFAR10(
        root=data_cfg["root"],
        train=True,
        download=bool(data_cfg["download"]),
        transform=tfm_train,
    )
    ds_test = datasets.CIFAR10(
        root=data_cfg["root"],
        train=False,
        download=bool(data_cfg["download"]),
        transform=tfm_test,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=(device == "cuda"),
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
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

    criterion = ReconstructionLoss(cfg)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(train_cfg["epochs"])
    )

    best_psnr = -1.0
    best_epoch = -1

    for ep in range(1, int(train_cfg["epochs"]) + 1):
        encoder.train()
        decoder.train()

        loss_sum = 0.0
        steps = 0

        for x, _ in dl_train:
            x = x.to(device)
            snr_db = sample_snr_db(channel_cfg)

            optimizer.zero_grad(set_to_none=True)

            z = encoder(x)
            z_noisy = semantic_channel_real(
                z,
                snr_db=snr_db,
                pt=float(channel_cfg["pt"]),
                eps=float(channel_cfg["eps"]),
            )
            x_hat = decoder(z_noisy)

            loss, stats = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            steps += 1

        scheduler.step()
        test_psnr = evaluate_sc(encoder, decoder, dl_test, device, channel_cfg)

        print(
            f"[Epoch {ep:03d}/{train_cfg['epochs']}] "
            f"train_loss={loss_sum / max(steps,1):.6f} "
            f"test_psnr@{channel_cfg['snr_db']}dB={test_psnr:.3f}"
        )

        if test_psnr > best_psnr:
            best_psnr = test_psnr
            best_epoch = ep
            torch.save(encoder.state_dict(), os.path.join(train_cfg["save_dir"], "sc_encoder.pth"))
            torch.save(decoder.state_dict(), os.path.join(train_cfg["save_dir"], "sc_decoder.pth"))
            print(f"Saved best SC checkpoint @ epoch {ep}, PSNR={best_psnr:.3f}")

    print(f"Done. Best PSNR={best_psnr:.3f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()