import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.semanticencoder import ImageSemanticEncoder, SemanticDecoder, TaskHead
from utils.seed import set_seed


def acc_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def semantic_channel_real(z: torch.Tensor, snr_db: float, pt: float = 1.0, eps: float = 1e-8):

    # 每个样本单独计算平均功率 Pz = (1/d)||z||^2
    dims = tuple(range(1, z.dim()))
    pz = torch.mean(z.pow(2), dim=dims, keepdim=True)   # shape [B,1,...]

    # 功率归一化到 Pt
    z_tx = math.sqrt(pt) * z / torch.sqrt(pz + eps)

    # 线性 SNR
    gamma = 10.0 ** (snr_db / 10.0)

    # 噪声方差 sigma^2 = Pt / gamma
    noise_var = pt / gamma
    noise_std = math.sqrt(noise_var)

    n = noise_std * torch.randn_like(z_tx)
    y = z_tx + n
    return y


@torch.no_grad()
def evaluate_snr(encoder, decoder, head, dl, device, snr_db: float, pt: float, eps: float) -> float:
    encoder.eval()
    decoder.eval()
    head.eval()

    acc_sum, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)

        z = encoder(x)
        z_noisy = semantic_channel_real(z, snr_db=snr_db, pt=pt, eps=eps)

        feat = decoder(z_noisy)
        logits = head(feat)

        acc_sum += acc_from_logits(logits, y)
        n += 1

    return acc_sum / max(n, 1)


def train_one_epoch(
    encoder, decoder, head, dl, optimizer, device,
    snr_mode: str, snr_db: float, snr_min: float, snr_max: float,
    pt: float, eps: float
):
    encoder.train()
    decoder.train()
    head.train()

    ce = nn.CrossEntropyLoss()
    loss_sum, n = 0.0, 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)

        if snr_mode == "fixed":
            snr = snr_db
        elif snr_mode == "uniform":
            snr = random.uniform(snr_min, snr_max)
        else:
            raise ValueError("snr_mode must be fixed or uniform")

        optimizer.zero_grad(set_to_none=True)

        # 编码
        z = encoder(x)

        # 按图中公式：功率约束 + AWGN
        y_ch = semantic_channel_real(z, snr_db=snr, pt=pt, eps=eps)

        # 解码与任务头
        feat = decoder(y_ch)
        logits = head(feat)

        loss = ce(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        n += 1

    return loss_sum / max(n, 1)


def main():
    parser = argparse.ArgumentParser()

    # train hyper
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    # model
    parser.add_argument("--semantic_dim", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=10)

    # data/save
    parser.add_argument("--data_root", type=str, default="./data/raw")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_noisy")

    # channel / noise
    parser.add_argument("--snr_mode", type=str, default="uniform", choices=["fixed", "uniform"])
    parser.add_argument("--snr_db", type=float, default=10.0)
    parser.add_argument("--snr_min", type=float, default=0.0)
    parser.add_argument("--snr_max", type=float, default=20.0)

    # 图中公式里的 Pt 和 eps
    parser.add_argument("--pt", type=float, default=1.0, help="Average transmit power Pt")
    parser.add_argument("--eps", type=float, default=1e-8, help="Numerical epsilon in normalization")

    # device
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # CIFAR10 mean/std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds_train = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=tfm_train)
    ds_test = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=tfm_test)

    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )
    dl_test = DataLoader(
        ds_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )

    # model
    encoder = ImageSemanticEncoder(args.semantic_dim).to(device)
    decoder = SemanticDecoder(args.semantic_dim).to(device)
    head = TaskHead(args.semantic_dim, args.num_classes).to(device)

    optimizer = optim.SGD(
        list(encoder.parameters()) + list(decoder.parameters()) + list(head.parameters()),
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = -1.0
    best_epoch = -1

    print(f"Train: {len(ds_train)}  Test: {len(ds_test)}")
    print(
        f"snr_mode={args.snr_mode}, snr_db={args.snr_db}, "
        f"snr_min={args.snr_min}, snr_max={args.snr_max}, "
        f"Pt={args.pt}, eps={args.eps}"
    )

    eval_snrs = [-5, 0, 5, 10, 15, 20]

    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(
            encoder, decoder, head, dl_train, optimizer, device,
            snr_mode=args.snr_mode,
            snr_db=args.snr_db,
            snr_min=args.snr_min,
            snr_max=args.snr_max,
            pt=args.pt,
            eps=args.eps
        )

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        # 用 10dB 做 best checkpoint
        acc10 = evaluate_snr(
            encoder, decoder, head, dl_test, device,
            snr_db=10.0, pt=args.pt, eps=args.eps
        )

        print(f"[Epoch {ep:03d}/{args.epochs}] loss={loss:.4f}  acc@10dB={acc10:.4f}  lr={lr_now:.6f}")

        if acc10 > best:
            best = acc10
            best_epoch = ep
            torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder.pth"))
            torch.save(decoder.state_dict(), os.path.join(args.save_dir, "decoder.pth"))
            torch.save(head.state_dict(), os.path.join(args.save_dir, "task_head.pth"))
            print(f"Saved best checkpoint (acc@10dB={best:.4f})")

    print(f"\nDone. Best acc@10dB={best:.4f} @ epoch {best_epoch}")
    print("Saved to:", args.save_dir)

    print("\n=== Final Acc vs SNR (using best weights saved on acc@10dB) ===")
    encoder.load_state_dict(torch.load(os.path.join(args.save_dir, "encoder.pth"), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(args.save_dir, "decoder.pth"), map_location=device))
    head.load_state_dict(torch.load(os.path.join(args.save_dir, "task_head.pth"), map_location=device))

    for s in eval_snrs:
        a = evaluate_snr(
            encoder, decoder, head, dl_test, device,
            snr_db=float(s), pt=args.pt, eps=args.eps
        )
        print(f"SNR={s:>3} dB  acc={a:.4f}")


if __name__ == "__main__":
    main()