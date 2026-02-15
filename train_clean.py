# train_clean.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from models.semanticencoder import ImageSemanticEncoder, TaskHead
from utils.seed import set_seed


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def evaluate(encoder: nn.Module, head: nn.Module, dl: DataLoader, device: str) -> float:
    encoder.eval()
    head.eval()
    acc_sum = 0.0
    n = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        logits = head(encoder(x))
        acc_sum += accuracy_from_logits(logits, y)
        n += 1
    return acc_sum / max(n, 1)


def train_one_epoch(
    encoder: nn.Module,
    head: nn.Module,
    dl: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    encoder.train()
    head.train()

    ce = nn.CrossEntropyLoss()
    loss_sum = 0.0
    n = 0

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = head(encoder(x))
        loss = ce(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        n += 1

    return loss_sum / max(n, 1)


@torch.no_grad()
def visualize_predictions(
    encoder: nn.Module,
    head: nn.Module,
    dl_test: DataLoader,
    device: str,
    save_path: str,
    num_images: int = 16,
):
    """
    从测试集取一批图，展示 GT / Pred。
    注意：训练时我们对图像做了 Normalize，这里为了显示，需要反归一化。
    """
    encoder.eval()
    head.eval()

    x, y = next(iter(dl_test))
    x = x.to(device)
    y = y.to(device)

    logits = head(encoder(x))
    pred = logits.argmax(dim=1)

    # 只展示前 num_images 张
    n = min(num_images, x.shape[0])
    x = x[:n].detach().cpu()
    y = y[:n].detach().cpu()
    pred = pred[:n].detach().cpu()

    # CIFAR10 mean/std
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    # 反归一化用于显示
    x_vis = x * std + mean
    x_vis = x_vis.clamp(0, 1)

    cols = 4
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img = x_vis[i].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.title(f"GT:{y[i].item()} Pred:{pred[i].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_curves(train_losses, test_accs, save_path: str):
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_loss.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, test_accs)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Curve")
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_acc.png"), dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--semantic_dim", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--data_root", type=str, default="./data/raw")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--fig_dir", type=str, default="./figures")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_cuda", action="store_true", help="force using cuda if available")
    args = parser.parse_args()

    set_seed(args.seed)

    # device
    if args.use_cuda and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

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
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    encoder = ImageSemanticEncoder(args.semantic_dim).to(device)
    head = TaskHead(args.semantic_dim, args.num_classes).to(device)

    optimizer = optim.SGD(
        list(encoder.parameters()) + list(head.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_epoch = -1

    train_losses = []
    test_accs = []

    print(f"Train samples: {len(ds_train)}, Test samples: {len(ds_test)}")
    print(
        f"epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, "
        f"mom={args.momentum}, wd={args.weight_decay}, semantic_dim={args.semantic_dim}"
    )

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(encoder, head, dl_train, optimizer, device)
        test_acc = evaluate(encoder, head, dl_test, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        train_losses.append(loss)
        test_accs.append(test_acc)

        print(f"[Epoch {epoch:03d}/{args.epochs}] loss={loss:.4f}  test_acc={test_acc:.4f}  lr={lr_now:.6f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder.pth"))
            torch.save(head.state_dict(), os.path.join(args.save_dir, "task_head.pth"))
            print(f"✅ Saved best checkpoint (acc={best_acc:.4f})")

    print(f"\nTraining done. Best test acc: {best_acc:.4f} @ epoch {best_epoch}")
    print(f"Checkpoints saved to: {args.save_dir}/encoder.pth and {args.save_dir}/task_head.pth")

    # ====== 可视化：曲线 + 示例预测 ======
    curves_path = os.path.join(args.fig_dir, "train_curves.png")
    plot_curves(train_losses, test_accs, curves_path)

    pred_vis_path = os.path.join(args.fig_dir, "pred_grid.png")
    visualize_predictions(encoder, head, dl_test, device, save_path=pred_vis_path, num_images=16)

    print(f"✅ Saved figures to: {args.fig_dir}/")


if __name__ == "__main__":
    main()
