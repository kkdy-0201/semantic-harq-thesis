# train.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows OpenMP 冲突兜底

import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.semanticencoder import ImageSemanticEncoder, TaskHead

@torch.no_grad()
def evaluate(encoder, head, dl, device):
    encoder.eval()
    head.eval()
    correct, total = 0, 0
    for img, y in dl:
        img, y = img.to(device), y.to(device)
        x = encoder(img)          # [B, D]
        logits = head(x)          # [B, C]
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/raw")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--semantic_dim", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # CIFAR-10
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=tfm_train)
    test_ds  = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=tfm_test)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    encoder = ImageSemanticEncoder(args.semantic_dim).to(device)
    head = TaskHead(args.semantic_dim, args.num_classes).to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        head.train()

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}")
        for img, y in pbar:
            img, y = img.to(device), y.to(device)
            x = encoder(img)
            logits = head(x)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        acc = evaluate(encoder, head, test_dl, device)
        print(f"[Epoch {epoch}] test_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(encoder.state_dict(), os.path.join(args.save_dir, "encoder.pth"))
            torch.save(head.state_dict(), os.path.join(args.save_dir, "task_head.pth"))
            print(f"✅ Saved best checkpoint (acc={best_acc:.4f})")

    print("Training done. Best test acc:", best_acc)

if __name__ == "__main__":
    main()
