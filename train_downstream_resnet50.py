import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.config import load_config, set_seed, ensure_dir, get_device
from models.downstream_resnet import build_resnet50_cifar10


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = get_device(cfg)

    data_cfg = cfg["dataset"]
    train_cfg = cfg["downstream"]
    ensure_dir(train_cfg["save_dir"])

    tfm_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tfm_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds_train = datasets.CIFAR10(
        root=data_cfg["root"], train=True,
        download=bool(data_cfg["download"]), transform=tfm_train
    )
    ds_test = datasets.CIFAR10(
        root=data_cfg["root"], train=False,
        download=bool(data_cfg["download"]), transform=tfm_test
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

    model = build_resnet50_cifar10(
        pretrained=bool(train_cfg["pretrained"]),
        num_classes=int(data_cfg["num_classes"])
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(train_cfg["epochs"])
    )

    best_acc = -1.0
    best_epoch = -1

    for ep in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        steps = 0

        for x, y in dl_train:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            steps += 1

        scheduler.step()
        acc = evaluate(model, dl_test, device)
        print(f"[Epoch {ep:03d}/{train_cfg['epochs']}] loss={loss_sum/max(steps,1):.6f} test_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_epoch = ep
            torch.save(model.state_dict(), os.path.join(train_cfg["save_dir"], "resnet50_cifar10.pth"))
            print(f"Saved best classifier @ epoch {ep}, acc={best_acc:.4f}")

    print(f"Done. Best classifier acc={best_acc:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()