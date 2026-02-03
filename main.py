# main.py
import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from models.semanticencoder import ImageSemanticEncoder, SemanticDecoder, TaskHead
from harq.softcombining import SoftCombiningConfig, build_soft_combiner
from transceiver.sender import Sender
from transceiver.receiver import Receiver, ReceiverConfig
from utils.metrics import accuracy

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== CIFAR10 图像输入 =====
    tfm = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = datasets.CIFAR10(root="./data/raw", train=False, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    # ===== 系统超参 =====
    semantic_dim = 128
    num_blocks = 8
    num_classes = 10

    cfg_rx = ReceiverConfig(
        semantic_dim=semantic_dim,
        num_blocks=num_blocks,
        num_classes=num_classes,
        entropy_target=1.5,   # CIFAR10 10类，熵范围大一点，先给一个可触发的阈值
        max_rounds=4,
        snr_db=0.0,
        topk=4,
        ig_steps=16,
        snr_map_a=6.0,
        snr_map_b=-2.0
    )

    # ===== 模块 =====
    encoder = ImageSemanticEncoder(semantic_dim).to(device)
    decoder = SemanticDecoder(semantic_dim).to(device)
    task_head = TaskHead(semantic_dim, num_classes).to(device)

    soft_combiner = build_soft_combiner(SoftCombiningConfig(hidden_dim=64), dim=semantic_dim).to(device)

    tx = Sender(encoder=encoder, num_blocks=num_blocks).to(device)
    rx = Receiver(decoder=decoder, soft_combiner=soft_combiner, task_head=task_head, cfg=cfg_rx).to(device)

    # ===== 预运行 =====
    encoder.eval(); decoder.eval(); task_head.eval(); soft_combiner.eval()

    img, y = next(iter(dl))
    img, y = img.to(device), y.to(device)

    with torch.no_grad():
        x_tx, xb_tx = tx(img)

    out = rx.run_harq(x_tx, xb_tx)
    logits = out["logits_final"]

    print("=== System Run OK ===")
    print("Batch acc (untrained, just sanity):", accuracy(logits, y))
    print("Avg rounds used:", out["rounds_used"].mean().item())
    print("Avg retrans blocks:", out["blocks_retx_total"].mean().item())

if __name__ == "__main__":
    main()