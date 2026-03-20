import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.semanticencoder import ImageSemanticEncoder, SemanticDecoder, TaskHead
from harq.softcombining import SoftCombiningConfig, build_soft_combiner
from transceiver.sender import Sender
from transceiver.receiver import Receiver, ReceiverConfig


def parse_snrs(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


@torch.no_grad()
def eval_clean_acc(encoder, decoder, head, dl, device, max_batches=None):
    encoder.eval()
    decoder.eval()
    head.eval()

    correct = 0
    total = 0
    for bi, (x, y) in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)

        z = encoder(x)
        feat = decoder(z)
        logits = head(feat)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return correct / max(total, 1)


@torch.no_grad()
def eval_harq_system(encoder, decoder, head, soft_combiner, dl, device, num_blocks, cfg_rx, max_batches=None):
    encoder.eval()
    decoder.eval()
    head.eval()
    soft_combiner.eval()

    tx = Sender(encoder=encoder, num_blocks=num_blocks).to(device)
    rx = Receiver(decoder=decoder, soft_combiner=soft_combiner, task_head=head, cfg=cfg_rx).to(device)
    tx.eval()
    rx.eval()

    correct = 0
    total = 0

    rounds_sum = 0.0
    blocks_retx_sum = 0.0
    batches = 0

    for bi, (x, y) in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        x_tx, xb_tx = tx(x)
        out = rx.run_harq(x_tx, xb_tx)

        logits = out["logits_final"]
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

        rounds_sum += float(out["rounds_used"].float().mean().item())
        blocks_retx_sum += float(out["blocks_retx_total"].float().mean().item())
        batches += 1

    acc = correct / max(total, 1)
    avg_rounds = rounds_sum / max(batches, 1)
    avg_retx_blocks = blocks_retx_sum / max(batches, 1)
    return acc, avg_rounds, avg_retx_blocks


def plot_and_save(x, y_list, labels, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    for y, lab in zip(y_list, labels):
        plt.plot(x, y, marker="o", label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/raw")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_noisy")
    parser.add_argument("--fig_dir", type=str, default="./figures_harq")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--use_cuda", action="store_true")

    # system
    parser.add_argument("--semantic_dim", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_blocks", type=int, default=8)

    # HARQ
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--min_rounds", type=int, default=1)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--ig_steps", type=int, default=16)

    # threshold
    parser.add_argument("--entropy_target", type=float, default=0.65)
    parser.add_argument("--entropy_hysteresis", type=float, default=0.05)
    parser.add_argument("--confidence_target", type=float, default=0.70)

    # block selection
    parser.add_argument("--select_alpha", type=float, default=1.0)
    parser.add_argument("--select_beta", type=float, default=1.0)

    # channel
    parser.add_argument("--pt", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-8)

    # reserved mapping params
    parser.add_argument("--snr_map_a", type=float, default=6.0)
    parser.add_argument("--snr_map_b", type=float, default=-2.0)

    parser.add_argument("--snrs", type=str, default="-5,0,5,10,15,20")
    args = parser.parse_args()

    if args.use_cuda and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds_test = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=tfm_test)
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    encoder = ImageSemanticEncoder(args.semantic_dim).to(device)
    decoder = SemanticDecoder(args.semantic_dim).to(device)
    head = TaskHead(args.semantic_dim, args.num_classes).to(device)
    soft_combiner = build_soft_combiner(SoftCombiningConfig(hidden_dim=64), dim=args.semantic_dim).to(device)

    enc_path = os.path.join(args.save_dir, "encoder.pth")
    dec_path = os.path.join(args.save_dir, "decoder.pth")
    head_path = os.path.join(args.save_dir, "task_head.pth")

    if not (os.path.exists(enc_path) and os.path.exists(dec_path) and os.path.exists(head_path)):
        raise FileNotFoundError(
            f"找不到 checkpoint：\n  {enc_path}\n  {dec_path}\n  {head_path}"
        )

    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    decoder.load_state_dict(torch.load(dec_path, map_location=device))
    head.load_state_dict(torch.load(head_path, map_location=device))

    print(f"✅ Loaded checkpoints:\n  {enc_path}\n  {dec_path}\n  {head_path}")

    clean_acc = eval_clean_acc(encoder, decoder, head, dl_test, device, max_batches=args.max_batches)
    print(f"Clean acc (no channel): {clean_acc:.4f}")

    snrs = parse_snrs(args.snrs)

    sys_accs = []
    avg_rounds_list = []
    avg_retx_list = []
    clean_line = []

    for snr_db in snrs:
        cfg_rx = ReceiverConfig(
            semantic_dim=args.semantic_dim,
            num_blocks=args.num_blocks,
            num_classes=args.num_classes,
            max_rounds=args.max_rounds,
            min_rounds=args.min_rounds,
            topk=args.topk,
            ig_steps=args.ig_steps,

            entropy_target=args.entropy_target,
            entropy_hysteresis=args.entropy_hysteresis,
            confidence_target=args.confidence_target,

            select_alpha=args.select_alpha,
            select_beta=args.select_beta,

            snr_db=snr_db,
            pt=args.pt,
            eps=args.eps,

            snr_map_a=args.snr_map_a,
            snr_map_b=args.snr_map_b,
        )

        sys_acc, avg_rounds, avg_retx = eval_harq_system(
            encoder=encoder,
            decoder=decoder,
            head=head,
            soft_combiner=soft_combiner,
            dl=dl_test,
            device=device,
            num_blocks=args.num_blocks,
            cfg_rx=cfg_rx,
            max_batches=args.max_batches,
        )

        sys_accs.append(sys_acc)
        avg_rounds_list.append(avg_rounds)
        avg_retx_list.append(avg_retx)
        clean_line.append(clean_acc)

        print(
            f"[SNR={snr_db:>5.1f} dB] "
            f"System acc={sys_acc:.4f} | Avg rounds={avg_rounds:.3f} | Avg retrans blocks={avg_retx:.3f}"
        )

    plot_and_save(
        snrs,
        [clean_line, sys_accs],
        ["Clean (no channel)", "System (HARQ)"],
        "Accuracy vs SNR",
        "SNR (dB)",
        "Accuracy",
        os.path.join(args.fig_dir, "acc_vs_snr.png"),
    )

    plot_and_save(
        snrs,
        [avg_rounds_list],
        ["Avg HARQ rounds"],
        "Avg HARQ Rounds vs SNR",
        "SNR (dB)",
        "Avg HARQ rounds",
        os.path.join(args.fig_dir, "rounds_vs_snr.png"),
    )

    plot_and_save(
        snrs,
        [avg_retx_list],
        ["Avg retrans blocks"],
        "Avg Retrans Blocks vs SNR",
        "SNR (dB)",
        "Avg retrans blocks",
        os.path.join(args.fig_dir, "retx_blocks_vs_snr.png"),
    )

    print(f"\n✅ Figures saved to: {args.fig_dir}/")
    print("   - acc_vs_snr.png")
    print("   - rounds_vs_snr.png")
    print("   - retx_blocks_vs_snr.png")


if __name__ == "__main__":
    main()