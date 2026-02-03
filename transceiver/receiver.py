# transceiver/receiver.py
from dataclasses import dataclass
import torch
import torch.nn as nn

from channel.awgn import awgn
from utils.metrics import softmax_entropy
from harq.threshold import merge_blocks, BlockSelector, BlockSelectorConfig

@dataclass
class ReceiverConfig:
    semantic_dim: int
    num_blocks: int
    num_classes: int
    entropy_target: float
    max_rounds: int
    snr_db: float
    topk: int
    ig_steps: int

    # 熵 -> 所需SNR (dB) 的简单标定（可替换为查表/拟合）
    snr_map_a: float = 6.0
    snr_map_b: float = -2.0

class Receiver(nn.Module):
    def __init__(self, decoder: nn.Module, soft_combiner: nn.Module, task_head: nn.Module, cfg: ReceiverConfig):
        super().__init__()
        self.decoder = decoder
        self.soft_combiner = soft_combiner
        self.task_head = task_head
        self.cfg = cfg

        self.selector = BlockSelector(BlockSelectorConfig(
            num_blocks=cfg.num_blocks,
            topk=cfg.topk,
            ig_steps=cfg.ig_steps
        ))

    def _snr_lin(self, snr_db: float) -> float:
        return 10.0 ** (snr_db / 10.0)

    def snr_required_db(self, entropy: torch.Tensor) -> torch.Tensor:
        # 简单线性映射：熵越大需要更高SNR
        return self.cfg.snr_map_a * entropy + self.cfg.snr_map_b

    def run_harq(self, x_tx: torch.Tensor, xb_tx: torch.Tensor):
        device = x_tx.device
        B, D = x_tx.shape
        nb = self.cfg.num_blocks

        snr_lin_per_tx = self._snr_lin(self.cfg.snr_db)

        # per-block 累计SNR（线性叠加）
        snr_acc_lin = torch.full((B, nb), snr_lin_per_tx, device=device)
        snr_acc_db = 10.0 * torch.log10(torch.clamp(snr_acc_lin, min=1e-12))

        # round 0: 全量发送一次
        y0 = awgn(x_tx, self.cfg.snr_db, assume_unit_power=True)
        z_old = self.decoder(y0)

        rounds_used = torch.ones(B, device=device)
        blocks_retx_total = torch.zeros(B, device=device)

        for r in range(1, self.cfg.max_rounds + 1):
            logits_old = self.task_head(z_old)
            ent = softmax_entropy(logits_old)  # [B]
            snr_eff_db_global = snr_acc_db.mean(dim=1)  # [B]

            # 触发判据（更严谨版本）：熵>目标 且 当前SNR低于“熵映射的所需SNR”
            if r == self.cfg.max_rounds:
                decision = torch.zeros_like(ent, dtype=torch.bool)
            else:
                snr_req = self.snr_required_db(ent)
                decision = (ent > self.cfg.entropy_target) & (snr_eff_db_global < snr_req)

            if not decision.any():
                break

            # 用“全量重传候选更新”评估IG贡献（用于块选择）
            with torch.no_grad():
                y_full = awgn(x_tx, self.cfg.snr_db, assume_unit_power=True)
                z_inc_full = self.decoder(y_full)
                z_cand = self.soft_combiner(z_old, z_inc_full, snr_eff_db_global)

            value = self.selector.score(
                task_model=self.task_head,
                z_prev=z_old,
                z_new=z_cand,
                logits_new=self.task_head(z_cand).detach(),
                snr_db_blocks=snr_acc_db
            )
            mask = self.selector.select_topk_mask(value)      # [B,nb]
            mask = mask & decision.view(-1, 1)               # 仅对需要重传的样本生效

            blocks_retx_total += mask.float().sum(dim=1)
            rounds_used[decision] = r + 1

            # 只重传 top-k 块
            xb_retx = xb_tx.clone()
            xb_retx[~mask] = 0.0
            x_retx = merge_blocks(xb_retx)

            y = awgn(x_retx, self.cfg.snr_db, assume_unit_power=True)
            z_inc = self.decoder(y)

            # 更新累计SNR（仅对重传块）
            snr_acc_lin = snr_acc_lin + mask.float() * snr_lin_per_tx
            snr_acc_db = 10.0 * torch.log10(torch.clamp(snr_acc_lin, min=1e-12))

            # 软合并更新
            z_old = self.soft_combiner(z_old, z_inc, snr_eff_db_global)

        logits_final = self.task_head(z_old)
        return {
            "logits_final": logits_final,
            "rounds_used": rounds_used,
            "blocks_retx_total": blocks_retx_total
        }