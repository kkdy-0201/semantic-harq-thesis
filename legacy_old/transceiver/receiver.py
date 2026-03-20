from dataclasses import dataclass

import torch
import torch.nn as nn

from channel.semantic_channel import semantic_channel_real
from harq.threshold import (
    split_blocks,
    merge_blocks,
    integrated_gradients_blockwise,
    BlockSelector,
    BlockSelectorConfig,
)


@dataclass
class ReceiverConfig:
    semantic_dim: int = 128
    num_blocks: int = 8
    num_classes: int = 10

    # HARQ
    max_rounds: int = 4
    min_rounds: int = 1
    topk: int = 4
    ig_steps: int = 16

    # 停止阈值
    entropy_target: float = 0.65
    entropy_hysteresis: float = 0.05
    confidence_target: float = 0.70

    # 块选择打分
    select_alpha: float = 1.0
    select_beta: float = 1.0

    # 信道
    snr_db: float = 10.0
    pt: float = 1.0
    eps: float = 1e-8

    # 保留原来的映射参数
    snr_map_a: float = 6.0
    snr_map_b: float = -2.0


class Receiver(nn.Module):
    def __init__(self, decoder, soft_combiner, task_head, cfg: ReceiverConfig):
        super().__init__()
        self.decoder = decoder
        self.soft_combiner = soft_combiner
        self.task_head = task_head
        self.cfg = cfg

        self.selector = BlockSelector(
            BlockSelectorConfig(
                alpha=cfg.select_alpha,
                beta=cfg.select_beta,
                topk=cfg.topk,
            )
        )

    def entropy_from_logits(self, logits: torch.Tensor):
        p = torch.softmax(logits, dim=1)
        ent = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=1)
        conf = p.max(dim=1).values
        pred = p.argmax(dim=1)
        return ent, conf, pred

    def should_stop(self, entropy, confidence, rounds_used):

        enough_rounds = rounds_used >= self.cfg.min_rounds
        entropy_ok = entropy <= (self.cfg.entropy_target - self.cfg.entropy_hysteresis)
        conf_ok = confidence >= self.cfg.confidence_target
        return enough_rounds & entropy_ok & conf_ok

    def run_harq(self, x_tx: torch.Tensor, xb_tx: torch.Tensor):
        device = x_tx.device
        B, D = x_tx.shape
        K = self.cfg.num_blocks

        # 第1轮：全量发送
        y0 = semantic_channel_real(
            x_tx, snr_db=self.cfg.snr_db, pt=self.cfg.pt, eps=self.cfg.eps
        )
        z_cur = y0

        rounds_used = torch.ones(B, device=device)
        block_rounds = torch.ones(B, K, device=device)   # 每块已接收轮数
        blocks_retx_total = torch.zeros(B, device=device)

        logits = self.task_head(self.decoder(z_cur))
        entropy, confidence, pred_class = self.entropy_from_logits(logits)

        done = self.should_stop(entropy, confidence, rounds_used)

        # 后续重传
        for r in range(2, self.cfg.max_rounds + 1):
            if bool(done.all()):
                break

            # 当前 z 分块
            zb_cur = split_blocks(z_cur, K)

            # 假设所有块若再收一轮的候选版本
            y_candidate_full = semantic_channel_real(
                x_tx, snr_db=self.cfg.snr_db, pt=self.cfg.pt, eps=self.cfg.eps
            )
            zb_candidate_full = split_blocks(y_candidate_full, K)

            # 计算 block-wise IG 贡献
            contrib = integrated_gradients_blockwise(
                task_head=self.task_head,
                decoder=self.decoder,
                z_prev=z_cur,
                z_candidate=y_candidate_full,
                target_class=pred_class,
                num_blocks=K,
                steps=self.cfg.ig_steps,
            )

            # 当前每块累计SNR
            gamma = 10.0 ** (self.cfg.snr_db / 10.0)
            gamma_blk = block_rounds * gamma
            snr_block_db = 10.0 * torch.log10(gamma_blk.clamp_min(1e-8))

            # 块选择
            topk_idx, score = self.selector.select(contrib, snr_block_db)

            # 对未完成样本执行重传
            for b in range(B):
                if done[b]:
                    continue

                idx = topk_idx[b]  # [topk]

                # 仅对选中的块加一轮新观测
                block_new_obs = semantic_channel_real(
                    xb_tx[b:b+1, idx, :],
                    snr_db=self.cfg.snr_db,
                    pt=self.cfg.pt,
                    eps=self.cfg.eps,
                )  # [1, topk, bd]

                # 软合并：仅更新这些块
                old_blocks = zb_cur[b:b+1, idx, :]
                new_blocks = self.soft_combiner(old_blocks, block_new_obs)

                zb_cur[b:b+1, idx, :] = new_blocks
                block_rounds[b, idx] += 1.0
                blocks_retx_total[b] += float(idx.numel())

            z_cur = merge_blocks(zb_cur)

            rounds_used = torch.where(done, rounds_used, torch.full_like(rounds_used, float(r)))

            logits = self.task_head(self.decoder(z_cur))
            entropy, confidence, pred_class = self.entropy_from_logits(logits)

            done = done | self.should_stop(entropy, confidence, rounds_used)

        out = {
            "z_final": z_cur,
            "logits_final": logits,
            "entropy_final": entropy,
            "confidence_final": confidence,
            "rounds_used": rounds_used,
            "blocks_retx_total": blocks_retx_total,
        }
        return out