import math
from dataclasses import dataclass

import torch

def split_blocks(z: torch.Tensor, num_blocks: int):
    """
    z: [B, D]
    return: [B, K, block_dim]
    """
    B, D = z.shape
    assert D % num_blocks == 0, f"semantic_dim={D} 必须能被 num_blocks={num_blocks} 整除"
    bd = D // num_blocks
    return z.view(B, num_blocks, bd)

def merge_blocks(zb: torch.Tensor):
    """
    zb: [B, K, block_dim]
    return: [B, D]
    """
    B, K, bd = zb.shape
    return zb.reshape(B, K * bd)

def reliability_from_snr_db(snr_db_tensor: torch.Tensor):
    return torch.sigmoid((snr_db_tensor - 0.0) / 3.0)

def integrated_gradients_blockwise(
    task_head,
    decoder,
    z_prev: torch.Tensor,
    z_candidate: torch.Tensor,
    target_class: torch.Tensor,
    num_blocks: int,
    steps: int = 16,
):
    """
    计算从 z_prev -> z_candidate 的 block-level integrated gradients importance
    返回: [B, num_blocks]
    """
    device = z_prev.device
    B, D = z_prev.shape
    alphas = torch.linspace(0.0, 1.0, steps, device=device)

    total_grads = torch.zeros_like(z_prev)

    for a in alphas:
        z_interp = z_prev + a * (z_candidate - z_prev)
        z_interp.requires_grad_(True)

        feat = decoder(z_interp)
        logits = task_head(feat)

        score = logits.gather(1, target_class.view(-1, 1)).sum()
        grads = torch.autograd.grad(score, z_interp, retain_graph=False, create_graph=False)[0]
        total_grads += grads.detach()

    avg_grads = total_grads / float(steps)
    ig = (z_candidate - z_prev) * avg_grads   # [B, D]

    ig_blocks = split_blocks(ig, num_blocks)  # [B, K, bd]
    contrib = ig_blocks.abs().sum(dim=-1)     # [B, K]
    return contrib


@dataclass
class BlockSelectorConfig:
    alpha: float = 1.0   # 不可靠度权重
    beta: float = 1.0    # 贡献度权重
    topk: int = 4


class BlockSelector:
    def __init__(self, cfg: BlockSelectorConfig):
        self.cfg = cfg

    def score_blocks(self, contrib: torch.Tensor, snr_block_db: torch.Tensor):
        """
        contrib: [B, K]  块贡献
        snr_block_db: [B, K]  块累计SNR(dB)
        """
        reliability = reliability_from_snr_db(snr_block_db).clamp(1e-6, 1.0 - 1e-6)
        deficiency = 1.0 - reliability

        score = (deficiency ** self.cfg.alpha) * (contrib.clamp_min(1e-8) ** self.cfg.beta)
        return score

    def select(self, contrib: torch.Tensor, snr_block_db: torch.Tensor):
        """
        返回:
            topk_idx: [B, topk]
            score: [B, K]
        """
        score = self.score_blocks(contrib, snr_block_db)
        topk = min(self.cfg.topk, score.shape[1])
        topk_idx = torch.topk(score, k=topk, dim=1).indices
        return topk_idx, score