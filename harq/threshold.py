# harq/threshold.py
from dataclasses import dataclass
import torch
import torch.nn as nn

def split_blocks(x: torch.Tensor, num_blocks: int) -> torch.Tensor:
    B, D = x.shape
    assert D % num_blocks == 0, "semantic_dim must be divisible by num_blocks"
    return x.view(B, num_blocks, D // num_blocks)

def merge_blocks(xb: torch.Tensor) -> torch.Tensor:
    B, nb, bd = xb.shape
    return xb.reshape(B, nb * bd)

def reliability_from_snr_db(snr_db_blocks: torch.Tensor, alpha: float = 0.6, mid_db: float = 0.0) -> torch.Tensor:
    """
    单调映射：SNR越高可靠度越大
    snr_db_blocks: [B, nb]
    """
    return torch.sigmoid(alpha * (snr_db_blocks - mid_db))

def integrated_gradients_blockwise(task_model: nn.Module,
                                   z_from: torch.Tensor,
                                   z_to: torch.Tensor,
                                   target_idx: torch.Tensor,
                                   num_blocks: int,
                                   steps: int = 16) -> torch.Tensor:
    """
    Block-wise Integrated Gradients attribution from z_from -> z_to.
    返回每个块的贡献度（非负）。
    """
    assert z_from.shape == z_to.shape
    B, D = z_from.shape
    delta = (z_to - z_from).detach()
    grads_acc = torch.zeros_like(z_from)

    for i in range(1, steps + 1):
        alpha = i / steps
        z = (z_from + alpha * delta).detach().requires_grad_(True)
        logits = task_model(z)  # [B,C]
        f = logits.gather(1, target_idx.view(-1, 1)).sum()
        g = torch.autograd.grad(f, z, retain_graph=False, create_graph=False)[0]
        grads_acc += g.detach()

    avg_grads = grads_acc / steps
    attr = (delta * avg_grads).abs()                      # [B,D]
    contrib = split_blocks(attr, num_blocks).sum(dim=-1)  # [B,nb]
    return contrib

@dataclass
class BlockSelectorConfig:
    num_blocks: int
    topk: int
    ig_steps: int = 16
    rel_alpha: float = 0.6
    rel_mid_db: float = 0.0

class BlockSelector:
    """
    value = reliability(SNR) * contribution(IG)
    """
    def __init__(self, cfg: BlockSelectorConfig):
        self.cfg = cfg

    def score(self,
              task_model: nn.Module,
              z_prev: torch.Tensor,
              z_new: torch.Tensor,
              logits_new: torch.Tensor,
              snr_db_blocks: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            target = logits_new.detach().argmax(dim=1)  # [B]
            contrib = integrated_gradients_blockwise(
                task_model=task_model,
                z_from=z_prev,
                z_to=z_new,
                target_idx=target,
                num_blocks=self.cfg.num_blocks,
                steps=self.cfg.ig_steps
            )
        rel = reliability_from_snr_db(snr_db_blocks, self.cfg.rel_alpha, self.cfg.rel_mid_db)
        return rel * contrib

    @torch.no_grad()
    def select_topk_mask(self, value: torch.Tensor) -> torch.Tensor:
        B, nb = value.shape
        k = min(self.cfg.topk, nb)
        idx = torch.topk(value, k=k, dim=1).indices
        mask = torch.zeros_like(value, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask