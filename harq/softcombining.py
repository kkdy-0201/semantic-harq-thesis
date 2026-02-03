# harq/softcombining.py
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class SoftCombiningConfig:
    hidden_dim: int = 64

class GatedSoftCombiner(nn.Module):
    """
    语义软合并：
      z_new = gate * z_old + (1 - gate) * z_inc
    gate由 z_old 与 snr_eff 决定，snr越高 -> 越信任新信息（gate更小）
    """
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid()
        )

    def forward(self, z_old: torch.Tensor, z_inc: torch.Tensor, snr_eff_db: torch.Tensor) -> torch.Tensor:
        # snr_eff_db: [B]
        snr_feat = snr_eff_db.view(-1, 1)  # [B,1]
        gate = self.net(torch.cat([z_old, snr_feat], dim=1))  # [B,D]
        # snr越高希望更多引入新信息：让 gate 随 snr 增大而减小（可选增强）
        # 这里不额外改gate，保持稳定可训练性
        return gate * z_old + (1.0 - gate) * z_inc

def build_soft_combiner(cfg: SoftCombiningConfig, dim: int) -> nn.Module:
    return GatedSoftCombiner(dim=dim, hidden_dim=cfg.hidden_dim)