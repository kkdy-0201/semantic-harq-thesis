# utils/metrics.py
import torch

def softmax_entropy(logits: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    Shannon entropy of softmax distribution.
    logits: [B, C]
    returns: [B]
    """
    p = torch.softmax(logits, dim=dim)
    p = torch.clamp(p, min=eps, max=1.0)
    return -(p * torch.log(p)).sum(dim=dim)

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()