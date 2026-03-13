import math
import random
import torch


def sample_snr_db(channel_cfg: dict) -> float:
    mode = channel_cfg.get("snr_mode", "fixed")
    if mode == "fixed":
        return float(channel_cfg["snr_db"])
    if mode == "uniform":
        return random.uniform(float(channel_cfg["snr_min"]), float(channel_cfg["snr_max"]))
    raise ValueError(f"Unsupported snr_mode: {mode}")


def semantic_channel_real(z: torch.Tensor, snr_db: float, pt: float = 1.0, eps: float = 1e-8):
    """
    y = sqrt(Pt) * z / sqrt(mean(z^2) + eps) + n
    n ~ N(0, Pt/gamma * I)
    """
    dims = tuple(range(1, z.dim()))
    pz = torch.mean(z.pow(2), dim=dims, keepdim=True)
    z_tx = math.sqrt(pt) * z / torch.sqrt(pz + eps)

    gamma = 10.0 ** (snr_db / 10.0)
    noise_var = pt / gamma
    noise_std = math.sqrt(noise_var)

    noise = noise_std * torch.randn_like(z_tx)
    return z_tx + noise