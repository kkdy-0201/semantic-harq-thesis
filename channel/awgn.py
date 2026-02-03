# channel/awgn.py
import torch

def awgn(x: torch.Tensor, snr_db: float, assume_unit_power: bool = True) -> torch.Tensor:
    """
    AWGN: y = x + n
    若 assume_unit_power=True，则假设 x 的每个样本平均功率≈1，否则按样本功率自适应噪声。
    Args:
        x: [B, D] float tensor
        snr_db: float
    """
    if x.dtype not in (torch.float16, torch.float32, torch.float64):
        x = x.float()

    snr_lin = 10.0 ** (snr_db / 10.0)

    if assume_unit_power:
        noise_var = 1.0 / snr_lin
        noise = torch.randn_like(x) * (noise_var ** 0.5)
        return x + noise

    # 自适应：按每个样本的功率估噪声方差
    p = x.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-12)
    noise_var = p / snr_lin
    noise = torch.randn_like(x) * noise_var.sqrt()
    return x + noise