import math
import torch


def semantic_channel_real(z: torch.Tensor, snr_db: float, pt: float = 1.0, eps: float = 1e-8):
    """
    实值语义信道：
        y = sqrt(Pt) * z / sqrt((1/d)||z||^2 + eps) + n
        n ~ N(0, Pt/gamma * I)

    参数:
        z: [B, d] 或更高维张量
        snr_db: 信噪比(dB)
        pt: 发射平均功率 Pt
        eps: 数值稳定项

    返回:
        y: 信道输出
    """
    dims = tuple(range(1, z.dim()))
    pz = torch.mean(z.pow(2), dim=dims, keepdim=True)   # 每个样本平均功率
    z_tx = math.sqrt(pt) * z / torch.sqrt(pz + eps)

    gamma = 10.0 ** (snr_db / 10.0)
    noise_var = pt / gamma
    noise_std = math.sqrt(noise_var)

    n = noise_std * torch.randn_like(z_tx)
    y = z_tx + n
    return y


def effective_snr_from_rounds(snr_db: float, rounds: torch.Tensor):

    gamma = 10.0 ** (snr_db / 10.0)
    gamma_eff = rounds.float().clamp_min(1.0) * gamma
    snr_eff_db = 10.0 * torch.log10(gamma_eff)
    return snr_eff_db