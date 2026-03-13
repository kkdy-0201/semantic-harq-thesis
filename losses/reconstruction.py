import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ms_ssim
    HAS_MSSSIM = True
except Exception:
    HAS_MSSSIM = False


class ReconstructionLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.name = cfg["loss"]["name"].lower()
        self.lambda_l1 = float(cfg["loss"].get("lambda_l1", 1.0))
        self.lambda_msssim = float(cfg["loss"].get("lambda_msssim", 0.2))

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor):
        if self.name == "mse":
            loss = F.mse_loss(x_hat, x)
            stats = {"mse": float(loss.detach().item())}
            return loss, stats

        if self.name == "l1":
            loss = F.l1_loss(x_hat, x)
            stats = {"l1": float(loss.detach().item())}
            return loss, stats

        if self.name == "l1_msssim":
            if not HAS_MSSSIM:
                raise ImportError("Please install pytorch-msssim to use loss.name='l1_msssim'")
            l1 = F.l1_loss(x_hat, x)
            ms = 1.0 - ms_ssim(x_hat, x, data_range=1.0, size_average=True)
            loss = self.lambda_l1 * l1 + self.lambda_msssim * ms
            stats = {"l1": float(l1.detach().item()), "1-ms_ssim": float(ms.detach().item())}
            return loss, stats

        raise ValueError(f"Unsupported loss.name: {self.name}")


@torch.no_grad()
def psnr(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    mse = F.mse_loss(x_hat, x).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()