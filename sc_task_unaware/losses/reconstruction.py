import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def psnr(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    mse = F.mse_loss(x_hat, x).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


class CompositeSCLoss(nn.Module):
    """
    总损失:
        L = lambda_rec * L_rec + lambda_stru * L_stru

    其中:
        L_stru = sum_l alpha_l * d_feat(f^l(x_hat), f^l(x))
    """

    def __init__(self, cfg: dict, structure_extractor: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.structure_extractor = structure_extractor

        loss_cfg = cfg["loss"]
        stru_cfg = cfg["structure_model"]

        self.rec_mode = loss_cfg["rec_mode"].lower()
        self.lambda_rec = float(loss_cfg["lambda_rec"])
        self.lambda_stru = float(loss_cfg["lambda_stru"])

        self.layer_ids = list(stru_cfg["layer_ids"])
        self.layer_weights = [float(v) for v in stru_cfg["layer_weights"]]
        self.feat_distance = stru_cfg.get("feat_distance", "l1").lower()
        self.normalize_feats = bool(stru_cfg.get("normalize_feats", True))

        assert len(self.layer_ids) == len(self.layer_weights), "layer_ids 和 layer_weights 长度必须一致"

    def reconstruction_loss(self, x_hat: torch.Tensor, x: torch.Tensor):
        if self.rec_mode == "mse":
            return F.mse_loss(x_hat, x)
        if self.rec_mode == "l1":
            return F.l1_loss(x_hat, x)
        raise ValueError(f"Unsupported rec_mode: {self.rec_mode}")

    def feature_distance(self, a: torch.Tensor, b: torch.Tensor):
        if self.normalize_feats:
            a = F.normalize(a, p=2, dim=1)
            b = F.normalize(b, p=2, dim=1)

        if self.feat_distance == "l1":
            return F.l1_loss(a, b)
        if self.feat_distance == "l2":
            return F.mse_loss(a, b)
        raise ValueError(f"Unsupported feat_distance: {self.feat_distance}")

    def structure_loss(self, x_hat: torch.Tensor, x: torch.Tensor):
        # 原图分支固定为参照，不需要保留梯度
        with torch.no_grad():
            feat_ref = self.structure_extractor(x, self.layer_ids)

        # 重构图分支需要把梯度传回 x_hat -> decoder -> encoder
        feat_hat = self.structure_extractor(x_hat, self.layer_ids)

        total = 0.0
        layer_stats = {}

        for i, (fh, fr, alpha) in enumerate(zip(feat_hat, feat_ref, self.layer_weights)):
            d = self.feature_distance(fh, fr)
            total = total + alpha * d
            layer_stats[f"stru_l{i}"] = float(d.detach().item())

        return total, layer_stats

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor):
        l_rec = self.reconstruction_loss(x_hat, x)
        l_stru, layer_stats = self.structure_loss(x_hat, x)

        total = self.lambda_rec * l_rec + self.lambda_stru * l_stru

        stats = {
            "loss_total": float(total.detach().item()),
            "loss_rec": float(l_rec.detach().item()),
            "loss_stru": float(l_stru.detach().item()),
        }
        stats.update(layer_stats)
        return total, stats