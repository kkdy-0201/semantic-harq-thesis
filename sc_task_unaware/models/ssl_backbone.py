import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenDINOv2FeatureExtractor(nn.Module):
    """
    冻结的 DINOv2 backbone，仅用于提取结构特征，不参与训练。
    默认用 torch.hub 从官方仓库加载:
        torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    """

    def __init__(self, model_name: str = "dinov2_vits14", img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        self.model_name = model_name

        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone.eval()

        for p in self.backbone.parameters():
            p.requires_grad = False

        # DINOv2 / ImageNet 风格预处理
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False
        )
        x = (x - self.mean) / self.std
        return x

    def _tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, C]
        -> [B, C, H, W]
        """
        b, n, c = tokens.shape
        h = int(math.sqrt(n))
        w = h
        assert h * w == n, f"Patch token number {n} 不能还原成方形 feature map"
        feat = tokens.permute(0, 2, 1).contiguous().view(b, c, h, w)
        return feat

    def forward(self, x: torch.Tensor, layer_ids):
        """
        返回指定中间层的 feature maps 列表
        """
        x = self.preprocess(x)

        # 官方 DINOv2 backbone 支持 get_intermediate_layers
        outs = self.backbone.get_intermediate_layers(x, layer_ids)

        feat_maps = []
        for o in outs:
            # 兼容某些版本可能返回 tuple
            if isinstance(o, (tuple, list)):
                o = o[0]
            feat_maps.append(self._tokens_to_map(o))
        return feat_maps