# models/semanticencoder.py
import torch
import torch.nn as nn

class ImageSemanticEncoder(nn.Module):
    """
    图像 -> 语义向量 x_tx: [B, semantic_dim]
    默认适配 CIFAR10 3x32x32
    """
    def __init__(self, semantic_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, semantic_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        h = self.backbone(img).flatten(1)   # [B,128]
        x = self.fc(h)                      # [B,D]
        # 单样本功率归一：便于 AWGN 使用 unit power
        x = x / (x.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)
        return x

class SemanticDecoder(nn.Module):
    """
    受噪语义向量 -> 语义特征 z
    """
    def __init__(self, semantic_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)

class TaskHead(nn.Module):
    """
    语义特征 -> 分类logits
    """
    def __init__(self, semantic_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim),
            nn.ReLU(),
            nn.Linear(semantic_dim, num_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)