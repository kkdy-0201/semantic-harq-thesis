import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class SCEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, latent_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            ResBlock(base_channels),

            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(base_channels * 2),

            nn.Conv2d(base_channels * 2, latent_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(latent_channels),
        )

    def forward(self, x):
        return self.net(x)


class SCDecoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=64, latent_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(latent_channels),

            nn.ConvTranspose2d(latent_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(base_channels * 2),

            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(base_channels),

            nn.Conv2d(base_channels, out_channels, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)