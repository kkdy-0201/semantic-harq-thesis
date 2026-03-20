import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class DownsampleBlock(nn.Module):
    """
    你要求的结构：
    ResNet + 降采样卷积 + ResNet
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pre = ResBlock(in_ch)
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.post = ResBlock(out_ch)

    def forward(self, x):
        x = self.pre(x)
        skip = x
        x = self.down(x)
        x = self.act(x)
        x = self.post(x)
        return x, skip


class UpsampleBlock(nn.Module):
    """
    UpConv -> concat skip -> fuse conv -> ResBlock
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.post = ResBlock(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.act(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.post(x)
        return x


class SCUNetEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, latent_channels=256):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(base_channels)
        )

        self.down1 = DownsampleBlock(base_channels, base_channels * 2)    # 32 -> 16
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4)  # 16 -> 8
        self.down3 = DownsampleBlock(base_channels * 4, latent_channels)   # 8 -> 4

        self.bottleneck = ResBlock(latent_channels)

    def forward(self, x):
        x0 = self.stem(x)              # [B, 64, 32, 32]
        x1, skip1 = self.down1(x0)     # [B, 128, 16, 16], skip [B, 64, 32, 32]
        x2, skip2 = self.down2(x1)     # [B, 256, 8, 8],  skip [B, 128, 16, 16]
        x3, skip3 = self.down3(x2)     # [B, latent, 4,4], skip [B,256,8,8]

        z = self.bottleneck(x3)

        skips = {
            "skip1": skip1,
            "skip2": skip2,
            "skip3": skip3
        }
        return z, skips


class SCUNetDecoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=64, latent_channels=256):
        super().__init__()

        self.up1 = UpsampleBlock(latent_channels, base_channels * 4, base_channels * 4)   # 4 -> 8
        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2, base_channels * 2) # 8 -> 16
        self.up3 = UpsampleBlock(base_channels * 2, base_channels, base_channels)          # 16 -> 32

        self.out_head = nn.Sequential(
            ResBlock(base_channels),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, skips):
        x = self.up1(z, skips["skip3"])
        x = self.up2(x, skips["skip2"])
        x = self.up3(x, skips["skip1"])
        x_hat = self.out_head(x)
        return x_hat