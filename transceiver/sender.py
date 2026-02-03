# transceiver/sender.py
import torch
import torch.nn as nn
from harq.threshold import split_blocks

class Sender(nn.Module):
    def __init__(self, encoder: nn.Module, num_blocks: int):
        super().__init__()
        self.encoder = encoder
        self.num_blocks = num_blocks

    @torch.no_grad()
    def forward(self, img: torch.Tensor):
        x_tx = self.encoder(img)  # [B,D]
        xb_tx = split_blocks(x_tx, self.num_blocks)
        return x_tx, xb_tx