import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

class ResidualBlock(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    time_embedding_channels: Optional[int] = None,
    eps: float = 1e-6,
  ) -> None:
    super().__init__()
    self.norm1 = nn.GroupNorm(32, in_channels, eps=eps)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.norm2 = nn.GroupNorm(32, out_channels, eps=eps)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    self.non_linearity = nn.SiLU()

    self.time_emb_proj = nn.Linear(time_embedding_channels, out_channels) if time_embedding_channels is not None else None
    self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else None

  def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
    residual = x
    x = self.norm1(x)
    x = self.non_linearity(x)
    x = self.conv1(x)

    if self.time_emb_proj is not None and time_embedding is not None:
      time_embedding = self.non_linearity(time_embedding)
      time_embedding = self.time_emb_proj(time_embedding)
      time_embedding = time_embedding[:, :, None, None] #type: ignore
      x += time_embedding

    x = self.norm2(x)
    x = self.non_linearity(x)
    x = self.conv2(x)

    if self.conv_shortcut is not None:
      residual = self.conv_shortcut(residual)
    
    return x + residual