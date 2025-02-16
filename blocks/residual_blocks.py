import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

from utils import default_init_weights

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

# x4 RRDBNet model
class ResidualDenseBlock(nn.Module):
  def __init__(
    self,
    in_channels: int = 64,
    hidden_channels: int = 32,
  ):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels + (2 * hidden_channels), hidden_channels, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(in_channels + (3 * hidden_channels), hidden_channels, kernel_size=3, padding=1)
    self.conv5 = nn.Conv2d(in_channels + (4 * hidden_channels), in_channels, kernel_size=3, padding=1)
    self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x1 = self.conv1(x)
    x1 = self.lrelu(x1)
    x2 = self.conv2(torch.cat([x, x1], dim=1))
    x2 = self.lrelu(x2)
    x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
    x3 = self.lrelu(x3)
    x4 = self.conv4(torch.cat([x, x1, x2, x3], dim=1))
    x4 = self.lrelu(x4)
    x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
    
    return x5 * 0.2 + x

class ResidualResidualDenseBlock(nn.Module):
  def __init__(
    self,
    in_channels: int = 64,
    hidden_channels: int = 32,
  ):
    super().__init__()
    self.rdb1 = ResidualDenseBlock(in_channels, hidden_channels)
    self.rdb2 = ResidualDenseBlock(in_channels, hidden_channels)
    self.rdb3 = ResidualDenseBlock(in_channels, hidden_channels)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    x = self.rdb1(x)
    x = self.rdb2(x)
    x = self.rdb3(x)

    return x * 0.2 + residual