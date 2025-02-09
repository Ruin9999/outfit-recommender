import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

class Timesteps(nn.Module):
  def __init__(
    self,
    timestep_embedding_channels: int,
    frequency_shift: float,
    max_period: int = 10000,
  ):
    super().__init__()
    self.timestep_embedding_channels = timestep_embedding_channels
    self.frequency_shift = frequency_shift
    self.max_period = max_period

  def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
    # Implementation from original stable diffusion paper.
    half_dim = self.timestep_embedding_channels // 2
    exponent = -math.log(self.max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - self.frequency_shift)

    timesteps_embedding = torch.exp(exponent)
    timesteps_embedding = timesteps[:, None] * timesteps_embedding[None, :]
    timesteps_embedding = torch.cat([torch.cos(timesteps_embedding), torch.sin(timesteps_embedding)], dim=-1)

    if self.timestep_embedding_channels % 2:
      pad = (0, 1, 0, 0) # L, R, U, D
      timesteps_embedding = F.pad(timesteps_embedding, pad)

    return timesteps_embedding

class TimestepEmbedding(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int
  ):
    super().__init__()
    self.linear_1 = nn.Linear(in_channels, out_channels, bias=True)
    self.act = nn.SiLU()
    self.linear_2 = nn.Linear(out_channels, out_channels, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.linear_1(x)
    x = self.act(x)
    x = self.linear_2(x)
    return x