import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

from blocks import ResidualBlock
from diffusers.models.attention_processor import Attention

class MidBlock(nn.Module):
  def __init__(
    self,
    in_channels: int,
    num_layers: int = 1,
    attention_head_dim: int = 1,
    add_attention: bool = True,
    eps: float = 1e-6,
  ):
    super().__init__()
    self.resnets = nn.ModuleList([])
    self.attentions = nn.ModuleList([])

    # Always at least 1 residual block
    self.resnets.append(ResidualBlock(in_channels, in_channels, eps=eps))

    for _ in range(num_layers):
      if add_attention:
        self.attentions.append(Attention(
          query_dim=in_channels,
          heads=in_channels // attention_head_dim,
          dim_head=attention_head_dim,
          eps=eps,
          norm_num_groups=32,
          residual_connection=True,
          bias=True,
          _from_deprecated_attn_block=True,
        ))
      self.resnets.append(ResidualBlock(in_channels, in_channels, eps=eps))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.resnets[0](x)
    for attention, resnet in zip(self.attentions, self.resnets[1:]): #type: ignore
      if attention is not None:
        x = attention(x)
      x = resnet(x)

    return x