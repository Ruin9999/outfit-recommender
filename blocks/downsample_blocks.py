import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple, Dict, Any

from blocks import ResidualBlock
from diffusers.models.transformers.transformer_2d import Transformer2DModel

class Downsample(nn.Module):
  def __init__(self, in_channels: int, out_channels: int) -> None:
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv(x)
    return(x)
  
class DownsampleBlock(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    time_embedding_channels: Optional[int] = None,
    num_residual_layers: int = 2,
    downsample: bool = True,
    eps: float = 1e-6,
  ):
    super().__init__()
    self.resnets = nn.ModuleList([])
    self.downsamplers = nn.ModuleList([])

    for _ in range(num_residual_layers):
      self.resnets.append(ResidualBlock(in_channels, out_channels, time_embedding_channels, eps=eps))
      in_channels = out_channels
    if downsample:
      self.downsamplers.append(Downsample(out_channels, out_channels))

  def forward(self, x: torch.Tensor, time_embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
      output_states = []
      for resnet in self.resnets:
        x = resnet(x, time_embedding)
        output_states.append(x)
      for downsample in self.downsamplers:
        x = downsample(x)
        output_states.append(x)
      return x, tuple(output_states)
  
class DownsampleCrossAttentionBlock(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    time_embedding_channels: Optional[int] = None,
    num_attention_heads: int = 1,
    num_transformer_layers: int = 1,
    num_layers: int = 2,
    downsample: bool = True,
    cross_attention_dim: int = 2048,
    eps: float = 1e-6,
  ) -> None:
    super().__init__()
    self.has_cross_attention = True
    self.resnets = nn.ModuleList([])
    self.attentions = nn.ModuleList([])
    self.downsamplers = nn.ModuleList([])
    
    for _ in range(num_layers):
      self.resnets.append(ResidualBlock(in_channels, out_channels, time_embedding_channels, eps=eps))
      self.attentions.append(Transformer2DModel(
        num_attention_heads,
        out_channels // num_attention_heads,
        in_channels=out_channels,
        num_layers=num_transformer_layers,
        cross_attention_dim=cross_attention_dim,
        norm_num_groups=32,
        use_linear_projection=True,
        only_cross_attention=False,
        upcast_attention=False,
        attention_type="default",
      ))
      in_channels = out_channels

    if downsample:
      self.downsamplers.append(Downsample(out_channels, out_channels))
  
  def forward(
    self,
    x: torch.Tensor,
    time_embedding: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
  ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    output_states = ()
    for resnet, attention in zip(self.resnets, self.attentions):
      x = resnet(x, time_embedding)
      x = attention(x, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
      output_states += (x,)
    
    for layer in self.downsamplers:
      x = layer(x)
      output_states += (x,)

    return x, output_states