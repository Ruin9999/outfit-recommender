import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple, Dict, Any

from blocks import ResidualBlock
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.utils.import_utils import is_torch_version

class Upsample(nn.Module):
  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor: 
    dtype = x.dtype
    x = x.to(torch.float32)
    x = x.contiguous()
    x = F.interpolate(x, scale_factor=2, mode='nearest')
    x = x.to(dtype)

    x = self.conv(x)

    return x

class UpsampleBlock(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    prev_channels: Optional[int] = None,
    time_embedding_channels: Optional[int] = None,
    num_residual_layers: int = 2,
    upsample: bool = True,
    eps: float = 1e-6,
  ):
    super().__init__()
    self.prev_channels = prev_channels
    self.resnets = nn.ModuleList([])
    self.upsamplers = nn.ModuleList([])

    for i in range(num_residual_layers):
      if prev_channels is not None:
        res_skip_channels = in_channels if (i == num_residual_layers - 1) else out_channels
        res_in_channels = prev_channels if i == 0 else out_channels
        self.resnets.append(ResidualBlock(res_skip_channels + res_in_channels, out_channels, time_embedding_channels, eps=eps))
      else:
        in_channels = in_channels if i == 0 else out_channels
        self.resnets.append(ResidualBlock(in_channels, out_channels, time_embedding_channels, eps=eps))
      
    if upsample:
      self.upsamplers.append(Upsample(out_channels, out_channels))

  def forward(
    self,
    x: torch.Tensor,
    time_embedding: Optional[torch.Tensor] = None,
    res_hidden_states_tuple: Optional[Tuple[torch.Tensor, ...]] = None,
  ):
    if res_hidden_states_tuple is not None:
      assert self.prev_channels is not None, "Nah fam, you need to provide `prev_channels` in the constructor if you want to provide `res_hidden_states_tuple`"

    for layer in self.resnets:
      if self.prev_channels is not None and res_hidden_states_tuple is not None:
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]
        x = torch.cat([x, res_hidden_states], dim=1)
      x = layer(x, time_embedding)

    for upsample in self.upsamplers:
      x = upsample(x)

    return x

class UpsampleCrossAttentionBlock(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    prev_channels: Optional[int] = None,
    time_embedding_channels: Optional[int] = None,
    num_attention_heads: int = 1,
    num_transformer_layers: int = 1,
    num_layers: int = 1,
    upsample: bool = True,
    cross_attention_dim: int = 2048,
    eps: float = 1e-6,
  ) -> None:
    super().__init__()
    self.has_cross_attention = True
    self.resnets = nn.ModuleList([])
    self.attentions = nn.ModuleList([])
    self.upsamplers = nn.ModuleList([])

    for i in range(num_layers):
      res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
      res_in_channels = prev_channels if i == 0 else out_channels

      self.resnets.append(ResidualBlock(res_skip_channels + res_in_channels, out_channels, time_embedding_channels, eps=eps)) #type: ignore
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

    if upsample:
      self.upsamplers.append(Upsample(out_channels, out_channels))

  def forward(
    self,
    x: torch.Tensor,
    time_embedding: Optional[torch.Tensor] = None,
    res_hidden_states_tuple: Optional[Tuple[torch.Tensor, ...]] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
  ) -> torch.Tensor:
    for resnet, attention in zip(self.resnets, self.attentions):
      res_hidden_states = res_hidden_states_tuple[-1] #type: ignore
      res_hidden_states_tuple = res_hidden_states_tuple[:-1] #type: ignore 
      x = torch.cat([x, res_hidden_states], dim=1)

      x = resnet(x, time_embedding)
      x = attention(x, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs)[0]

    for layer in self.upsamplers:
      x = layer(x)

    return x