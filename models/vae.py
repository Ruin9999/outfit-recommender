import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Tuple
from blocks import DownsampleBlock, UpsampleBlock, MidBlock

from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.configuration_utils import ConfigMixin

class AutoencoderKL(ModelMixin, ConfigMixin):
  def __init__(self):
    super().__init__()

    # Encoder
    self.encoder = nn.Module()
    self.encoder.down_blocks = nn.ModuleList([])
    self.encoder.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1)
    self.encoder.down_blocks.append(DownsampleBlock(128, 128, num_residual_layers=2))
    self.encoder.down_blocks.append(DownsampleBlock(128, 256, num_residual_layers=2))
    self.encoder.down_blocks.append(DownsampleBlock(256, 512, num_residual_layers=2))
    self.encoder.down_blocks.append(DownsampleBlock(512, 512, num_residual_layers=2, downsample=False))
    self.encoder.mid_block = MidBlock(512, attention_head_dim=512, add_attention=True)
    self.encoder.conv_norm_out = nn.GroupNorm(32, 512, eps=1e-6, affine=True)
    self.encoder.conv_act = nn.SiLU()
    self.encoder.conv_out = nn.Conv2d(512, 4 * 2, kernel_size=3, padding=1)

    # Decoder
    self.decoder = nn.Module()
    self.decoder.up_blocks = nn.ModuleList([])
    self.decoder.conv_in = nn.Conv2d(4, 512, kernel_size=3, padding=1)
    self.decoder.mid_block = MidBlock(512, attention_head_dim=512, add_attention=True)
    self.decoder.up_blocks.append(UpsampleBlock(512, 512, num_residual_layers=3))
    self.decoder.up_blocks.append(UpsampleBlock(512, 512, num_residual_layers=3))
    self.decoder.up_blocks.append(UpsampleBlock(512, 256, num_residual_layers=3))
    self.decoder.up_blocks.append(UpsampleBlock(256, 128, num_residual_layers=3, upsample=False))
    self.decoder.conv_norm_out = nn.GroupNorm(32, 128, eps=1e-6, affine=True)
    self.decoder.conv_act = nn.SiLU()
    self.decoder.conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    self.quant_conv = nn.Conv2d(4 * 2, 4 * 2, kernel_size=1, padding=0)
    self.post_quant_conv = nn.Conv2d(4, 4, kernel_size=1, padding=0)

  @apply_forward_hook
  def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
    x = self.encoder.conv_in(x)
    for layer in self.encoder.down_blocks:
      x, _ = layer(x)
    x = self.encoder.mid_block(x)
    x = self.encoder.conv_norm_out(x)
    x = self.encoder.conv_act(x)
    x = self.encoder.conv_out(x)
    x = self.quant_conv(x)
    prosterior = DiagonalGaussianDistribution(x)

    if not return_dict:
      return(prosterior,)
    return AutoencoderKLOutput(latent_dist=prosterior)
  
  @apply_forward_hook
  def decode(self, x: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
    upscale_dtype = next(iter(self.decoder.up_blocks.parameters())).dtype

    x = self.post_quant_conv(x)
    x = self.decoder.conv_in(x)
    x = self.decoder.mid_block(x).to(upscale_dtype)
    for layer in self.decoder.up_blocks:
        x = layer(x)
    x = self.decoder.conv_norm_out(x)
    x = self.decoder.conv_act(x)
    x = self.decoder.conv_out(x)

    if not return_dict:
      return (x,)

    return DecoderOutput(sample=x)
  
  def forward(self, x: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
    x = self.encode(x, return_dict=True).latent_dist # type: ignore
    z = x.mode()
    x = self.decode(z, return_dict=True).sample # type: ignore

    if not return_dict:
      return (x,)
    return DecoderOutput(sample=x)