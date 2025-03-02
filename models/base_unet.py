import torch
import torch.nn as nn
from typing import Union, Optional, Tuple, Any, Dict

from utils import UNetOutput
from blocks import Timesteps, TimestepEmbedding, DownsampleBlock, DownsampleCrossAttentionBlock, UpsampleBlock, UpsampleCrossAttentionBlock

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn

class BaseUNet(ModelMixin, FromOriginalModelMixin, ConfigMixin):
  def __init__(self):
    super().__init__()
    self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)
    self.time_proj = Timesteps(320, frequency_shift=0.0)
    self.time_embedding = TimestepEmbedding(320, 320 * 4)
    self.add_time_proj = Timesteps(256, frequency_shift=0.0)
    self.add_embedding = TimestepEmbedding(2816, 320 * 4)

    # Down blocks
    self.down_blocks = nn.ModuleList([])
    self.down_blocks.append(DownsampleBlock(
      in_channels=320,
      out_channels=320,
      time_embedding_channels=320 * 4,
      num_residual_layers=2,
      eps=1e-05,
    ))

    self.down_blocks.append(DownsampleCrossAttentionBlock(
      in_channels=320,
      out_channels=640,
      time_embedding_channels=320 * 4,
      num_attention_heads=10,
      num_transformer_layers=2,
      num_layers=2,
      cross_attention_dim=2048,
      eps=1e-05,
    ))

    self.down_blocks.append(DownsampleCrossAttentionBlock(
      in_channels=640,
      out_channels=1280,
      time_embedding_channels=320 * 4,
      num_attention_heads=20,
      num_transformer_layers=10,
      num_layers=2,
      cross_attention_dim=2048,
      eps=1e-05,
      downsample=False,
    ))

    # Mid block
    self.mid_block = UNetMidBlock2DCrossAttn(
      temb_channels=320 * 4,
      in_channels=1280,
      resnet_eps=1e-5,
      resnet_act_fn="silu",
      resnet_groups=32,
      output_scale_factor=1.0,
      transformer_layers_per_block=10,
      num_attention_heads=20,
      cross_attention_dim=2048,
      dual_cross_attention=False,
      use_linear_projection=True,
      upcast_attention=False,
      resnet_time_scale_shift="default",
      attention_type="default",
      dropout=0.0,
    )

    # Up blocks
    self.up_blocks = nn.ModuleList([])
    self.up_blocks.append(UpsampleCrossAttentionBlock(
      in_channels=640,
      out_channels=1280,
      prev_channels=1280,
      time_embedding_channels=320 * 4,
      num_attention_heads=20,
      num_transformer_layers=10,
      num_layers=3,
      cross_attention_dim=2048,
      eps=1e-05,
    ))

    self.up_blocks.append(UpsampleCrossAttentionBlock(
      in_channels=320,
      out_channels=640,
      prev_channels=1280,
      time_embedding_channels=320 * 4,
      num_attention_heads=10,
      num_transformer_layers=2,
      num_layers=3,
      cross_attention_dim=2048,
      eps=1e-05,
    ))

    self.up_blocks.append(UpsampleBlock(
      in_channels=320,
      out_channels=320,
      prev_channels=640,
      time_embedding_channels=320 * 4,
      num_residual_layers=3,
      upsample=False,
      eps=1e-05,
    ))

    self.conv_norm_out = nn.GroupNorm(32, 320, eps=1e-05)
    self.conv_act = nn.SiLU()
    self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)
  
  def forward(
    self,
    x: torch.Tensor,
    timestep: torch.Tensor,
    added_cond_kwargs: Dict[str, torch.Tensor],
    encoder_hidden_states: Optional[torch.Tensor] = None,
    down_block_additional_residuals: Optional[torch.Tensor] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    return_dict: bool = True,
  ) -> Union[UNetOutput, Tuple[torch.Tensor]]:

    # 1. Time embedding
    if not isinstance(timestep, torch.Tensor):
      timestep = torch.tensor([timestep], dtype=torch.int64, device=x.device)
    else:
      timestep = timestep[None].to(x.device)

    timestep = timestep.expand(x.shape[0])
    timestep_projection = self.time_proj(timestep)
    timestep_projection = timestep_projection.to(x.dtype)
    timestep_embedding = self.time_embedding(timestep_projection)

    assert "text_embeds" in added_cond_kwargs, "`text_embeds` must be in `added_cond_kwargs`"
    assert "time_ids" in added_cond_kwargs, "`time_ids` must be in `added_cond_kwargs`"
    added_text_embedding = added_cond_kwargs["text_embeds"]
    added_time_ids = added_cond_kwargs["time_ids"]
    added_time_embedding = self.add_time_proj(added_time_ids.flatten()) 
    added_time_embedding = added_time_embedding.reshape(added_text_embedding.shape[0], -1)
    added_embeddings = torch.cat([added_text_embedding, added_time_embedding], dim=-1).to(timestep_embedding.dtype)
    added_embeddings = self.add_embedding(added_embeddings)
    timestep_embedding = timestep_embedding + added_embeddings

    # 2. Pre processing
    x = self.conv_in(x)

    # 3. Down blocks
    downsample_res_samples = (x,)
    for down_block in self.down_blocks:
      if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
        x, res_sample = down_block(
          x,
          timestep_embedding,
          encoder_hidden_states=encoder_hidden_states,
        )
      else:
        x, res_sample = down_block(x, timestep_embedding)
      downsample_res_samples += res_sample

    # 3.5 Insert controlnet layers.
    if down_block_additional_residuals is not None:
      new_down_block_res_samples = ()
      for down_block_res_sample, additional_residual in zip(downsample_res_samples, down_block_additional_residuals):
        new_down_block_res_sample = down_block_res_sample + additional_residual
        new_down_block_res_samples += (new_down_block_res_sample, )

      downsample_res_samples = new_down_block_res_samples

    # 4. Mid block
    x = self.mid_block(
      x,
      timestep_embedding,
      encoder_hidden_states=encoder_hidden_states,
    )

    if mid_block_additional_residual is not None:
      x = x + mid_block_additional_residual

    # 5. Up blocks
    for index, up_block in enumerate(self.up_blocks):
      res_samples = downsample_res_samples[-len(up_block.resnets) :]
      downsample_res_samples = downsample_res_samples[: -len(up_block.resnets)]

      forward_upsample_size = False
      for dimension in x.shape[-2:]:
        if dimension % (2 ** 2) != 0:
          forward_upsample_size = True
          break

      if forward_upsample_size and index != len(self.up_blocks) - 1:
        upsample_size = downsample_res_samples[-1].shape[2:]
      else:
        upsample_size = None

      if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
        x = up_block(
          x,
          timestep_embedding,
          res_samples,
          upsample_size=upsample_size,
          encoder_hidden_states=encoder_hidden_states,
        )
      else:
        x = up_block(x, timestep_embedding, res_samples, upsample_size=upsample_size)
      
    # 6. Post processing
    x = self.conv_norm_out(x)
    x = self.conv_act(x)
    x = self.conv_out(x)

    if not return_dict:
      return (x,)

    return UNetOutput(sample=x)