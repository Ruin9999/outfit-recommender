import torch
import torch.nn as nn
from torch.nn import functional as F
from blocks import Timesteps, TimestepEmbedding, DownsampleBlock, DownsampleCrossAttentionBlock, UpsampleBlock, UpsampleCrossAttentionBlock
from typing import Union, Optional, Tuple, Any, Dict

from dataclasses import dataclass
from diffusers.utils.outputs import BaseOutput
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn

@dataclass
class UNetOutput(BaseOutput):
    sample: torch.Tensor

# block_out_channels = [320, 640, 1280]
class BaseUNet(ModelMixin, FromOriginalModelMixin, ConfigMixin):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)
        self.time_proj = Timesteps(320, frequency_shift=0.0)
        self.time_embedding = TimestepEmbedding(320, 320 * 4)
        self.add_time_proj = Timesteps(256, frequency_shift=0.0)
        self.add_embedding = TimestepEmbedding(2816, 320 * 4)

        # Down Blocks
        self.down_blocks = nn.ModuleList([])
        self.down_blocks.append(DownsampleBlock(320, 320, 320 * 4, num_residual_layers=2, eps=1e-5))
        self.down_blocks.append(DownsampleCrossAttentionBlock(
            in_channels=320,
            out_channels=640,
            time_embedding_channels=320 * 4,
            num_attention_heads=10,
            num_transformer_layers=2,
            num_layers=2,
            cross_attention_dim=2048,
            eps=1e-5,
        ))

        self.down_blocks.append(DownsampleCrossAttentionBlock(
            in_channels=640,
            out_channels=1280,
            time_embedding_channels=320 * 4,
            num_attention_heads=20,
            num_transformer_layers=10,
            num_layers=2,
            cross_attention_dim=2048,
            eps=1e-5,
            downsample=False
        ))

        # Mid Block
        self.mid_block = UNetMidBlock2DCrossAttn( # TODO: Refactor
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

        # Up Blocks
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
            eps=1e-5,
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
            eps=1e-5,
        ))

        self.up_blocks.append(UpsampleBlock(320, 320, 640, 320 * 4, num_residual_layers=3, upsample=False, eps=1e-5))

        self.conv_norm_out = nn.GroupNorm(32, 320, eps=1e-5)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(320, 4, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        added_cond_kwargs: Dict[str, torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[torch.Tensor] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None, # TODO: Remove, added this for compatability with the refiner pipeline.
        return_dict: bool = True,
    ) -> Union[UNetOutput, Tuple[torch.Tensor]]:

        # forward_upsample_size = False

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
        downsample_res_samples = (x, )
        for down_block in self.down_blocks:
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                x, res_sample = down_block(
                    x,
                    timestep_embedding,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                x, res_sample = down_block(x, timestep_embedding)
            downsample_res_samples += res_sample

        if down_block_additional_residuals is not None:
            # assert len(down_block_additional_residuals) == len(downsample_res_samples), f"`down_block_additional_residuals` must have the same length as `downsample_res_samples` but has {len(down_block_additional_residuals)} and {len(downsample_res_samples)} respectively."
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
            cross_attention_kwargs=cross_attention_kwargs,
        )

        if mid_block_additional_residual is not None:
            x = x + mid_block_additional_residual

        # 5. Up blocks
        for up_block in self.up_blocks:
            res_samples = downsample_res_samples[-len(up_block.resnets) :]
            downsample_res_samples = downsample_res_samples[: -len(up_block.resnets)]

            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                x = up_block(
                    x,
                    timestep_embedding,
                    res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                x = up_block(x, timestep_embedding, res_samples)

        # 6. Post processing
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        if not return_dict:
            return (x, )

        return UNetOutput(sample=x)