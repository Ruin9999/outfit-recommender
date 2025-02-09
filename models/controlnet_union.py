import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, List, Union, Dict, Any
from dataclasses import dataclass
from collections import OrderedDict

from diffusers.utils.outputs import BaseOutput
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn
from blocks import DownsampleBlock, DownsampleCrossAttentionBlock

def zero_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        nn.init.zeros_(param)
    return module

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = super().forward(x)
        return x.to(dtype)

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_mask: Optional[torch.Tensor]= None):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ln_1 = LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(embed_dim, embed_dim * 4)), ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(embed_dim * 4, embed_dim))]))
        self.ln_2 = LayerNorm(embed_dim)
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        x = self.ln_1(x)
        x = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        x = x + residual

        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = x + residual

        return x

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    # From DDPM Paper 
    def get_timestep_embedding(
        self,
        timesteps: torch.Tensor,
        embedding_dim: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1,
        scale: float = 1,
        max_period: int = 10000,
    ):
        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - downscale_freq_shift)

        # Broadcast the timesteps to the embedding dimension
        embeddings = torch.exp(exponent)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = scale * embeddings

        if flip_sin_to_cos:
            embeddings = torch.cat([torch.cos(embeddings), torch.sin(embeddings)], dim=-1)
        else:
            embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # Zero padding incase we have a odd number of dimensions
        if embedding_dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))
        
        return embeddings

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timestep_embeddings = self.get_timestep_embedding(
            timesteps=timesteps,
            embedding_dim=self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale
        )

        return timestep_embeddings

class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, embedding_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, timestep: torch.Tensor, condition=None) -> torch.Tensor:
        timestep = self.linear_1(timestep)
        timestep = self.act(timestep)
        timestep = self.linear_2(timestep)

        return timestep


@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor

class ControlNetConditioningEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([])

        for i in range(len(hidden_channels) - 1):
            self.blocks.append(nn.Conv2d(hidden_channels[i], hidden_channels[i], kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(nn.Conv2d(hidden_channels[-1], out_channels, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x) # (batch_size, 3, height, width) -> (batch_size, 16, height, width)
        x = F.silu(x) # (batch_size, 16, height, width)

        for block in self.blocks: # (batch_size, 16, height, width) -> (batch_size, 256, height // 8, width // 8)
            x = block(x)
            x = F.silu(x)
        
        x = self.conv_out(x) # (batch_size, 256, height // 8, width // 8) -> (batch_size, 320, height // 8, width // 8)
    
        return x

class ControlNetModel_Union(ModelMixin, FromOriginalModelMixin, ConfigMixin):
    def __init__(self):
        super().__init__()

        self.conv_in = nn.Conv2d(4, 320, kernel_size=3, padding=1)
        
        # Time embedding
        self.time_proj = Timesteps(320, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(320, 320 * 4)

        self.add_time_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.add_embedding = TimestepEmbedding(2816, 320 * 4)
        
        # Control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(in_channels=3, hidden_channels=[16, 32, 96, 256], out_channels=320)

        # Copyright by Qi Xin (2024/07/06)
        # Condition Transformer augment the feature representation of conditions
        # The overall design is somewhat like resnet. The output of the Condition Transformer is used to predict a condition bias
        # that is added to the original condition feature
        num_control_type = 6
        num_trans_channel = 320
        num_trans_head = 8
        num_trans_layer = 1
        num_proj_channels = 320
        task_scale_factor = num_trans_channel ** 0.5

        self.task_embedding = nn.Parameter(task_scale_factor * torch.randn(num_control_type, num_trans_channel))
        self.transformer_layes = nn.Sequential(*[AttentionBlock(num_trans_channel, num_trans_head) for _ in range(num_trans_layer)])
        self.spatial_ch_projs = zero_module(nn.Linear(num_trans_channel, num_proj_channels))

        # Copyright by Qi Xin (2024/07/06)
        # Control Embedder to distinguish different control conditions
        # A simple but effective module, consists of an embedding layer and a linear layer,
        # to inject control info to time embedding.
        self.control_type_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.control_add_embedding = TimestepEmbedding(256 * num_control_type, 320 * 4)

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])
        
        controlnet_block = zero_module(nn.Conv2d(320, 320, kernel_size=1))
        self.controlnet_down_blocks.append(controlnet_block)

        # Down Blocks
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
            num_layers=2,
            eps=1e-05,
            num_transformer_layers=2,
            cross_attention_dim=2048,
        ))

        self.down_blocks.append(DownsampleCrossAttentionBlock(
            in_channels=640,
            out_channels=1280,
            time_embedding_channels=320 * 4,
            num_attention_heads=20,
            num_layers=2,
            eps=1e-05,
            num_transformer_layers=10,
            cross_attention_dim=2048,
            downsample=False,
        ))

        # For each down block, we have 3 controlnet layers except for the last block
        controlnet_block = nn.Conv2d(320, 320, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        controlnet_block = nn.Conv2d(320, 320, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        controlnet_block = nn.Conv2d(320, 320, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        controlnet_block = nn.Conv2d(640, 640, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        controlnet_block = nn.Conv2d(640, 640, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        controlnet_block = nn.Conv2d(640, 640, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        controlnet_block = nn.Conv2d(1280, 1280, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        controlnet_block = nn.Conv2d(1280, 1280, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        # Mid Block
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=1280,
            temb_channels=320 * 4,
            transformer_layers_per_block=10,
            resnet_eps=1e-05,
            resnet_time_scale_shift="default",
            resnet_act_fn="silu",
            resnet_groups=32,
            output_scale_factor=1,
            cross_attention_dim=2048,
            num_attention_heads=20,
            use_linear_projection=True,
        )

        controlnet_block = nn.Conv2d(1280, 1280, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond_list: torch.FloatTensor,
        conditioning_scale: float = 1.0,
        timestep_cond: Optional[torch.Tensor] = None, #This is not used in the timestep embedding original implementation at all
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:

        # 1. Prepare embeddings
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = timestep.expand(sample.shape[0])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = self.time_proj(timestep).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None


        assert added_cond_kwargs is not None, "`added_cond_kwargs` must be passed to the controlnet forward method."
        text_embeds = added_cond_kwargs["text_embeds"]
        time_ids = added_cond_kwargs["time_ids"]
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)

        # Copyright by Qi Xin(2024/07/06)
        # inject control type info to time embedding to distinguish different control conditions
        control_type = added_cond_kwargs['control_type']
        control_embeds = self.control_type_proj(control_type.flatten())
        control_embeds = control_embeds.reshape((t_emb.shape[0], -1))
        control_embeds = control_embeds.to(emb.dtype)
        control_emb = self.control_add_embedding(control_embeds)
        emb = emb + control_emb + aug_emb

        # 2. pre-process
        sample = self.conv_in(sample)
        indices = torch.nonzero(control_type[0])

        # Copyright by Qi Xin(2024/07/06)
        # add single/multi conditons to input image.
        # Condition Transformer provides an easy and effective way to fuse different features naturally
        inputs = []
        condition_list = []

        for idx in range(indices.shape[0] + 1):
            if idx == indices.shape[0]:
                controlnet_cond = sample
                feat_seq = torch.mean(controlnet_cond, dim=(2, 3)) # N * C
            else:
                controlnet_cond = self.controlnet_cond_embedding(controlnet_cond_list[indices[idx][0]])
                feat_seq = torch.mean(controlnet_cond, dim=(2, 3)) # N * C
                feat_seq = feat_seq + self.task_embedding[indices[idx][0]]

            inputs.append(feat_seq.unsqueeze(1))
            condition_list.append(controlnet_cond)

        x = torch.cat(inputs, dim=1)  # NxLxC
        x = self.transformer_layes(x)

        controlnet_cond_fuser = sample * 0.0
        for idx in range(indices.shape[0]):
            alpha = self.spatial_ch_projs(x[:, idx])
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
            controlnet_cond_fuser += condition_list[idx] + alpha
        
        sample = sample + controlnet_cond_fuser

        # 3. Down blocks
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    x=sample,
                    time_embedding=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    # attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(x=sample, time_embedding=emb)

            down_block_res_samples += res_samples

        # 4. Mid block
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. Applying controlnet zero blocks
        controlnet_down_block_res_samples = ()
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. Scaling
        down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )