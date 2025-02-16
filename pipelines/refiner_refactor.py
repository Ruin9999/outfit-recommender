import torch
import inspect
import warnings
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional, List

from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import ImageProjection
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

from models import RefinerUNet

class StableDiffusionXLRefinerPipeline(DiffusionPipeline):
  model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"

  def __init__(
    self,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModelWithProjection,
    tokenizer: CLIPTokenizer,
    unet: RefinerUNet,
    scheduler: KarrasDiffusionSchedulers,
    force_zeros_for_empty_prompt: bool = True,
  ):
    super().__init__()

    self.register_modules(
      vae=vae,
      text_encoder=text_encoder,
      tokenizer=tokenizer,
      unet=unet,
      scheduler=scheduler,
    )
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
    self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)

  def _get_prompt_embeddings(
    self,
    prompt: str,
    device: Optional[torch.device] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = self.text_encoder.dtype if self.text_encoder is not None else self.unet.dtype

    truncated_token_ids = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
    untruncated_token_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_token_ids.shape[-1] >= truncated_token_ids.shape[-1] and not torch.equal(untruncated_token_ids, truncated_token_ids):
      removed_text = self.tokenizer.batch_decode(untruncated_token_ids[:, self.tokenizer.model_max_length - 1: -1])
      warnings.warn(f"{removed_text} was truncated from the input because it exceeded the max token length of {tokenizer.model_max_length}")

    embeddings = self.text_encoder(truncated_token_ids.to(device), output_hidden_states=True)
    pooled_embeddings = embeddings[0]
    embeddings = embeddings.hidden_states[-2]

    return embeddings.to(device=device, dtype=dtype), pooled_embeddings.to(device=device, dtype=dtype)

  def _get_timesteps(
    self,
    num_inference_steps: int,
    strength: float,
  ):
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    start_timestep = max(num_inference_steps - init_timestep, 0)

    timesteps = self.scheduler.timesteps[start_timestep * self.scheduler.order :]
    if hasattr(self.scheduler, "set_begin_index"): self.scheduler.set_begin_index(start_timestep * self.scheduler.order)
    return timesteps, num_inference_steps - start_timestep

  def prepare_latents(
    self,
    image: torch.Tensor,
    timestep: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
  ):
    assert isinstance(image, (torch.Tensor, Image.Image))
    image = image.to(device=device, dtype=torch.float32)
    self.vae.to(device=device, dtype=torch.float32)

    latents = self.vae.encode(image)
    latents = latents.sample
    self.vae.to(dtype)
    latents = latents.to(dtype=dtype)
    latents = latents * self.vae.scaling_factor

    noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=dtype)
    latents = self.scheduler.add_noise(latents, noise, timestep)
    return latents

  @torch.no_grad()
  def __call__(
    self,
    prompt: str,
    image: PipelineImageInput,
    strength: float = 0.3,
    num_inference_steps: int = 50,
    return_dict: bool = True,
  ):
    device = self._execution_device

    # 1. Encode prompts
    prompt_embeddings, pooled_prompt_embeddings = self._get_prompt_embeddings(prompt, device=device)
    neg_prompt_embeddings, pooled_neg_prompt_embeddings = torch.zeros_like(prompt_embeddings), torch.zeros_like(pooled_prompt_embeddings)

    # 2. Prepare image
    image = self.image_processor.preprocess(image)

    # 3. Prepare timesteps

    # 4. Prepare latents
