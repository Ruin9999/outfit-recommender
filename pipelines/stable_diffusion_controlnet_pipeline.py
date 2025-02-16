import torch
import inspect
import warnings
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.models.attention_processor import AttnProcessor2_0, LoRAAttnProcessor2_0, LoRAXFormersAttnProcessor, XFormersAttnProcessor

from models import ControlNetUnion, BaseUNet

class StableDiffusionXLControlNetUnionPipeline(DiffusionPipeline):
  def __init__(
    self,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    unet: BaseUNet,
    controlnet: ControlNetUnion,
    scheduler: KarrasDiffusionSchedulers,
    force_zeros_for_empty_prompt: bool = True, # Just here to ignore pipeline error.
  ):
    super().__init__()
    self.register_modules(
      vae=vae,
      text_encoder=text_encoder,
      text_encoder_2=text_encoder_2,
      tokenizer=tokenizer,
      tokenizer_2=tokenizer_2,
      unet=unet,
      controlnet=controlnet,
      scheduler=scheduler,
    )
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
    self.control_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False)
    self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)

  def _get_prompt_embeddings(
    self,
    prompts: Union[List, str],
    device: Optional[torch.device] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(prompts, str): prompts = [prompts, prompts]
    dtype = self.text_encoder_2.dtype if self.text_encoder_2 is not None else self.unet.dtype

    embeddings_list = []
    for prompt, tokenizer, text_encoder in zip(prompts, [self.tokenizer, self.tokenizer_2], [self.text_encoder, self.text_encoder_2]):
      truncated_token_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
      untruncated_token_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

      if untruncated_token_ids.shape[-1] >= truncated_token_ids.shape[-1] and not torch.equal(untruncated_token_ids, truncated_token_ids):
        removed_text = tokenizer.batch_decode(untruncated_token_ids[:, tokenizer.model_max_length - 1: - 1])
        warnings.warn(f"{removed_text} was truncated from the input because it exceeded the max token length of {tokenizer.model_max_length}")

      embeddings = text_encoder(truncated_token_ids.to(device), output_hidden_states=True)
      pooled_embeddings = embeddings[0] # We are always interested in the final layers of the last text_encoder
      embeddings = embeddings.hidden_states[-2]
      embeddings_list.append(embeddings)
    
    concatenated_embeddings = torch.cat(embeddings_list, dim=-1)
    return concatenated_embeddings.to(device=device, dtype=dtype), pooled_embeddings.to(device=device, dtype=dtype)

  # Referenced from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
  def _get_extra_step_kwargs(
    self,
    generator: Optional[torch.Generator],
    eta: float = 0.0
  ):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    extra_step_kwargs = {}
    accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())

    if accepts_eta: extra_step_kwargs["eta"] = eta
    if accepts_generator: extra_step_kwargs["generator"] = generator

    return extra_step_kwargs

  def _get_add_time_ids(
    self,
    original_size: Tuple[int, int],
    crops_coords_top_left: Tuple[int, int],
    target_size: Tuple[int, int],
    dtype: Optional[torch.dtype],
    text_encoder_projection_dim: Optional[int] = None,
  ) -> torch.Tensor:
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    passed_add_embedding_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
    expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
    
    if expected_add_embed_dim != passed_add_embedding_dim:
      raise ValueError(f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embedding_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`.")
    return torch.tensor([add_time_ids], dtype=dtype)

  def prepare_image(
    self,
    image: Union[Image.Image, torch.Tensor, np.ndarray],
    width: int,
    height: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
  ):
    image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
    image = image.to(device=device, dtype=dtype)
    image = torch.cat([image] * 2)
    return image

  def prepare_latents(
    self,
    num_channel_latents: int,
    height: int,
    width: int,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
  ) -> torch.Tensor:
    shape = (1, num_channel_latents, int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor)
    latents = randn_tensor(shape=shape, generator=generator, device=device, dtype=dtype)
    latents = latents * self.scheduler.init_noise_sigma
    return latents

  # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
  def upcast_vae(self):
    dtype = self.vae.dtype
    self.vae.to(dtype=torch.float32)
    use_torch_2_0_or_xformers = isinstance(
      self.vae.decoder.mid_block.attentions[0].processor,
      (AttnProcessor2_0, XFormersAttnProcessor, LoRAXFormersAttnProcessor, LoRAAttnProcessor2_0),
    )

    # if xformers or torch_2_0 is used attention block does not need
    # to be in float32 which can save lots of memory
    if use_torch_2_0_or_xformers:
      self.vae.post_quant_conv.to(dtype=dtype)
      self.vae.decoder.conv_in.to(dtype=dtype)
      self.vae.decoder.mid_block.to(dtype=dtype)

  @torch.no_grad()
  def __call__(
    self,
    prompt: Union[List[str], str],
    image_list: List,
    union_control_type: torch.Tensor,
    neg_prompt: Optional[Union[List[str], str]] = None,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 8.0,
    generator: Optional[torch.Generator] = None,
    return_dict: bool = True,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
  ):
    original_size = original_size or (height, width)
    target_size = target_size or original_size
    device = self._execution_device

    # 1. Encode prompts
    prompt_embeddings, pooled_prompt_embeddings = self._get_prompt_embeddings(prompt, device=device)
    if neg_prompt is None:
      neg_prompt_embeddings = torch.zeros_like(prompt_embeddings)
      pooled_neg_prompt_embeddings = torch.zeros_like(pooled_prompt_embeddings)
    else:
      neg_prompt_embeddings, pooled_neg_prompt_embeddings = self._get_prompt_embeddings(neg_prompt, device=device)

    
    # 2. Prepare images
    for i in range(len(image_list)):
      if image_list[i] == 0: continue;
      image = self.prepare_image(
        image=image_list[i],
        height=height,
        width=width,
        device=device,
        dtype=self.controlnet.dtype
      )
      image_list[i] = image

    # 3. Prepare latents
    latents = self.prepare_latents(
      num_channel_latents = self.unet.config.in_channels,
      height=height,
      width=width,
      generator=generator,
      dtype=prompt_embeddings.dtype,
      device=device,
    )

    # 4. Prepare time and additional embeddings
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    add_text_embeddings = pooled_prompt_embeddings
    add_time_ids = self._get_add_time_ids(
      original_size=original_size,
      crops_coords_top_left=crops_coords_top_left,
      target_size=target_size,
      dtype=prompt_embeddings.dtype,
      text_encoder_projection_dim=self.text_encoder_2.config.projection_dim
    )
    neg_add_time_ids = add_time_ids

    prompt_embeddings = torch.cat([neg_prompt_embeddings, prompt_embeddings], dim=0)
    add_text_embeddings = torch.cat([pooled_neg_prompt_embeddings, add_text_embeddings], dim=0)
    add_time_ids = torch.cat([neg_add_time_ids, add_time_ids], dim=0)

    prompt_embeddings = prompt_embeddings.to(device=device)
    add_text_embeddings = add_text_embeddings.to(device=device)
    add_time_ids = add_time_ids.to(device=device)

    # 5. Get misc kwargs
    extra_step_kwargs = self._get_extra_step_kwargs(generator=generator)

    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
      for i, timestep in enumerate(timesteps):
        model_input = torch.cat([latents] * 2)
        model_input = self.scheduler.scale_model_input(model_input, timestep)
        add_cond_kwargs = {
          "text_embeds": add_text_embeddings,
          "time_ids": add_time_ids,
          "control_type": union_control_type.reshape(1, -1)
            .to(device, dtype=prompt_embeddings.dtype)
            .repeat(2, 1)
        }

        control_model_input = model_input
        control_prompt_embeddings = prompt_embeddings
        control_add_cond_kwargs = add_cond_kwargs

        # 6.1. ControlNet
        down_block_res_embeddings, mid_block_res_embedding = self.controlnet(
          control_model_input,
          timestep,
          encoder_hidden_states=control_prompt_embeddings,
          controlnet_cond_list=image_list,
          conditioning_scale=1.0,
          added_cond_kwargs=control_add_cond_kwargs,
          return_dict=False,
        )

        # 6.2. UNet
        predicted_noise = self.unet(
          model_input,
          timestep,
          encoder_hidden_states=prompt_embeddings,
          down_block_additional_residuals=down_block_res_embeddings,
          mid_block_additional_residual=mid_block_res_embedding,
          added_cond_kwargs=add_cond_kwargs,
          return_dict=False,
        )[0]

        # 6.3. Scheduler step
        unconditional_predicted_noise, text_conditioned_predicted_noise = predicted_noise.chunk(2)
        predicted_noise = unconditional_predicted_noise + guidance_scale * (text_conditioned_predicted_noise - unconditional_predicted_noise)
        latents = self.scheduler.step(predicted_noise, timestep, latents, **extra_step_kwargs, return_dict=False)[0]

        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
          progress_bar.update()

    # 7. Finally decoding!
    self.upcast_vae()
    latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
    latents = latents / self.vae.config.scaling_factor
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type="pil")
    self.vae.to(dtype=torch.float16)
    self.maybe_free_model_hooks()

    if not return_dict:
      return (image,)

    return StableDiffusionXLPipelineOutput(images=image) #type: ignore