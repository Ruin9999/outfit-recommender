import torch
import inspect
import warnings
import PIL.Image as Image
from typing import List, Optional, Tuple, Union
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from models import RefinerUNet
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

class StableDiffusionXLRefinerPipeline(DiffusionPipeline):
  model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
  _optional_components = [
    "tokenizer",
    "tokenizer_2",
    "text_encoder",
    "text_encoder_2",
    "image_encoder",
    "feature_extractor",
  ]

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
    self.register_to_config(force_zeros_for_empty_prompt=True)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

  def _get_prompt_embeddings(
    self,
    prompt: str,
    device: Optional[torch.device] = None
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(prompt, str), f"Prompts must be a string, but got type {type(prompt)}"
    dtype = self.text_encoder.dtype if self.text_encoder is not None else self.unet.dtype

    truncated_token_ids = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
    untruncated_token_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_token_ids.shape[-1] >= truncated_token_ids.shape[-1] and not torch.equal(untruncated_token_ids, truncated_token_ids):
      removed_text = self.tokenizer.batch_decode(untruncated_token_ids[:, self.tokenizer.model_max_length - 1: - 1])
      warnings.warn(f"{removed_text} was truncated from the input because it exceeded the max model length of {self.tokenizer.model_max_length}")

    self.text_encoder = self.text_encoder.to(device=device, dtype=torch.float32)
    embeddings = self.text_encoder(truncated_token_ids.to(device=device), output_hidden_states=True)
    pooled_embeddings = embeddings[0]
    embeddings = embeddings.hidden_states[-2]

    self.text_encoder = self.text_encoder.to(dtype=dtype)
    return embeddings.to(device=device, dtype=dtype), pooled_embeddings.to(device=device, dtype=dtype)

  # Referenced from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
  def _get_extra_step_kwargs(
    self, 
    generator: Optional[torch.Generator] = None,
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

  def _get_latents(
    self,
    image: torch.Tensor,
    timestep: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
  ):
    image = image.to(device=device, dtype=torch.float32)
    self.vae = self.vae.to(device=device, dtype=torch.float32)

    latents = self.vae.encode(image).latent_dist.sample(generator)
    latents = latents.to(device=device, dtype=dtype)
    latents = self.vae.config.scaling_factor * latents
    latents = torch.cat([latents], dim=0)

    batch_size, num_channels, height, width = latents.shape
    noise = randn_tensor((batch_size, num_channels, height, width), generator=generator, device=device, dtype=dtype)
    latents = self.scheduler.add_noise(latents, noise, timestep)
    
    self.vae = self.vae.to(dtype=dtype)
    return latents

  def _get_add_time_ids(
    self,
    original_size: Tuple[int, int],
    crops_coords_top_left: Tuple[int, int],
    target_size: Tuple[int, int],
    aesthetic_score: float = 6.0,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Optional[Tuple[int, int]] = None,
    negative_target_size: Optional[Tuple[int, int]] = None,
    negative_aesthetic_score: float = 2.5,
    text_encoder_projection_dim: int = 1280,
    dtype: Optional[torch.dtype] = None,
  ):
    if negative_original_size is None: negative_original_size = original_size
    if negative_crops_coords_top_left is None: negative_crops_coords_top_left = crops_coords_top_left
    if negative_target_size is None: negative_target_size = target_size

    add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
    add_neg_time_ids = list(negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,))

    passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
    expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
      warnings.warn(f"Expected addition time embedding dimension of {expected_add_embed_dim}, but got {passed_add_embed_dim}. Please check if the addition time embedding dimension is correct.")
    
    return torch.tensor(add_time_ids, dtype=dtype), torch.tensor(add_neg_time_ids, dtype=dtype)

  # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
  def upcast_vae(self):
    dtype = self.vae.dtype
    self.vae.to(dtype=torch.float32)
    use_torch_2_0_or_xformers = isinstance(
      self.vae.decoder.mid_block.attentions[0].processor,
      (AttnProcessor2_0, XFormersAttnProcessor),
    )
    if use_torch_2_0_or_xformers:
      self.vae.post_quant_conv.to(dtype)
      self.vae.decoder.conv_in.to(dtype)
      self.vae.decoder.mid_block.to(dtype)

  @torch.no_grad()
  def __call__(
    self,
    prompt: str,
    image: Image.Image,
    strength: float = 0.3,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    aesthetic_score: float = 6.0,
    negative_aesthetic_score: float = 2.5,
    return_dict: bool = True,
  ):
    device = self._execution_device

    # 1. Get prompt embeddings
    prompt_embeds, pooled_prompt_embeds = self._get_prompt_embeddings(prompt, device=device)
    if negative_prompt is None:
      negative_prompt_embeds = torch.zeros_like(prompt_embeds)
      negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    else:
      negative_prompt_embeds, negative_pooled_prompt_embeds = self._get_prompt_embeddings(negative_prompt, device=device)

    # 2. Get image latents
    image_tensor = self.image_processor.preprocess(image)

    # 3. Get timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    initial_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    timestep_start = max(num_inference_steps - initial_timestep, 0)
    timesteps = timesteps[timestep_start * self.scheduler.order :]
    num_inference_steps = num_inference_steps - timestep_start

    # 4. Get latent variables
    latents = self._get_latents(
      image=image_tensor,
      timestep=timesteps[:1],
      dtype=prompt_embeds.dtype,
      device=device,
      generator=generator
    )

    # 5. Get extra step kwargs
    extra_step_kwargs = self._get_extra_step_kwargs(generator)

    # 6. Get addition time ids and merge with prompt embeds
    height, width = latents.shape[-2:]
    height = height * self.vae_scale_factor
    width = width * self.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    add_time_ids, add_neg_time_ids = self._get_add_time_ids(
      original_size=original_size,
      crops_coords_top_left=crops_coords_top_left,
      target_size=target_size,
      aesthetic_score=aesthetic_score,
      negative_original_size=negative_original_size,
      negative_crops_coords_top_left=negative_crops_coords_top_left,
      negative_target_size=negative_target_size,
      negative_aesthetic_score=negative_aesthetic_score,
      dtype=prompt_embeds.dtype,
      text_encoder_projection_dim=self.text_encoder.config.projection_dim,
    )

    add_text_embeds = pooled_prompt_embeds
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(device=device)
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0).to(device=device)
    add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0).to(device=device)

    # 7. Denoising Loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
      for i, timestep in enumerate(timesteps):
        model_input = torch.cat([latents] * 2)
        model_input = self.scheduler.scale_model_input(model_input, timestep)
        
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        predicted_noise = self.unet(
          model_input,
          timestep,
          encoder_hidden_states=prompt_embeds,
          added_cond_kwargs=added_cond_kwargs,
          return_dict=False
        )[0]

        unconditional_predicted_noise, text_conditioned_predicted_noise = predicted_noise.chunk(2)
        predicted_noise = unconditional_predicted_noise + guidance_scale * (text_conditioned_predicted_noise - unconditional_predicted_noise)
        latents = self.scheduler.step(predicted_noise, timestep, latents, **extra_step_kwargs, return_dict=False)[0]

        if i == len(timesteps) - 1 or ((i +  1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
          progress_bar.update()


    self.upcast_vae()
    latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
    latents = latents / self.vae.config.scaling_factor
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type="pil") #type: ignore
    self.vae = self.vae.to(dtype=torch.float16)
    self.maybe_free_model_hooks()

    if not return_dict:
      return(image,)

    return StableDiffusionXLPipelineOutput(images=image) #type: ignore