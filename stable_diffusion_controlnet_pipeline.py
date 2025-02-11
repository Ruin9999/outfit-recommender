import os
import torch
import inspect
import warnings
import numpy as np
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders.single_file import FromSingleFileMixin
from diffusers.loaders.lora_pipeline import StableDiffusionXLLoraLoaderMixin
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin
from diffusers.loaders.ip_adapter import IPAdapterMixin
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from models import ControlNetUnion
from diffusers.models.attention_processor import AttnProcessor2_0, LoRAAttnProcessor2_0, LoRAXFormersAttnProcessor, XFormersAttnProcessor
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from diffusers.utils.import_utils import is_accelerate_available, is_accelerate_version
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel

from utils import check_stable_diffusion_controlnet_inputs

class StableDiffusionXLControlNetUnionPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, StableDiffusionXLLoraLoaderMixin, FromSingleFileMixin,IPAdapterMixin
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        text_encoder_2 ([`~transformers.CLIPTextModelWithProjection`]):
            Second frozen text-encoder
            ([laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        tokenizer_2 ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings should always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
    """
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae" 

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
    ):
        super().__init__()

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

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

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def _get_prompt_embeddings(
        self,
        prompts: List[List[str]],
        tokenizers: List[CLIPTokenizer],
        text_encoders: Any,
        clip_skip: Optional[int],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings_list = []
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, tokenizer)

            truncated_token_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
            untruncated_token_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_token_ids.shape[-1] >= truncated_token_ids.shape[-1] and not torch.equal(untruncated_token_ids, truncated_token_ids):
                removed_text = tokenizer.batch_decode(untruncated_token_ids[:, tokenizer.model_max_length - 1 : -1])
                warnings.warn(f"{removed_text} was truncated from input because it exceeded the max token length of {tokenizer.model_max_length}")

            truncated_token_ids.to(device)
            text_encoder.to(device)
            embeddings = text_encoder(truncated_token_ids.to(device), output_hidden_states=True)
            pooled_embeddings = embeddings[0] # We are ALWAYS interested in the pooled output of the final text_encoder
            embeddings = embeddings.hidden_states[-2] if clip_skip is None else embeddings.hidden_states[-(clip_skip + 2)]
            embeddings_list.append(embeddings)

        # Concatenate the features from all the text_encoders
        concatenated_embeddings = torch.cat(embeddings_list, dim=-1)
        return concatenated_embeddings.to(device), pooled_embeddings.to(device)

    def encode_prompt(
        self,
        prompt: Union[List[str], str],
        prompt_2: Optional[Union[List[str], str]] = None,
        neg_prompt: Optional[Union[List[str], str]] = None,
        neg_prompt_2: Optional[Union[List[str], str]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        neg_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_neg_prompt_embeds: Optional[torch.Tensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        do_classifier_free_guidance: bool = True,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        dtype = self.text_encoder_2.dtype if self.text_encoder_2 is not None else self.unet.dtype

        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt if prompt_2 is None else prompt_2

        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        prompts = [prompt, prompt_2]
        
        prompt_embeds, pooled_prompt_embeds = self._get_prompt_embeddings(prompts, tokenizers, text_encoders, clip_skip, device=device)
        assert prompt_embeds is not None and pooled_prompt_embeds is not None, "`_get_prompt_embeddings` returned None for either `prompt_embeds` or `pooled_prompt_embeds`"
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)

        zero_out_neg_prompt = neg_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and neg_prompt_embeds is None and zero_out_neg_prompt:
            neg_prompt_embeds = torch.zeros_like(prompt_embeds)
            pooled_neg_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and neg_prompt_embeds is None:
            neg_prompt = neg_prompt or ""
            neg_prompt = [neg_prompt] if isinstance(neg_prompt, str) else neg_prompt
            neg_prompt_2 = [neg_prompt_2] if isinstance(neg_prompt_2, str) else neg_prompt if neg_prompt_2 is None else neg_prompt_2
            neg_prompts = [neg_prompt, neg_prompt_2]

            neg_prompt_embeds, pooled_neg_prompt_embeds = self._get_prompt_embeddings(neg_prompts, tokenizers, text_encoders, clip_skip=None, device=device)
            assert neg_prompt_embeds is not None and pooled_neg_prompt_embeds is not None, "`_get_prompt_embeddings` returned None for either `neg_prompt_embeds` or `pooled_neg_prompt_embeds`"
            neg_prompt_embeds = neg_prompt_embeds.to(device=device, dtype=dtype)
            pooled_neg_prompt_embeds = pooled_neg_prompt_embeds.to(device=device, dtype=dtype)

        assert num_images_per_prompt is not None # Just to get rid of the stupid error lines
        batch_size, sequence_length, embed_dim = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, sequence_length, embed_dim)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        if do_classifier_free_guidance:
            assert neg_prompt_embeds is not None, "`neg_prompt_embeds` is None but `do_classifier_free_guidance` is True"
            assert pooled_neg_prompt_embeds is not None, "`pooled_neg_prompt_embeds` is None but `do_classifier_free_guidance` is True"

            neg_prompt_embeds = neg_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            neg_prompt_embeds = neg_prompt_embeds.view(batch_size * num_images_per_prompt, sequence_length, embed_dim)
            pooled_neg_prompt_embeds = pooled_neg_prompt_embeds.repeat(1, num_images_per_prompt)
            pooled_neg_prompt_embeds = pooled_neg_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, pooled_neg_prompt_embeds
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Referenced from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image
    def prepare_image(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        width: int,
        height: int,
        num_images_per_prompt: int,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        do_classifier_free_guidance: bool = True,
        guess_mode: bool = False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image = image.repeat_interleave(num_images_per_prompt, dim=0)
        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)
        return image

    # Referenced from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size: int,
        num_channel_latents: int,
        height: int,
        width: int,
        dtype: Optional[torch.dtype],
        device: Optional[torch.device],
        generator: Optional[torch.Generator],
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor : 
        shape = (batch_size, num_channel_latents, int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor)
        if latents is None:
            latents = randn_tensor(shape=shape, generator=generator, device=device, dtype=dtype)

        latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        assert latents is not None, "`latents` in `prepare_latents` returned None"
        return latents

    def _get_add_time_ids( # Returns shape [1, 6]
        self,
        original_size: Tuple[int, int],
        crops_coords_top_left: Tuple[int, int],
        target_size: Tuple[int, int],
        dtype: Optional[torch.dtype],
        text_encoder_projection_dim: Optional[int] = None,
    ) -> torch.Tensor:
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        paassed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        
        if expected_add_embed_dim != paassed_add_embed_dim:
            raise ValueError(f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`.")
        return torch.tensor([add_time_ids], dtype=dtype)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image_list: List,
        union_control_type: torch.Tensor,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        neg_prompt: Optional[Union[str, List[str]]] = None,
        neg_prompt_2: Optional[Union[str, List[str]]] = None,
        neg_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_neg_prompt_embeds: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[List] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        controlnet_conditioning_scale: float = 1.0,
        controlnet_guidance_start: Union[List[float], float] = 0.0,
        controlnet_guidance_end: Union[List[float], float] = 1.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image. Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            neg_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `neg_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            neg_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. This is sent to `tokenizer_2`
                and `text_encoder_2`. If not defined, `neg_prompt` is used in both text-encoders.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            neg_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `neg_prompt_embeds` are generated from the `neg_prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, pooled text embeddings are generated from `prompt` input argument.
            pooled_neg_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs (prompt
                weighting). If not provided, pooled `neg_prompt_embeds` are generated from `neg_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            controlnet_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            controlnet_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(width, height)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(width, height)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned containing the output images.
        """
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        assert height is not None and width is not None
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
                
        controlnet_guidance_start, controlnet_guidance_end = [controlnet_guidance_start], [controlnet_guidance_end]


        # 1. Check inputs. Raise error if not correct
        check_stable_diffusion_controlnet_inputs(
            pipeline=self,
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            neg_prompt=neg_prompt,
            neg_prompt_2=neg_prompt_2,
            neg_prompt_embeds=neg_prompt_embeds,
            pooled_neg_prompt_embeds=pooled_neg_prompt_embeds,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            controlnet_guidance_start=controlnet_guidance_start,
            controlnet_guidance_end=controlnet_guidance_end,
            callback_steps=callback_steps,
            image_list=image_list,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
        )
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            assert isinstance(prompt_embeds, torch.Tensor), "`prompt_embeds` was not set to torch.Tensor"
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        global_pool_conditions = controlnet.config.global_pool_conditions
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, pooled_neg_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            neg_prompt=neg_prompt,
            neg_prompt_2=neg_prompt_2,
            neg_prompt_embeds=neg_prompt_embeds,
            pooled_neg_prompt_embeds=pooled_neg_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            lora_scale=text_encoder_lora_scale,
            device=device,
        )

        # 4. Prepare image
        assert num_images_per_prompt is not None
        assert image_list is not None, "`image_list` is None. Something went wrong above if an image was passed."
        for idx in range(len(image_list)):
            if image_list[idx]:
                image = self.prepare_image(
                    image=image_list[idx],
                    width=width,
                    height=height,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                height, width = image.shape[-2:]
                image_list[idx] = image


        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        assert isinstance(controlnet_guidance_start, list) and isinstance(controlnet_guidance_end, list)
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(controlnet_guidance_start, controlnet_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetUnion) else keeps)

        # 7.2 Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, 
            crops_coords_top_left, 
            target_size, 
            dtype=prompt_embeds.dtype, 
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=self.text_encoder_2.config.projection_dim
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            assert neg_prompt_embeds is not None and prompt_embeds is not None and pooled_neg_prompt_embeds is not None and add_text_embeds is not None # Get rid of error line
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([pooled_neg_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0) # (2, 6)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1) # (2 * batch_size, num_images_per_prompt, 6)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, timestep in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                    "control_type": union_control_type.reshape(1, -1)
                        .to(device, dtype=prompt_embeds.dtype)
                        .repeat(batch_size * num_images_per_prompt * 2, 1)
                }

                control_model_input = latent_model_input
                control_prompt_embeds = prompt_embeds
                control_added_cond_kwargs = added_cond_kwargs

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_embeds, mid_block_res_embeds = self.controlnet(
                    control_model_input,
                    timestep,
                    encoder_hidden_states=control_prompt_embeds,
                    controlnet_cond_list=image_list,
                    conditioning_scale=cond_scale,
                    added_cond_kwargs=control_added_cond_kwargs,
                    return_dict=False,
                )
                
                predicted_noise = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_embeds,
                    mid_block_additional_residual=mid_block_res_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # Perform guidance
                if do_classifier_free_guidance:
                    unconditional_predicted_noise, text_conditioned_predicted_noise = predicted_noise.chunk(2)
                    predicted_noise = unconditional_predicted_noise + guidance_scale * (text_conditioned_predicted_noise - unconditional_predicted_noise)
                    
                # Compute prevously noisy sample
                latents = self.scheduler.step(predicted_noise, timestep, latents, **extra_step_kwargs, return_dict=False)[0]
                assert latents is not None, "The fuck? The scheduler returned nothing?"
                # Call the callbacks
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, timestep, latents)
        
        if not output_type == "latent":
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # Cast back to fp16
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)

        else:
            image = latents
        
        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    # Overrride to properly handle the loading and unloading of the additional text encoder.
    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.load_lora_weights
    def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        # We could have accessed the unet config from `lora_state_dict()` too. We pass
        # it here explicitly to be able to tell that it's coming from an SDXL
        # pipeline.

        # Remove any existing hooks.
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module
        else:
            raise ImportError("Offloading requires `accelerate v0.17.0` or higher.")

        is_model_cpu_offload = False
        is_sequential_cpu_offload = False
        recursive = False
        for _, component in self.components.items():
            if isinstance(component, torch.nn.Module):
                if hasattr(component, "_hf_hook"):
                    is_model_cpu_offload = isinstance(getattr(component, "_hf_hook"), CpuOffload)
                    is_sequential_cpu_offload = isinstance(getattr(component, "_hf_hook"), AlignDevicesHook)
                    warnings.warn(
                        "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                    )
                    recursive = is_sequential_cpu_offload
                    remove_hook_from_module(component, recurse=recursive)
        state_dict, network_alphas = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )
        self.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=self.unet)

        text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
        if len(text_encoder_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=self.lora_scale,
            )

        text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
        if len(text_encoder_2_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder_2,
                prefix="text_encoder_2",
                lora_scale=self.lora_scale,
            )

        # Offload back.
        if is_model_cpu_offload:
            self.enable_model_cpu_offload()
        elif is_sequential_cpu_offload:
            self.enable_sequential_cpu_offload()

    @classmethod
    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.save_lora_weights
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]],
        text_encoder_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]],
        text_encoder_2_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]],
        weight_name: str,
        save_function: Callable,
        is_main_process: bool = True,
        safe_serialization: bool = True,
    ):
        state_dict = {}

        def pack_weights(layers, prefix):
            layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module) else layers
            layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
            return layers_state_dict

        if not (unet_lora_layers or text_encoder_lora_layers or text_encoder_2_lora_layers):
            raise ValueError(
                "You must pass at least one of `unet_lora_layers`, `text_encoder_lora_layers` or `text_encoder_2_lora_layers`."
            )

        if unet_lora_layers:
            state_dict.update(pack_weights(unet_lora_layers, "unet"))

        if text_encoder_lora_layers and text_encoder_2_lora_layers:
            state_dict.update(pack_weights(text_encoder_lora_layers, "text_encoder"))
            state_dict.update(pack_weights(text_encoder_2_lora_layers, "text_encoder_2"))

        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline._remove_text_encoder_monkey_patch
    def _remove_text_encoder_monkey_patch(self):
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder)
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder_2)