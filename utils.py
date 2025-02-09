import os
from datetime import datetime
from typing import List, Union

import torch
import math
import numpy as np
from PIL import Image
from torch.nn import functional as F
from typing import List, Union, Tuple, Optional, Any
from models import ControlNetModel_Union


# Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.check_image
def check_image(
    image: Any,
    prompt: Union[List[str], str],
    prompt_embeds: Optional[torch.Tensor],
):
    image_is_pil = isinstance(image, Image.Image)
    image_is_tensor = isinstance(image, torch.Tensor)
    image_is_np = isinstance(image, np.ndarray)
    image_is_pil_list = isinstance(image, list) and isinstance(image[0], Image.Image)
    image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
    image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)
    
    if not image_is_pil and not image_is_tensor and not image_is_np and not image_is_pil_list and not image_is_tensor_list and not image_is_np_list:
        raise ValueError(f"`image` has to be a PIL Image, a torch Tensor, a numpy array, a list of PIL Images, a list of torch Tensors or a list of numpy arrays but is {image} of type {type(image)}")

    image_batch_size = 1 if image_is_pil else len(image)
    if prompt_embeds is not None:
        prompt_batch_size = 1 if isinstance(prompt, str) else len(prompt)
    elif prompt_embeds is not None:
        prompt_batch_size = prompt_embeds.shape[0]

    if image_batch_size != 1 and image_batch_size != prompt_batch_size:
        raise ValueError(f"Batch size of `image` and `prompt` have to be the same but are {image_batch_size} and {prompt_batch_size}")

def check_stable_diffusion_controlnet_inputs(
    pipeline: Any,
    prompt: Union[List[str], str],
    prompt_2: Optional[Union[List[str], str]],
    prompt_embeds: Optional[torch.Tensor],
    pooled_prompt_embeds: Optional[torch.Tensor],
    neg_prompt: Optional[Union[List[str], str]],
    neg_prompt_2: Optional[Union[List[str], str]],
    neg_prompt_embeds: Optional[torch.Tensor],
    pooled_neg_prompt_embeds: Optional[torch.Tensor],
    controlnet_conditioning_scale: Optional[float],
    controlnet_guidance_start: Optional[Union[Tuple, List, float]],
    controlnet_guidance_end: Optional[Union[Tuple, List, float]],
    callback_steps: Optional[int],
    image_list: Optional[List],
    ip_adapter_image: Optional[List[torch.Tensor]],
    ip_adapter_image_embeds: Optional[Union[List, torch.Tensor]],
):
    if (callback_steps is None) or (callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)):
        raise ValueError(f"`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}")
    
    if prompt is not None and prompt_embeds is not None:
        raise ValueError(f"Only one of `prompt` and `prompt_embeds` can be provided but both are")

    if prompt_2 is not None and prompt_embeds is not None:
        raise ValueError(f"Only one of `prompt_2` and `prompt_embeds` can be provided but both are")

    if prompt is None and prompt_embeds is None:
        raise ValueError(f"At least one of `prompt` and `prompt_embeds` has to be provided but none are")

    if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
        raise ValueError(f"`prompt` has to be a string or a list of strings but is {prompt} of type {type(prompt)}")

    if prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
        raise ValueError(f"`prompt_2` has to be a string or a list of strings but is {prompt_2} of type {type(prompt_2)}")

    if neg_prompt is not None and neg_prompt_embeds is not None:
        raise ValueError(f"Only one of `neg_prompt` and `neg_prompt_embeds` can be provided but both are")

    if neg_prompt_2 is not None and neg_prompt_embeds is not None:
        raise ValueError(f"Only one of `neg_prompt_2` and `neg_prompt_embeds` can be provided but both are")

    if neg_prompt is not None and (not isinstance(neg_prompt, str) and not isinstance(neg_prompt, list)):
        raise ValueError(f"`neg_prompt` has to be a string or a list of strings but is {neg_prompt} of type {type(neg_prompt)}")

    if neg_prompt_2 is not None and (not isinstance(neg_prompt_2, str) and not isinstance(neg_prompt_2, list)):
        raise ValueError(f"`neg_prompt_2` has to be a string or a list of strings but is {neg_prompt_2} of type {type(neg_prompt_2)}")

    # Check shape consistency
    if prompt_embeds is not None and neg_prompt_embeds is not None and prompt_embeds.shape != neg_prompt_embeds.shape:
        raise ValueError(f"Shapes of `prompt_embeds` and `neg_prompt_embeds` have to be the same but are {prompt_embeds.shape} and {neg_prompt_embeds.shape}")

    # Check that prompt_embeds and pooled_prompt_embeds are always passed together
    if prompt_embeds is not None and pooled_prompt_embeds is None:
        raise ValueError(f"`pooled_prompt_embeds` has to be provided if `prompt_embeds` is provided but is None")

    if neg_prompt_embeds is not None and pooled_neg_prompt_embeds is None:
        raise ValueError(f"`pooled_neg_prompt_embeds` has to be provided if `neg_prompt_embeds` is provided but is None")

    if pooled_prompt_embeds is not None and prompt_embeds is None:
        raise ValueError(f"`prompt_embeds` has to be provided if `pooled_prompt_embeds` is provided but is None")

    if pooled_neg_prompt_embeds is not None and neg_prompt_embeds is None:
        raise ValueError(f"`neg_prompt_embeds` has to be provided if `pooled_neg_prompt_embeds` is provided but is None")

    is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(pipeline.controlnet, torch._dynamo.eval_frame.OptimizedModule)
    if isinstance(pipeline.controlnet, ControlNetModel_Union) or is_compiled and isinstance(pipeline.controlnet._orig_mod, ControlNetModel_Union):
        if image_list is not None:
            for image in image_list:
                if image:
                    check_image(image, prompt, prompt_embeds)

    if isinstance(pipeline.controlnet, ControlNetModel_Union) or is_compiled and isinstance(pipeline.controlnet._orig_mod, ControlNetModel_Union):
            if not isinstance(controlnet_conditioning_scale, float):
                raise ValueError(f"`controlnet_conditioning_scale` has to be a float but is {controlnet_conditioning_scale} of type {type(controlnet_conditioning_scale)}")

    if not isinstance(controlnet_guidance_start, (tuple, list)):
        controlnet_guidance_start = [controlnet_guidance_start]
    
    if not isinstance(controlnet_guidance_end, (tuple, list)):
        controlnet_guidance_end = [controlnet_guidance_end]

    if len(controlnet_guidance_start) != len(controlnet_guidance_end):
        raise ValueError(f"Length of `controlnet_guidance_start` and `controlnet_guidance_end` have to be the same but are {len(controlnet_guidance_start)} and {len(controlnet_guidance_end)}")

    for start, end in zip(controlnet_guidance_start, controlnet_guidance_end):
        if start >= end:
            raise ValueError(f"Values in `controlnet_guidance_start` have to be smaller than values in `controlnet_guidance_end` but are {controlnet_guidance_start} and {controlnet_guidance_end}")
        if start < 0.0:
            raise ValueError(f"Values in `controlnet_guidance_start` have to be greater or equal to 0.0 but are {controlnet_guidance_start}")
        if end > 1.0:
            raise ValueError(f"Values in `controlnet_guidance_end` have to be smaller or equal to 1.0 but are {controlnet_guidance_end}")
    
    if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
        raise ValueError(f"Only one of `ip_adapter_image` and `ip_adapter_image_embeds` can be provided but both are")

    if ip_adapter_image is not None:
        if not isinstance(ip_adapter_image_embeds, list):
            raise ValueError(
                f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
            )
        elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
            raise ValueError(
                f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
            )

def save_images(images: List[Union[torch.Tensor, np.ndarray, Image.Image]], output_dir: str = "outputs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, img in enumerate(images):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{idx}.png"
        filepath = os.path.join(output_dir, filename)
        
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray((img * 255).astype(np.uint8))
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            pass
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        img.save(filepath)