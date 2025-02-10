import cv2
import torch
import random
import numpy as np
from PIL import Image
from controlnet_aux import OpenposeDetector
from models import ControlNetUnion, BaseUNet, AutoencoderKL
from utils import save_images

from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from stable_diffusion_xl_controlet import StableDiffusionXLControlNetUnionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline

# CONFIGURATION
PROMPT="Model in layered street style, standing against a vibrant graffiti wall, Vivid colors, Mirrorless, 28mm lens"
NEG_PROMPT="out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature."
CONTROLNET_IMG_PATH="test.jpg"
ACTIVE_DEVICE="cuda:0"
IDLE_DEVICE="cpu"
IMG_OUTPUT_PATH="/outputs"

# LOAD MODELS
scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", torch_dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", torch_dtype=torch.float16)
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", torch_dtype=torch.float16)
tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer", torch_dtype=torch.float16)
tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2", torch_dtype=torch.float16)
base_unet = BaseUNet.from_pretrained("SG161222/RealVisXL_V4.0", subfolder="unet", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
# refiner_unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", subfolder="unet", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

controlnet = ControlNetUnion.from_pretrained("xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True)

# PRE-PROCESS CONTROLNET IMAGE
pose_processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
controlnet_img = cv2.imread(CONTROLNET_IMG_PATH)
controlnet_img = pose_processor(controlnet_img, hand_and_face=False, output_type='cv2')

# need to resize the image resolution to 1024 * 1024 or same bucket resolution to get the best performance
height, width, _  = controlnet_img.shape # type: ignore
ratio = np.sqrt(1024. * 1024. / (width * height))
new_width, new_height = int(width * ratio), int(height * ratio)
controlnet_img = cv2.resize(controlnet_img, (new_width, new_height)) # type: ignore
controlnet_img = Image.fromarray(controlnet_img)

# INFERENCE
base_pipeline = StableDiffusionXLControlNetUnionPipeline(
    vae=vae, # type: ignore
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    unet=base_unet, # type: ignore
    controlnet=controlnet, # type: ignore
    scheduler=scheduler,
)

# 0 -- openpose
# 1 -- depth
# 2 -- hed/pidi/scribble/ted
# 3 -- canny/lineart/anime_lineart/mlsd
# 4 -- normal
# 5 -- segment
base_pipeline = base_pipeline.to(ACTIVE_DEVICE)
seed = random.randint(0, 2147483647)
seed = 12398612
image = base_pipeline(
    prompt=PROMPT,
    neg_prompt=NEG_PROMPT,
    image_list=[controlnet_img, 0, 0, 0, 0, 0],
    generator=torch.Generator(ACTIVE_DEVICE).manual_seed(seed),
    width=new_width,
    height=new_height,
    num_inference_steps=40,
    union_control_type=torch.Tensor([1, 0, 0, 0, 0, 0]),
    guidance_scale=8.0
).images[0]
base_pipeline.to(IDLE_DEVICE)

# refiner_pipeline = StableDiffusionXLImg2ImgPipeline( # We need to code up different components fro this different UNet if we want to use the refiner Unet
#     vae=vae, # type: ignore
#     text_encoder=text_encoder,
#     text_encoder_2=text_encoder_2,
#     tokenizer=tokenizer,
#     tokenizer_2=tokenizer_2,
#     unet=base_unet, # type: ignore
#     scheduler=scheduler,
# ).to(torch.float16)
# refiner_pipeline = refiner_pipeline.to(ACTIVE_DEVICE)
# image = refiner_pipeline(PROMPT, image=image, num_inference_steps=40).images[0]

# refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
# refiner_pipeline = refiner_pipeline.to(ACTIVE_DEVICE)
# image = refiner_pipeline(PROMPT, image=image, num_inference_steps=40).images[0]
save_images([image])