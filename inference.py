import cv2
import torch
import numpy as np
import PIL.Image as Image
from controlnet_aux import OpenposeDetector

from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from utils import save_images
from models import ControlNetUnion, BaseUNet, AutoencoderKL, RefinerUNet, RRDBNet
from pipelines import StableDiffusionXLControlNetUnionPipeline, ESRGANPipeline

# Default values
PROMPT = "model in layered street style, standing against a vibrant graffiti wall, Vivid colors, Mirrorless, 28mm lens"
NEG_PROMPT = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature."
CONTROLNET_IMG_PATH = "controlnet_default.jpg"
ACTIVE_DEVICE="cuda:0"
IDLE_DEVICE="cpu"
IMG_OUTPUT_PATH="/outputs"
SEED=12398612

# Load models
scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", torch_dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", torch_dtype=torch.float16)
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", torch_dtype=torch.float16)
tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer", torch_dtype=torch.float16)
tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pose_processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

rrdbnet = RRDBNet.from_pretrained("safetensors/realesrgan_x4plus.safetensors")
controlnet = ControlNetUnion.from_pretrained("xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True)
base_unet = BaseUNet.from_pretrained("SG161222/RealVisXL_V4.0", subfolder="unet", torch_dtype=torch.float16)
refiner_unet = RefinerUNet.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", subfolder="unet", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

# Get pose image
controlnet_img = cv2.imread(CONTROLNET_IMG_PATH)
controlnet_img = pose_processor(controlnet_img, hand_and_face=False, output_type='cv2')

# Resize image for better performance
height, width, _ = controlnet_img.shape #type: ignore
ratio = np.sqrt(1024. * 1024. / (width * height))
new_width, new_height = int(width * ratio), int(height * ratio)
controlnet_img = cv2.resize(controlnet_img, (new_width, new_height)) #type: ignore
controlnet_img = Image.fromarray(controlnet_img)

base_pipeline = StableDiffusionXLControlNetUnionPipeline(
  vae=vae, #type: ignore
  text_encoder=text_encoder,
  text_encoder_2=text_encoder_2,
  tokenizer=tokenizer,
  tokenizer_2=tokenizer_2,
  unet=base_unet, #type: ignore
  controlnet=controlnet, #type: ignore
  scheduler=scheduler,
).to(device=ACTIVE_DEVICE)

print("Generating image...")
image = base_pipeline(
  prompt=PROMPT,
  neg_prompt=NEG_PROMPT,
  image_list=[controlnet_img, 0, 0, 0, 0, 0],
  generator=torch.Generator(ACTIVE_DEVICE).manual_seed(SEED),
  width=new_width,
  height=new_height,
  num_inference_steps=40,
  union_control_type=torch.Tensor([1, 0, 0, 0, 0, 0]),
  guidance_scale=8.0
).images[0]
base_pipeline.to(IDLE_DEVICE)

print("Upsampling...")
upsampler_pipeline = ESRGANPipeline(rrdbnet=rrdbnet).to(device=ACTIVE_DEVICE)
image = upsampler_pipeline(image, outscale=4)


save_images([image])