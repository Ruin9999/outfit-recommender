# import io
# import litserve as ls
# import PIL.Image as Image
# class StableDiffusionLitAPI(ls.LitAPI):
#     def setup(self, device):
#         self.model1 = lambda x: x**2
#         self.model2 = lambda x: x**3

#     def decode_request(self, request):
#       return request["input"]

#     def predict(self, x):
#         return x.get("debug")

#     def encode_response(self, output):
#       result_image = Image.new("RGB", (512, 512), color="blue")
#       buffer = io.BytesIO()
#       result_image.save(buffer, format="PNG")
#       buffer.seek(0)
#       return 
    
# if __name__ == "__main__":
#     api = StableDiffusionLitAPI()
#     server = ls.LitServer(api, accelerator="auto")
#     server.run(port=8000)

import io
import cv2
import torch
import base64
import random
import requests
import numpy as np
import litserve as ls
import PIL.Image as Image

from controlnet_aux import OpenposeDetector
from models import ControlNetUnion, BaseUNet, AutoencoderKL
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from stable_diffusion_controlnet_pipeline import StableDiffusionXLControlNetUnionPipeline
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler

# DEFAULT VALUES
PROMPT="Model in layered street style, standing against a vibrant graffiti wall, Vivid colors, Mirrorless, 28mm lens"
NEG_PROMPT="out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
CONTROLNET_IMG_PATH="controlnet_default.jpg"

class StableDiffusionLitAPI(ls.LitAPI):
  def setup(self, device):
    self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", torch_dtype=torch.float16)
    self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder", torch_dtype=torch.float16)
    self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2", torch_dtype=torch.float16)
    self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer", torch_dtype=torch.float16)
    self.tokenizer_2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2", torch_dtype=torch.float16)
    self.base_unet = BaseUNet.from_pretrained("SG161222/RealVisXL_V4.0", subfolder="unet", torch_dtype=torch.float16)
    self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    self.controlnet = ControlNetUnion.from_pretrained("xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True)
    self.pose_processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    self.pipeline = StableDiffusionXLControlNetUnionPipeline(
      vae=self.vae, #type: ignore
      text_encoder=self.text_encoder,
      text_encoder_2=self.text_encoder_2,
      tokenizer=self.tokenizer,
      tokenizer_2=self.tokenizer_2,
      unet=self.base_unet, #type: ignore
      controlnet=self.controlnet, #type: ignore
      scheduler=self.scheduler,
    ).to(device) # Moving to CPU should cause some issues with fp16.

  def decode_request(self, request):# -> Any:# -> Any:
    return request["input"]
  
  def predict(self, arguments):
    prompt = arguments.get("prompt", PROMPT)
    neg_prompt = arguments.get("neg_prompt", NEG_PROMPT)
    num_inference_steps = arguments.get("num_inference_steps", 40)
    guidance_scale = arguments.get("guidance_scale", 8.0)
    controlnet_image_url = arguments.get("controlnet_image_url", CONTROLNET_IMG_PATH)
    seed = arguments.get("seed", random.randint(0, 2147483647))
    log = arguments.get("log", False)

    # Download controlnet image
    if controlnet_image_url.startswith("http"):
      response = requests.get(controlnet_image_url)
      response.raise_for_status()
      image_array = np.frombuffer(response.content, dtype=np.uint8)
      controlnet_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR) # Getting an error saying that this image has a size of 85 instead of 86, but where the fuck is throwing that error? I dont know
      if controlnet_image is None: raise ValueError("Invalid image")
    else:
      controlnet_image = cv2.imread(controlnet_image_url, cv2.IMREAD_COLOR)

    # Pre-process controlnet image
    controlnet_image = self.pose_processor(controlnet_image, hand_and_face=False, output_type='cv2')
    height, width, _ = controlnet_image.shape #type: ignore
    ratio = np.sqrt(1024. * 1024. / (width * height))
    new_width, new_height = int(width * ratio), int(height * ratio)
    controlnet_image = cv2.resize(controlnet_image, (new_width, new_height)) #type: ignore
    controlnet_image = Image.fromarray(controlnet_image)

    # TODO: If log, store controlnet_image

    # Run inference
    image = self.pipeline(
      prompt=prompt,
      neg_prompt=neg_prompt,
      image_list=[controlnet_image, 0, 0, 0, 0, 0],
      generator=torch.Generator("cuda:0").manual_seed(seed),
      width=new_width,
      height=new_height,
      num_inference_steps=num_inference_steps,
      union_control_type=torch.Tensor([1, 0, 0, 0, 0, 0]),
      guidance_scale=guidance_scale
    ).images[0]

    # TODO: If log, store image

    # Send image back to client
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_bytes = buffer.read()
    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
    
    return encoded_string

  def encode_response(self, output):
    return {"output": output}
  
if __name__ == "__main__":
  api = StableDiffusionLitAPI()
  server = ls.LitServer(api, accelerator="auto")
  server.run(port=8000)