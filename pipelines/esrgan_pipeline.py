import cv2
import math
import torch
import numpy as np
import torch.nn as nn
import PIL.Image as Image
from typing import Optional
from torch.nn import functional as F
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from models import RRDBNet

# Code referenced from https://github.com/xinntao/Real-ESRGAN
class ESRGANPipeline(DiffusionPipeline): # Inheriting for some basic functions, its not a diffusion pipeline.
  def __init__(
    self,
    rrdbnet: RRDBNet,
    tile_size: int = 0,
    tile_padding: int = 10,
    pre_padding: int = 10,
  ):
    super().__init__()
    self.scale = 4
    self.tile_size = tile_size
    self.tile_padding = tile_padding
    self.pre_padding = pre_padding
    self.register_modules(rrdbnet=rrdbnet)

  def process_tiles(self, image_array: torch.Tensor):
    # Modified from: https://github.com/ata4/esrgan-launcher
    batch_size, num_channels, height, width = image_array.shape
    output_height, output_width = height * self.scale, width * self.scale
    output_shape = (batch_size, num_channels, output_height, output_width)

    # Start with a black image
    output = image_array.new_zeros(output_shape)
    tiles_x = math.ceil(width / self.tile_size)
    tiles_y = math.ceil(height / self.tile_size)

    for y in range(tiles_y):
      for x in range(tiles_x):
        x_offset = x * self.tile_size
        y_offset = y * self.tile_size

        input_start_x = x_offset
        input_start_y = y_offset
        input_end_x = min(x_offset + self.tile_size, width)
        input_end_y = min(y_offset + self.tile_size, height)

        input_start_x_pad = max(input_start_x - self.tile_padding, 0)
        input_start_y_pad = max(input_start_y - self.tile_padding, 0)
        input_end_x_pad = min(input_end_x + self.tile_padding, width)
        input_end_y_pad = min(input_end_y + self.tile_padding, height)

        input_tile_width = input_end_x - input_start_x
        input_tile_height = input_end_y - input_start_y
        tile_idx = y * tiles_x + x + 1
        input_tile = image_array[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

        # Upscale tile
        output_tile = self.rrdbnet(input_tile)

        # Output tile area on total image
        output_start_x = input_start_x * self.scale
        output_start_y = input_start_y * self.scale
        output_end_x = input_end_x * self.scale
        output_end_y = input_end_y * self.scale

        # Output tile area without padding
        output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
        output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
        output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
        output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

        # Paste tile into output image
        output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
    return output

  @torch.no_grad()
  def __call__(
    self,
    image: Image.Image,
    outscale: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
  ):
    device = device if device is not None else next(self.rrdbnet.parameters()).device
    dtype = dtype if dtype is not None else next(self.rrdbnet.parameters()).dtype

    image_array = np.array(image).astype(np.float32)
    input_height, input_width = image_array.shape[:2]
    image_array = image_array / 255.0

    # Pre-process
    image_array = torch.from_numpy(np.transpose(image_array, (2, 0, 1))).float()
    image_array = image_array.unsqueeze(0).to(device=device, dtype=dtype)
    image_array = F.pad(image_array, (0, self.pre_padding, 0, self.pre_padding), "reflect") if self.pre_padding != 0 else image_array

    # Process
    if self.tile_size > 0:
      image_array = self.process_tiles(image_array)
    else:
      image_array = self.rrdbnet(image_array)

    # Post-process
    if self.pre_padding != 0:
      batch_size, num_channels, padded_height, padded_width = image_array.shape
      image_array = image_array[:, :, 0:int(padded_height - self.pre_padding * self.scale), 0:int(padded_width - self.pre_padding * self.scale)]

    image_array = image_array.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    image_array = np.transpose(image_array[[2, 1, 0], :, :], (1, 2, 0))
    image_array = (image_array * 255.0).round().astype(np.uint8)

    if outscale is not None and outscale != float(self.scale):
      image_array = cv2.resize(
        image_array,
        (int(input_width * outscale), int(input_height * outscale)),
        interpolation=cv2.INTER_LANCZOS4,
      )

    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_array)

    return pil_image