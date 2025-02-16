import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from safetensors.torch import load_file
from blocks import ResidualResidualDenseBlock

# Referenced from https://github.com/XPixelGroup/BasicSR , RRDB x4 model
class RRDBNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv_first = nn.Conv2d(3, 64, kernel_size=3, padding=1)

    # Body
    self.body = nn.ModuleList([])
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 1
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 2
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 3
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 4
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 5
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 6
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 7
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 8
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 9
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 10
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 11
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 12
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 13
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 14
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 15
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 16
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 17
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 18
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 19
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 20
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 21
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 22
    self.body.append(ResidualResidualDenseBlock(64, 32)) # 23
    self.conv_body = nn.Conv2d(64, 64, kernel_size=3, padding=1)

    # Upsampling
    self.conv_up1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.conv_up2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.conv_hr = nn.Conv2d(64, 64, kernel_size=3, padding=1)
    self.conv_last = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv_first(x)
    
    residual = x
    for layer in self.body:
      x = layer(x)
    x = self.conv_body(x)
    x += residual

    x = F.interpolate(x, scale_factor=2, mode="nearest")
    x = self.conv_up1(x)
    x = self.lrelu(x)

    x = F.interpolate(x, scale_factor=2, mode="nearest")
    x = self.conv_up2(x)
    x = self.lrelu(x)

    x = self.conv_hr(x)
    x = self.lrelu(x)
    x = self.conv_last(x) 

    return x

  @classmethod
  def from_pretrained(cls, model_dir_or_path: str, prefix: str = "params_ema", **kwargs):
    model = cls(**kwargs)
    
    state_dict = load_file(model_dir_or_path)
    new_state_dict = {}
    prefix = prefix + '_'  # e.g., "params_ema_"
    for key, value in state_dict.items():
      if key.startswith(prefix):
        new_key = key[len(prefix):]
      else:
        new_key = key
      new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    return model