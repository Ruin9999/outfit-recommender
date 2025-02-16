import torch
from dataclasses import dataclass
from diffusers.utils.outputs import BaseOutput

@dataclass
class UNetOutput(BaseOutput):
    sample:  torch.Tensor