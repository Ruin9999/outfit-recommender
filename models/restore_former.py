# Code referenced from https://github.com/TencentARC/GFPGAN
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional, Dict
from safetensors.torch import load_file

from blocks import ResidualBlock, BasicAttention, Upsample

# Separete implementation since this is implemented slightly differently from our implementation
class Downsample(nn.Module):
  def __init__(self, in_channels: int, out_channels: int) -> None:
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    pad = (0, 1, 0, 1) # L R U D
    x = F.pad(x, pad, mode="constant", value=0)
    x = self.conv(x)

    return x

# Copied from https://github.com/TencentARC/GFPGAN/blob/master/gfpgan/archs/restoreformer_arch.py
class VectorQuantizer(nn.Module):
  """
  see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
  ____________________________________________
  Discretization bottleneck part of the VQ-VAE.
  Inputs:
  - n_e : number of embeddings
  - e_dim : dimension of embedding
  - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
  _____________________________________________
  """

  def __init__(self, n_e: int, e_dim: int, beta: float):
    super().__init__()
    self.n_e = n_e
    self.e_dim = e_dim
    self.beta = beta

    self.embedding = nn.Embedding(self.n_e, self.e_dim)
    self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
  
  def forward(self, z: torch.Tensor):
    """
    Inputs the output of the encoder network z and maps it to a discrete
    one-hot vector that is the index of the closest embedding vector e_j
    z (continuous) -> z_q (discrete)
    z.shape = (batch, channel, height, width)
    quantization pipeline:
        1. get encoder input (B,C,H,W)
        2. flatten input to (B*H*W,C)
    """
    z = z.permute(0, 2, 3, 1).contiguous()
    z_flattened = z.view(-1, self.e_dim)

    d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
        torch.matmul(z_flattened, self.embedding.weight.t())

    min_value, min_encoding_indices = torch.min(d, dim=1)
    min_encoding_indices = min_encoding_indices.unsqueeze(1)

    min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
    min_encodings.scatter_(1, min_encoding_indices, 1)

    z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
    loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)
    z_q = z + (z_q - z).detach()

    e_mean = torch.mean(min_encodings, dim=0)
    perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

    z_q = z_q.permute(0, 3, 1, 2).contiguous()
    return z_q, loss, (perplexity, min_encodings, min_encoding_indices, d)
    
class TransformerEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_in = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

    # Down blocks
    self.down = nn.ModuleList([])

    down_block_1 = nn.Module()
    down_block_1.attn = nn.ModuleList([])
    down_block_1.block = nn.ModuleList([])
    down_block_1.block.append(ResidualBlock(in_channels=64, out_channels=64))
    down_block_1.block.append(ResidualBlock(in_channels=64, out_channels=64))
    down_block_1.downsample = Downsample(in_channels=64, out_channels=64) # Resolution: 512 / 2 = 256
    self.down.append(down_block_1)

    down_block_2 = nn.Module()
    down_block_2.attn = nn.ModuleList([])
    down_block_2.block = nn.ModuleList([])
    down_block_2.block.append(ResidualBlock(in_channels=64, out_channels=128))
    down_block_2.block.append(ResidualBlock(in_channels=128, out_channels=128))
    down_block_2.downsample = Downsample(in_channels=128, out_channels=128) # Resolution: 256 / 2 = 128
    self.down.append(down_block_2)

    down_block_3 = nn.Module()
    down_block_3.attn = nn.ModuleList([])
    down_block_3.block = nn.ModuleList([])
    down_block_3.block.append(ResidualBlock(in_channels=128, out_channels=128))
    down_block_3.block.append(ResidualBlock(in_channels=128, out_channels=128))
    down_block_3.downsample = Downsample(in_channels=128, out_channels=128) # Resolution: 128 / 2 = 64
    self.down.append(down_block_3)

    down_block_4 = nn.Module()
    down_block_4.attn = nn.ModuleList([])
    down_block_4.block = nn.ModuleList([])
    down_block_4.block.append(ResidualBlock(in_channels=128, out_channels=256))
    down_block_4.block.append(ResidualBlock(in_channels=256, out_channels=256))
    down_block_4.downsample = Downsample(in_channels=256, out_channels=256) # Resolution: 64 / 2 = 32
    self.down.append(down_block_4)

    down_block_5 = nn.Module()
    down_block_5.attn = nn.ModuleList([])
    down_block_5.block = nn.ModuleList([])
    down_block_5.block.append(ResidualBlock(in_channels=256, out_channels=256))
    down_block_5.block.append(ResidualBlock(in_channels=256, out_channels=256))
    down_block_5.downsample = Downsample(in_channels=256, out_channels=256) # Resolution: 32/ 2 = 16
    self.down.append(down_block_5)

    # NOTE: This layer has attention, but why only 8 heads? hmm...
    down_block_6 = nn.Module()
    down_block_6.attn = nn.ModuleList([])
    down_block_6.block = nn.ModuleList([])
    down_block_6.block.append(ResidualBlock(in_channels=256, out_channels=512))
    down_block_6.block.append(ResidualBlock(in_channels=512, out_channels=512))
    down_block_6.attn.append(BasicAttention(in_channels=512, out_channels=512, num_heads=8))
    down_block_6.attn.append(BasicAttention(in_channels=512, out_channels=512, num_heads=8))
    self.down.append(down_block_6)

    # Mid blocks
    self.mid = nn.Module()
    self.mid.block_1 = ResidualBlock(in_channels=512, out_channels=512)
    self.mid.attn_1 = BasicAttention(in_channels=512, out_channels=512, num_heads=8)
    self.mid.block_2 = ResidualBlock(in_channels=512, out_channels=512)

    # Out projection
    self.norm_out = nn.GroupNorm(32, 512, eps=1e-6, affine=True)
    self.conv_out = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    attention_outputs = {}
    x = self.conv_in(x)

    # Downsample
    for down_block in self.down:
      for block_index in range(len(down_block.block)):
        x = down_block.block[block_index](x)
        if len(down_block.attn) > 0:
          x = down_block.attn[block_index](x)

      if hasattr(down_block, 'downsample') and down_block.downsample:
        x = down_block.downsample(x)

    # Mid
    x = self.mid.block_1(x)
    attention_outputs['pre-mid'] = x
    x = self.mid.attn_1(x)
    x = self.mid.block_2(x)
    attention_outputs['post-mid'] = x

    # Out
    x = self.norm_out(x)
    x = F.silu(x)
    x = self.conv_out(x)

    return x, attention_outputs

class TransformerDecoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_in = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    # Mid blocks
    self.mid = nn.Module()
    self.mid.block_1 = ResidualBlock(in_channels=512, out_channels=512)
    self.mid.attn_1 = BasicAttention(in_channels=512, out_channels=512, num_heads=8)
    self.mid.block_2 = ResidualBlock(in_channels=512, out_channels=512)

    # Up blocks
    self.up = nn.ModuleList([])

    up_block_1 = nn.Module()
    up_block_1.attn = nn.ModuleList([])
    up_block_1.block = nn.ModuleList([])
    up_block_1.block.append(ResidualBlock(in_channels=512, out_channels=512))
    up_block_1.block.append(ResidualBlock(in_channels=512, out_channels=512))
    up_block_1.block.append(ResidualBlock(in_channels=512, out_channels=512))
    up_block_1.attn.append(BasicAttention(in_channels=512, out_channels=512, num_heads=8))
    up_block_1.attn.append(BasicAttention(in_channels=512, out_channels=512, num_heads=8))
    up_block_1.attn.append(BasicAttention(in_channels=512, out_channels=512, num_heads=8))
    up_block_1.upsample = Upsample(in_channels=512, out_channels=512) # Resolution: 16 * 2 = 32
    self.up.append(up_block_1)

    up_block_2 = nn.Module()
    up_block_2.attn = nn.ModuleList([])
    up_block_2.block = nn.ModuleList([])
    up_block_2.block.append(ResidualBlock(in_channels=512, out_channels=256))
    up_block_2.block.append(ResidualBlock(in_channels=256, out_channels=256))
    up_block_2.block.append(ResidualBlock(in_channels=256, out_channels=256))
    up_block_2.upsample = Upsample(in_channels=256, out_channels=256) # Resolution: 32 * 2 = 64
    self.up.append(up_block_2)

    up_block_3 = nn.Module()
    up_block_3.attn = nn.ModuleList([])
    up_block_3.block = nn.ModuleList([])
    up_block_3.block.append(ResidualBlock(in_channels=256, out_channels=256))
    up_block_3.block.append(ResidualBlock(in_channels=256, out_channels=256))
    up_block_3.block.append(ResidualBlock(in_channels=256, out_channels=256))
    up_block_3.upsample = Upsample(in_channels=256, out_channels=256) # Resolution: 64 * 2 = 128
    self.up.append(up_block_3)

    up_block_4 = nn.Module()
    up_block_4.attn = nn.ModuleList([])
    up_block_4.block = nn.ModuleList([])
    up_block_4.block.append(ResidualBlock(in_channels=256, out_channels=128))
    up_block_4.block.append(ResidualBlock(in_channels=128, out_channels=128))
    up_block_4.block.append(ResidualBlock(in_channels=128, out_channels=128))
    up_block_4.upsample = Upsample(in_channels=128, out_channels=128) # Resolution: 128 * 2 = 256
    self.up.append(up_block_4)

    up_block_5 = nn.Module()
    up_block_5.attn = nn.ModuleList([])
    up_block_5.block = nn.ModuleList([])
    up_block_5.block.append(ResidualBlock(in_channels=128, out_channels=128))
    up_block_5.block.append(ResidualBlock(in_channels=128, out_channels=128))
    up_block_5.block.append(ResidualBlock(in_channels=128, out_channels=128))
    up_block_5.upsample = Upsample(in_channels=128, out_channels=128) # Resolution: 256 * 2 = 512
    self.up.append(up_block_5)

    up_block_6 = nn.Module()
    up_block_6.attn = nn.ModuleList([])
    up_block_6.block = nn.ModuleList([])
    up_block_6.block.append(ResidualBlock(in_channels=128, out_channels=64))
    up_block_6.block.append(ResidualBlock(in_channels=64, out_channels=64))
    up_block_6.block.append(ResidualBlock(in_channels=64, out_channels=64))
    self.up.append(up_block_6)

    self.norm_out = nn.GroupNorm(32, 64, eps=1e-6, affine=True)
    self.conv_out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

  def forward(self, x: torch.Tensor, y: Dict[str, torch.Tensor]) -> torch.Tensor:
    x = self.conv_in(x)

    # Mid
    x = self.mid.block_1(x)
    x = self.mid.attn_1(x, y['post-mid'])
    x = self.mid.block_2(x)

    # Upsample
    for up_block in self.up:
      for block_index in range(len(up_block.block)):
        x = up_block.block[block_index](x)
        if len(up_block.attn) > 0:
          x = up_block.attn[block_index](x, y['pre-mid'])
      
      if hasattr(up_block, 'upsample') and up_block.upsample:
        x = up_block.upsample(x)

    # Out
    x = self.norm_out(x)
    x = F.silu(x)
    x = self.conv_out(x)

    return x

class RestoreFormer(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = TransformerEncoder()
    self.decoder = TransformerDecoder()
    self.quantize = VectorQuantizer(1024, 256, beta=0.25)

    self.quant_conv = nn.Conv2d(256, 256, kernel_size=1, padding=0)
    self.post_quant_conv = nn.Conv2d(256, 256, kernel_size=1, padding=0)

    # Since fix_codebook is True
    for _, param in self.quantize.named_parameters():
      param.requires_grad = False

  def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    x, attention_outputs = self.encoder(x)
    x = self.quant_conv(x)
    quantized_x, _, _ = self.quantize(x)
    
    return quantized_x, attention_outputs

  def decode(self, x: torch.Tensor, attention_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    x = self.post_quant_conv(x)
    x = self.decoder(x, attention_outputs)

    return x

  def forward(self, x: torch.Tensor):
    x, attention_outputs = self.encode(x)
    x = self.decode(x, attention_outputs)

    return x

  @classmethod
  def from_pretrained(cls, model_dir_or_path: str, prefix: str = "params", **kwargs):
    model = cls(**kwargs)

    state_dict = load_file(model_dir_or_path)
    new_state_dict = {}
    prefix = prefix + '_'  # e.g., "params_"
    for key, value in state_dict.items():
      if key.startswith(prefix):
        new_key = key[len(prefix):]
      else:
        new_key = key
      new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model
  
  def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> nn.Module:
    new_model = super().to(device=device, dtype=dtype, **kwargs)
    if dtype is not None: self._dtype = dtype

    return new_model

  @property
  def dtype(self) -> torch.dtype:
    return self._dtype if hasattr(self, "_dtype") else next(self.parameters()).dtype

  @dtype.setter
  def dtype(self, dtype: torch.dtype) -> None:
    self.to(dtype=dtype)