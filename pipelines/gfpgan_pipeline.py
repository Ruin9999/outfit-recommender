import cv2
import torch
import numpy as np
import torch.nn as nn
import PIL.Image as Image
from typing import Optional, Any
from torch.nn import functional as F
from torchvision.transforms.functional import normalize
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from pipelines import ESRGANPipeline
from models import RRDBNet, RestoreFormer

class GFPGANPipeline(DiffusionPipeline):
  def __init__(
    self,
    rrdbnet: RRDBNet,
    restoreformer: RestoreFormer,
    outscale: int = 2, # We place the outscale here since we need to load the face helper.
    device: Optional[str] = None,
  ):
    super().__init__()
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if device is None else device

    self.outscale = outscale
    self.register_modules(restoreformer=restoreformer, rrdbnet=rrdbnet)
    self.bg_upscaler = ESRGANPipeline(rrdbnet=rrdbnet) # The rrdbnet is fat as FUCK
    self.face_helper = FaceRestoreHelper(upscale_factor=outscale, use_parse=True, device=device)
    self.to(device)
    
  @torch.no_grad()
  def __call__(self, image: Any):
    self.face_helper.clean_all()
    self.face_helper.read_image(image)
    self.face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
    self.face_helper.align_warp_face()

    for cropped_face in self.face_helper.cropped_faces:
      cropped_face = cropped_face / 255.
      cropped_face = cropped_face.astype('float32')
      cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
      cropped_face = cropped_face.transpose(2, 0, 1)
      cropped_face_tensor = torch.from_numpy(cropped_face).float()
      cropped_face_tensor = normalize(cropped_face_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
      cropped_face_tensor = cropped_face_tensor.unsqueeze(0).to(device=next(self.restoreformer.encoder.parameters()).device)

      restored_face = self.restoreformer(cropped_face_tensor)
      restored_face = restored_face.squeeze(0).float().detach().cpu().clamp_(-1, 1)
      restored_face = (restored_face + 1) / (2)
      restored_face = restored_face.numpy()
      restored_face = restored_face.transpose(1, 2, 0)
      restored_face = cv2.cvtColor(restored_face, cv2.COLOR_RGB2BGR)
      restored_face = (restored_face * 255).round().astype(np.uint8)
      restored_face = restored_face.astype('uint8')

      self.face_helper.add_restored_face(restored_face)
      self.face_helper.get_inverse_affine(None)

      upscaled_bg = self.bg_upscaler(image, outscale=self.outscale)
      restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=upscaled_bg)

    return restored_img