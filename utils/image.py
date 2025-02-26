import os
import torch
import numpy as np
import PIL.Image as Image
from datetime import datetime
from typing import List, Union, Optional

def save_images(images: List[Union[torch.Tensor, np.ndarray, Image.Image]], output_dir: str = "outputs", filename: Optional[str] = None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, img in enumerate(images):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{idx}.png" if filename is None else f"{filename}_{idx}.png"
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

def get_timestamp():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestamp