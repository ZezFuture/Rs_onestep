"""The dataset used by Z-Image train_tiny.py, with its actual degradation path."""

import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

from . import basicsr_compat  # noqa: F401; must precede the copied BasicSR imports
from .realesrgan import RealESRGAN_degradation


class BlindSRDataset(Dataset):
    """Read HR images; degrade each crop with the copied RealESRGAN implementation."""

    def __init__(self, image_dir: str | list[str], resolution: int):
        self.images = []
        for image_root in image_dir if isinstance(image_dir, list) else [image_dir]:
            for root, _, files in os.walk(image_root):
                for name in files:
                    if name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.images.append(os.path.join(root, name))
        self.images.sort()
        if not self.images:
            raise ValueError(f"No JPG or PNG images found under {image_dir}")
        if resolution < 64 or resolution % 64:
            raise ValueError("resolution must be a positive multiple of 64")
        self.resolution = resolution
        # Identical config and CPU execution to MyDataset_blind_plus.
        self.degradation = RealESRGAN_degradation("params_realesrgan_seesr.yml", device="cpu")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]
        target = cv2.imread(filename)
        if target is None:
            raise ValueError(f"Could not read {filename}")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        height, width, _ = target.shape
        # This follows MyDataset_blind_plus(deterministic_k=False) exactly.
        if height >= self.resolution and width >= self.resolution:
            x = random.randint(0, width - self.resolution)
            y = random.randint(0, height - self.resolution)
            target = target[y:y + self.resolution, x:x + self.resolution]
        else:
            target = cv2.resize(target, (self.resolution, self.resolution), cv2.INTER_CUBIC)
        target, source = self.degradation.degrade_process(
            np.asarray(target) / 255.0, resize_bak=True
        )
        return {
            "output_pixel_values": target.squeeze(0) * 2 - 1,
            "conditioning_pixel_values": source.squeeze(0) * 2 - 1,
        }
