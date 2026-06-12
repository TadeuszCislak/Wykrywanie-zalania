from pathlib import Path

import rasterio
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class FloodDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_size=256):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_size = image_size

        self.images = sorted(list(self.image_dir.glob("*.tif")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        mask_name = image_path.name.replace(
            "_S2Hand.tif",
            "_LabelHand.tif"
        )

        mask_path = self.mask_dir / mask_name

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Brak maski dla obrazu:\n"
                f"{image_path.name}\n"
                f"Szukałem:\n"
                f"{mask_path}"
            )

        with rasterio.open(image_path) as src:
            image = src.read()

        image = np.transpose(image, (1, 2, 0))

        image = image.astype(np.float32)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        image = image / (image.max() + 1e-6)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        mask = (mask == 1).astype(np.float32)

        image = torch.tensor(image).permute(2, 0, 1)

        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask