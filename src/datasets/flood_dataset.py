from pathlib import Path

import rasterio
from torch.utils.data import Dataset


class FloodDataset(Dataset):

    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)

        self.images = sorted(
            list(self.image_dir.glob("*.tif")) +
            list(self.image_dir.glob("*.tiff"))
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        with rasterio.open(image_path) as src:
            image = src.read()

        return image, image_path