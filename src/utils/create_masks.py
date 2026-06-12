from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw


IMAGES_DIR = Path("data/images")
LABELS_DIR = Path("data/labels")
MASKS_DIR = Path("data/masks")

MASKS_DIR.mkdir(exist_ok=True)


for image_path in IMAGES_DIR.glob("*.png"):

    label_path = LABELS_DIR / (
        image_path.stem + ".geojson"
    )

    if not label_path.exists():
        continue

    image = Image.open(image_path)

    width, height = image.size

    mask = Image.new(
        "L",
        (width, height),
        0
    )

    draw = ImageDraw.Draw(mask)

    gdf = gpd.read_file(label_path)

    for geometry in gdf.geometry:

        coords = list(
            geometry.exterior.coords
        )

        draw.polygon(
            coords,
            fill=255
        )

    mask.save(
        MASKS_DIR /
        f"{image_path.stem}.png"
    )