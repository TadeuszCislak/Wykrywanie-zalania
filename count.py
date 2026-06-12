from pathlib import Path
import rasterio

image = next(Path("data/train/images").glob("*.tif"))

with rasterio.open(image) as src:
    print("Kanały:", src.count)
    print("Rozmiar:", src.width, src.height)