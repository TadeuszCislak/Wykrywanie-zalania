from pathlib import Path
import rasterio

mask = next(Path("data/train/masks").glob("*.tif"))

with rasterio.open(mask) as src:
    arr = src.read(1)

print(arr.min(), arr.max())
print(set(arr.flatten()[:1000]))