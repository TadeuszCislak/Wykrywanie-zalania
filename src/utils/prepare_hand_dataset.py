# src/utils/prepare_hand_dataset.py

from pathlib import Path
import shutil

from sklearn.model_selection import train_test_split


ROOT = Path(
    r"C:\Users\Komputer\.cache\kagglehub\datasets\robertomarinoformica\sen1floods11-dataset\versions\4"
)

IMAGE_DIR = ROOT / "HandLabeled" / "S2Hand"
MASK_DIR = ROOT / "HandLabeled" / "LabelHand"

OUTPUT = Path("data")

pairs = []

for image_path in IMAGE_DIR.glob("*.tif"):

    base_name = (
        image_path.name
        .replace("_S2Hand.tif", "")
    )

    mask_name = (
        base_name +
        "_LabelHand.tif"
    )

    mask_path = MASK_DIR / mask_name

    if mask_path.exists():
        pairs.append(
            (image_path, mask_path)
        )

print("Znaleziono par:", len(pairs))

train_pairs, test_pairs = train_test_split(
    pairs,
    test_size=0.15,
    random_state=42
)

train_pairs, val_pairs = train_test_split(
    train_pairs,
    test_size=0.15,
    random_state=42
)

splits = {
    "train": train_pairs,
    "val": val_pairs,
    "test": test_pairs
}

for split_name, split_pairs in splits.items():

    image_out = OUTPUT / split_name / "images"
    mask_out = OUTPUT / split_name / "masks"

    image_out.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_out.mkdir(
        parents=True,
        exist_ok=True
    )

    for image_path, mask_path in split_pairs:

        shutil.copy2(
            image_path,
            image_out / image_path.name
        )

        shutil.copy2(
            mask_path,
            mask_out / mask_path.name
        )

print("Dataset przygotowany.")