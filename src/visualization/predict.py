from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch

from src.models.unet import UNetModel


IMAGE_SIZE = 256


def load_image(image_path):

    with rasterio.open(image_path) as src:
        image = src.read()

    image = np.transpose(image, (1, 2, 0))
    image = image.astype(np.float32)

    image = image / (image.max() + 1e-6)

    original_shape = image.shape[:2]

    image_resized = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    tensor = torch.tensor(
        image_resized,
        dtype=torch.float32
    ).permute(2, 0, 1)

    return tensor, image, original_shape


def create_rgb_preview(multispectral_image):
    """
    Sentinel-2:
    B2 = Blue
    B3 = Green
    B4 = Red

    indeksy:
    B1=0
    B2=1
    B3=2
    B4=3
    """

    rgb = multispectral_image[:, :, [3, 2, 1]]

    rgb = rgb.astype(np.float32)

    rgb -= rgb.min()
    rgb /= (rgb.max() + 1e-6)

    rgb = (rgb * 255).astype(np.uint8)

    return rgb


def predict():

    root = Path(__file__).resolve().parents[2]

    model_path = root / "results" / "unet_flood.pth"

    test_dir = root / "data" / "test" / "images"

    output_dir = root / "results" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = UNetModel(
        in_channels=13,
        out_channels=1
    ).to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    image_files = sorted(
        list(test_dir.glob("*.tif"))
    )

    print(f"Znaleziono {len(image_files)} obrazów.")

    with torch.no_grad():

        for image_path in image_files:

            print(f"Przetwarzanie: {image_path.name}")

            image_tensor, original_image, original_shape = load_image(
                image_path
            )

            image_tensor = image_tensor.unsqueeze(0).to(device)

            prediction = model(image_tensor)

            prediction = torch.sigmoid(prediction)

            prediction = prediction.squeeze().cpu().numpy()

            prediction = cv2.resize(
                prediction,
                (original_shape[1], original_shape[0])
            )

            mask = (prediction > 0.5).astype(np.uint8)

            mask_path = (
                output_dir /
                image_path.name.replace(
                    ".tif",
                    "_mask.png"
                )
            )

            cv2.imwrite(
                str(mask_path),
                mask * 255
            )

            rgb = create_rgb_preview(
                original_image
            )

            overlay = rgb.copy()

            overlay[mask == 1] = [255, 0, 0]

            visualization = cv2.addWeighted(
                rgb,
                0.7,
                overlay,
                0.3,
                0
            )

            vis_path = (
                output_dir /
                image_path.name.replace(
                    ".tif",
                    "_overlay.png"
                )
            )

            cv2.imwrite(
                str(vis_path),
                cv2.cvtColor(
                    visualization,
                    cv2.COLOR_RGB2BGR
                )
            )

    print("\nGotowe.")
    print(f"Wyniki zapisano do:\n{output_dir}")


if __name__ == "__main__":
    predict()