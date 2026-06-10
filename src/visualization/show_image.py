import matplotlib.pyplot as plt
import numpy as np


def display_image(image, title="Sentinel Image"):
    """
    image shape:
    (bands, height, width)
    """

    if image.ndim != 3:
        raise ValueError(
            f"Oczekiwano obrazu 3D, otrzymano {image.shape}"
        )

    bands, height, width = image.shape

    if bands >= 3:
        rgb = np.stack(
            [
                image[0],
                image[1],
                image[2]
            ],
            axis=-1
        )

        rgb = rgb.astype(np.float32)

        rgb_min = rgb.min()
        rgb_max = rgb.max()

        if rgb_max > rgb_min:
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min)

        plt.figure(figsize=(8, 8))
        plt.imshow(rgb)
        plt.title(title)
        plt.axis("off")
        plt.show()

    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(image[0], cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()