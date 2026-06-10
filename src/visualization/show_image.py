import matplotlib.pyplot as plt


def display_image(image, title="Sentinel Image"):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()