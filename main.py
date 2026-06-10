from src.datasets.flood_dataset import FloodDataset
from src.visualization.show_image import display_image


def main():
    dataset = FloodDataset("data/raw/sentinel2")

    print(f"Liczba znalezionych obrazów: {len(dataset)}")

    if len(dataset) == 0:
        print("Brak obrazów w katalogu.")
        return

    image, path = dataset[0]

    print(f"Wyświetlanie: {path}")

    display_image(image, title=path.name)


if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
