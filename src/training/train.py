from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.flood_dataset import FloodDataset
from src.models.unet import UNetModel


def train():

    root = Path(__file__).resolve().parents[2]

    train_images = root / "data" / "train" / "images"
    train_masks = root / "data" / "train" / "masks"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = FloodDataset(
        train_images,
        train_masks,
        image_size=256
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    model = UNetModel(
        in_channels=13,
        out_channels=1
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    epochs = 10

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, masks)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {running_loss/len(loader):.4f}"
        )

    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        results_dir / "unet_flood.pth"
    )

    print("Model zapisany.")


if __name__ == "__main__":
    train()