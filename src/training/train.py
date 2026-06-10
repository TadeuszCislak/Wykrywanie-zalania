import torch
from torch.utils.data import DataLoader

from src.datasets.flood_dataset import FloodDataset
from src.models.unet import UNetModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 1e-4


def train():

    dataset = FloodDataset(
        image_dir="data/images",
        mask_dir="data/masks"
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = UNetModel().to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, masks)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"Loss: {epoch_loss / len(loader):.4f}"
        )

    torch.save(
        model.state_dict(),
        "results/checkpoints/unet_flood.pth"
    )

    print("Model zapisany.")


if __name__ == "__main__":
    train()