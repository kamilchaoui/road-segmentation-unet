import os
import random

import torch
from torch import optim
from torch.utils.data import DataLoader

from augmentations import get_train_transform, get_val_transform
from dataset import RoadSegmentationDataset, list_image_ids
from losses import bce_dice_loss
from metrics import dice_score, iou_score
from model import UNet


def load_ids(val_ratio=0.1, seed=1):
    """
    Load all image ids and split into train/val.

    val_ratio: fraction of data used for validation (default 0.1 -> 90/10 split)
    """
    all_ids = list_image_ids()
    ids = all_ids.copy()
    random.Random(seed).shuffle(ids)  # shuffle to avoid ordering bias

    n = len(ids)
    n_val = max(1, int(val_ratio * n))
    n_train = n - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    return train_ids, val_ids


def _main(seed):
    """
    Main training loop for a single model.

    This function handles the entire training process for a single model,
    including data loading, model initialization, training, validation,
    and model saving.

    Args:
        seed (int): The random seed for this training run.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"({seed=:03}) Using device: {device}")

    # -----------------------------
    # 1. Get a single 90/10 split
    # -----------------------------
    train_ids, val_ids = load_ids(val_ratio=0.1, seed=seed)
    print(f"({seed=:03}) Training samples: {len(train_ids)}, Validation samples: {len(val_ids)}")

    # -----------------------------
    # 2. Datasets & loaders
    # -----------------------------
    train_dataset = RoadSegmentationDataset(train_ids, transform=get_train_transform())
    val_dataset = RoadSegmentationDataset(val_ids, transform=get_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=6)

    # -----------------------------
    # 3. Model & optimizer
    # -----------------------------
    model = UNet(in_channels=3, out_channels=1).to(device)
    if os.path.exists("base_model.pth"):
        model.load_state_dict(torch.load("base_model.pth", map_location=device))
        optimizer = optim.SGD(model.parameters(), lr=1e-12)
        print(f"({seed=:03}) Loaded base model for continued training.")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        print(f"({seed=:03}) Training model from scratch.")

    max_epochs = 400

    best_val_loss = 10.0
    best_val_dice = 0.0
    patience = 20

    # -----------------------------
    # 4. Training loop
    # -----------------------------
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)

            loss = bce_dice_loss(logits, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                val_batch_loss = bce_dice_loss(logits, masks)

                val_loss += val_batch_loss.item() * images.size(0)
                val_dice += dice_score(logits, masks) * images.size(0)
                val_iou += iou_score(logits, masks) * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)

        print(
            f"({seed=:03}) | "
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Dice: {val_dice:.4f} | IoU: {val_iou:.4f}"
        )

        # -------- Best model saving --------
        os.makedirs("models", exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"models/model_loss_{seed:03}.pth")
            print(f"({seed=:03}) Saved new best loss model!")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), f"models/model_dice_{seed:03}.pth")
            print(f"({seed=:03}) Saved new best dice model!")

        # -------- Early stopping --------
        if epoch > 20:
            if val_loss > best_val_loss:
                patience -= 1
                if patience == 0:
                    print(f"({seed=:03}) Early stopping at epoch {epoch:02d}.")
                    break
            else:
                patience = 20  # reset patience

    print(f"({seed=:03}) Training complete. {best_val_loss=:.4f} {best_val_dice=:.4f}\n")


if __name__ == "__main__":
    for seed in range(65):
        _main(seed)
