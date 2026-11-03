import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from augmentations import get_val_transform_mp_safe
from dataset import RoadSegmentationDataset
from metrics import dice_score
from model import UNet
from train import load_ids

MODEL_PATH = "best_model.pth"


def load_model(device: torch.device) -> UNet:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found.")

    model = UNet(in_channels=3, out_channels=1)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_thresholds(
    model: UNet,
    val_loader: DataLoader,
    thresholds: np.ndarray,
    device: torch.device,
):
    num_samples = len(val_loader.dataset)

    best_dice = -1.0
    best_dice_thr = None

    for thr in thresholds:
        total_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)

                batch_size = images.size(0)
                total_dice += dice_score(logits, masks, threshold=float(thr)) * batch_size

        mean_dice = total_dice / num_samples

        print(f"Threshold {thr:.3f} -> Dice: {mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_dice_thr = float(thr)

    return best_dice_thr, best_dice


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the same split as in train.py
    _, val_ids = load_ids(val_ratio=0.1, seed=42)
    print(f"Validation samples: {len(val_ids)}")

    val_dataset = RoadSegmentationDataset(val_ids, transform=get_val_transform_mp_safe())
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    model = load_model(device)
    print(f"Loaded model from {MODEL_PATH}")

    thresholds = np.linspace(0.0, 1.0, 200)
    print(f"Evaluating {len(thresholds)} thresholds between {thresholds[0]:.2f} and {thresholds[-1]:.2f}")

    best_dice_thr, best_dice = evaluate_thresholds(model, val_loader, thresholds, device)

    print("\nBest thresholds on validation set:")
    print(f"- Best Dice threshold: {best_dice_thr:.3f} (Dice = {best_dice:.4f})")


if __name__ == "__main__":
    main()
