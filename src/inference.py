import os
import re

import numpy as np
import torch
from PIL import Image

from dataset import load_image
from model import UNet

TEST_IMAGES_DIR = "test_set_images"
PRED_DIR = "predictions"
MODEL_PATH = "best_model.pth"

os.makedirs(PRED_DIR, exist_ok=True)


# ---------------------------
# Utility functions
# ---------------------------


def save_mask(mask_np, path):
    """Save binary mask (0/1) as uint8 PNG (0 or 255)."""
    mask_img = (mask_np * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(path)


def extract_image_id(path):
    """
    Extracts image ID such as 'test_1' from:
        test_set_images/test_1/test_1.png
    """
    return os.path.splitext(os.path.basename(path))[0]


# ---------------------------
# Load single model
# ---------------------------


def load_model(device):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Train the model first.")

    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")
    return model


# ---------------------------
# TTA transforms
# ---------------------------


def tta_augmentations(img_np):
    """
    Returns 8 equivalent augmentations:
      original + 90°, 180°, 270° rotations
      mirrored + 90°, 180°, 270° rotations
    img_np shape: (H, W, 3)
    """
    mirrored = np.fliplr(img_np).copy()
    aug_list = [
        img_np,  # original
        np.rot90(img_np, k=1, axes=(0, 1)).copy(),  # original + 90°
        np.rot90(img_np, k=2, axes=(0, 1)).copy(),  # original + 180°
        np.rot90(img_np, k=3, axes=(0, 1)).copy(),  # original + 270°
        mirrored,  # mirrored
        np.rot90(mirrored, k=1, axes=(0, 1)).copy(),  # mirrored + 90°
        np.rot90(mirrored, k=2, axes=(0, 1)).copy(),  # mirrored + 180°
        np.rot90(mirrored, k=3, axes=(0, 1)).copy(),  # mirrored + 270°
    ]
    return aug_list


def undo_tta(mask_np_list):
    """
    Reverses TTA augmentations on predicted masks.
    mask_np_list must follow the same order as tta_augmentations.
    """
    reversed = [
        mask_np_list[0],
        np.rot90(mask_np_list[1], k=3, axes=(0, 1)),
        np.rot90(mask_np_list[2], k=2, axes=(0, 1)),
        np.rot90(mask_np_list[3], k=1, axes=(0, 1)),
        np.fliplr(mask_np_list[4]),
        np.fliplr(np.rot90(mask_np_list[5], k=3, axes=(0, 1))),
        np.fliplr(np.rot90(mask_np_list[6], k=2, axes=(0, 1))),
        np.fliplr(np.rot90(mask_np_list[7], k=1, axes=(0, 1))),
    ]
    return reversed


# ---------------------------
# Single-model TTA prediction
# ---------------------------


def predict_tta(model, img_np, device, threshold=0.5):
    """
    Apply TTA with flips using a single model.
    img_np: (H, W, 3) float32 in [0,1]
    Returns final binary mask (H, W) in {0,1}.
    """
    tta_imgs = tta_augmentations(img_np)
    tta_preds = []

    with torch.no_grad():
        for aug_img in tta_imgs:
            tensor = torch.from_numpy(aug_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            logits = model(tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W)
            tta_preds.append(probs)

    # Undo all flips
    restored = undo_tta(tta_preds)

    # Average all 8 restored predictions
    final_probs = np.mean(restored, axis=0)

    final_mask = (final_probs > threshold).astype(np.float32)
    return final_mask


# ---------------------------
# Main inference loop
# ---------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(device)

    # Collect test images
    test_image_paths = []
    for root, _, files in os.walk(TEST_IMAGES_DIR):
        for fname in files:
            if fname.endswith(".png"):
                test_image_paths.append(os.path.join(root, fname))

    # Sort numerically (test_1, test_2, ..., test_50)
    test_image_paths.sort(key=lambda p: int(re.search(r"\d+", p).group(0)))

    print(f"Found {len(test_image_paths)} test images.\n")

    for img_path in test_image_paths:
        image_id = extract_image_id(img_path)  # e.g. "test_1"
        print(f"Processing {image_id} ...")

        # Use dataset.py loader (no resizing, RGB, [0,1], HxWx3)
        img_np = load_image(img_path)

        mask = predict_tta(model, img_np, device, threshold=0.5)

        out_path = os.path.join(PRED_DIR, f"{image_id}.png")
        save_mask(mask, out_path)

        print(f"Saved: {out_path}")

    print("\nInference complete.")
    print(f"Predictions saved in: {PRED_DIR}\n")


if __name__ == "__main__":
    main()
