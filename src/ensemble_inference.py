import gc
import os
import pickle
import re

import numpy as np
import torch
from PIL import Image

from dataset import load_image
from model import UNet

# ---------------------------
# Configuration
# ---------------------------

TEST_IMAGES_DIR = "test_set_images"
PRED_DIR = "predictions"
MODELS_DIR = "./models"

USE_MAJORITY_VOTE = False
PIXEL_THRESHOLD = 0.73

os.makedirs(PRED_DIR, exist_ok=True)


# ---------------------------
# Utility functions
# ---------------------------


def save_mask(mask_np, path):
    """Save binary mask (0/1) as uint8 PNG (0 or 255)."""
    mask_img = (mask_np * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(path)


def extract_image_id(path):
    """Extracts image ID such as 'test_1' from path."""
    return os.path.splitext(os.path.basename(path))[0]


# ---------------------------
# Model Loading
# ---------------------------


def load_model_from_path(model_path, device):
    """Loads a specific model file onto the device."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found.")

    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model: {os.path.basename(model_path)}")
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
        np.rot90(img_np, k=1, axes=(0, 1)).copy(),  # 90°
        np.rot90(img_np, k=2, axes=(0, 1)).copy(),  # 180°
        np.rot90(img_np, k=3, axes=(0, 1)).copy(),  # 270°
        mirrored,  # mirrored
        np.rot90(mirrored, k=1, axes=(0, 1)).copy(),  # mirrored + 90°
        np.rot90(mirrored, k=2, axes=(0, 1)).copy(),  # mirrored + 180°
        np.rot90(mirrored, k=3, axes=(0, 1)).copy(),  # mirrored + 270°
    ]
    return aug_list


def undo_tta(mask_np_list):
    """
    Reverses TTA augmentations on predicted masks.
    """
    reversed_masks = [
        mask_np_list[0],
        np.rot90(mask_np_list[1], k=3, axes=(0, 1)),
        np.rot90(mask_np_list[2], k=2, axes=(0, 1)),
        np.rot90(mask_np_list[3], k=1, axes=(0, 1)),
        np.fliplr(mask_np_list[4]),
        np.fliplr(np.rot90(mask_np_list[5], k=3, axes=(0, 1))),
        np.fliplr(np.rot90(mask_np_list[6], k=2, axes=(0, 1))),
        np.fliplr(np.rot90(mask_np_list[7], k=1, axes=(0, 1))),
    ]
    return reversed_masks


# ---------------------------
# Prediction Logic
# ---------------------------


def predict_probabilities(model, img_np, device):
    """
    Apply TTA and return averaged PROBABILITIES (0.0 to 1.0).
    Does NOT apply thresholding yet.
    """
    tta_imgs = tta_augmentations(img_np)
    tta_preds = []

    with torch.no_grad():
        for aug_img in tta_imgs:
            tensor = torch.from_numpy(aug_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            logits = model(tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W)
            tta_preds.append(probs)

    # Undo all flips/rotations
    restored = undo_tta(tta_preds)

    # Soft Average of TTA steps (always better than hard voting for TTA)
    final_probs = np.mean(restored, axis=0)
    return final_probs


# ---------------------------
# Main Inference Loop
# ---------------------------


def main():
    """
    Main function to run the ensemble inference.

    This function orchestrates the entire inference process, including:
    1. Collecting test images and models.
    2. Iterating over each model and accumulating the predictions.
    3. Applying Test-Time Augmentation (TTA) for each prediction.
    4. Averaging the predictions from all models and TTA steps.
    5. Thresholding the averaged probabilities to create binary masks.
    6. Saving the final masks as PNG images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 1. Collect test images
    test_image_paths = []
    for root, _, files in os.walk(TEST_IMAGES_DIR):
        for fname in files:
            if fname.endswith(".png"):
                test_image_paths.append(os.path.join(root, fname))

    # Sort numerically (test_1, test_2, ...)
    test_image_paths.sort(key=lambda p: int(re.search(r"\d+", p).group(0)))
    print(f"Found {len(test_image_paths)} test images.")

    # 2. Collect Models
    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models directory '{MODELS_DIR}' not found.")

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
    model_files.sort()

    if not model_files:
        raise FileNotFoundError(f"No model files found in {MODELS_DIR}")

    print(f"Found {len(model_files)} models: {model_files}\n")
    print(f"Ensemble Mode: {'MAJORITY VOTE' if USE_MAJORITY_VOTE else 'AVERAGE PROBABILITIES'}")

    # 3. Initialize Accumulators
    # Dictionary to store running totals for each image: {image_path: np_array}
    ensemble_accumulators = {}

    # 4. Iterate over Models (Outer Loop) to save GPU Memory
    if os.path.exists("cache/ensemble_accumulators.pkl"):
        with open("cache/ensemble_accumulators.pkl", "rb") as f:
            ensemble_accumulators = pickle.load(f)
        print("Resumed from pickled ensemble state.")
    else:
        for i, model_file in enumerate(model_files):
            model_path = os.path.join(MODELS_DIR, model_file)
            print(f"--- Processing Model {i + 1}/{len(model_files)}: {model_file} ---")

            # Load ONE model
            model = load_model_from_path(model_path, device)

            # Iterate over all images with this model
            for img_path in test_image_paths:
                # We only load the image from disk when needed.
                img_np = load_image(img_path)

                # Get probability map from this model (0.0 - 1.0)
                probs = predict_probabilities(model, img_np, device)

                # Initialize accumulator if first model
                if img_path not in ensemble_accumulators:
                    ensemble_accumulators[img_path] = np.zeros_like(probs, dtype=np.float32)

                if USE_MAJORITY_VOTE:
                    # Add 1 vote if prob > threshold, else 0
                    vote = (probs > PIXEL_THRESHOLD).astype(np.float32)
                    ensemble_accumulators[img_path] += vote
                else:
                    # Add raw probability
                    ensemble_accumulators[img_path] += probs

            # CLEANUP: Unload model from GPU immediately
            del model
            torch.cuda.empty_cache()
            gc.collect()

            with open("cache/ensemble_accumulators.pkl", "wb") as f:
                pickle.dump(ensemble_accumulators, f)

    print("\n--- Finalizing Ensemble ---")
    num_models = len(model_files)

    for img_path in test_image_paths:
        image_id = extract_image_id(img_path)
        total_value = ensemble_accumulators[img_path]

        final_mask = None

        if USE_MAJORITY_VOTE:
            # Majority Vote: Did more than half the models vote 1?
            # total_value holds the count of '1' votes
            final_mask = (total_value > (num_models / 2)).astype(np.float32)
        else:
            # Averaging: total_value holds sum of probabilities
            avg_prob = total_value / num_models
            final_mask = (avg_prob > PIXEL_THRESHOLD).astype(np.float32)

        out_path = os.path.join(PRED_DIR, f"{image_id}.png")
        save_mask(final_mask, out_path)
        print(f"{out_path}", end=" ")

    print("\nInference complete.")
    print(f"Predictions saved in: {PRED_DIR}")


if __name__ == "__main__":
    main()
