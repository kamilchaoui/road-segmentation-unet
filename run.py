"""
This script is the main entry point for the road segmentation inference pipeline.
"""

import gc
import os
import pickle
import re
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.dataset import load_image
from src.ensemble_inference import (
    extract_image_id,
    load_model_from_path,
    predict_probabilities,
    save_mask,
)

TEST_IMAGES_DIR = "test_set_images"
PRED_DIR = "predictions"
MODELS_DIR = "./models"
PIXEL_THRESHOLD = 0.73

# ------------------------------------------
# 0. Setup
# ------------------------------------------

os.makedirs(PRED_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ------------------------------------------
# 1. Collect test images
# ------------------------------------------

test_image_paths = []
for root, _, files in os.walk(TEST_IMAGES_DIR):
    for fname in files:
        if fname.endswith(".png"):
            test_image_paths.append(os.path.join(root, fname))

test_image_paths.sort(key=lambda p: int(re.search(r"\d+", p).group(0)))  # type: ignore
print(f"Found {len(test_image_paths)} test images.")
if len(test_image_paths) == 0:
    raise FileNotFoundError(f"No test images found in {TEST_IMAGES_DIR}")

# ------------------------------------------
# 2. Collect models
# ------------------------------------------

model_files = []
if os.path.exists(MODELS_DIR):
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]
    model_files.sort()
    print(f"Found {len(model_files)} models: {model_files}\n")
else:
    print(f"Models directory '{MODELS_DIR}' not found.")
    print("Will proceed with cached results if available.")

# ------------------------------------------
# 3. Gather predictions from each model and accumulate
# ------------------------------------------

ensemble_accumulators = {}

if os.path.exists("cache/ensemble_accumulators.pkl"):
    # load cached accumulator state
    with open("cache/ensemble_accumulators.pkl", "rb") as f:
        ensemble_accumulators = pickle.load(f)
    print("Loaded cached accumulator state.")

elif os.path.exists(MODELS_DIR):
    for i, model_file in enumerate(model_files):
        model_path = os.path.join(MODELS_DIR, model_file)
        print(f"Processing Model {i + 1}/{len(model_files)}: {model_file}")
        model = load_model_from_path(model_path, device)

        # Iterate over all images with this model
        for img_path in test_image_paths:
            img_np = load_image(img_path)
            probs = predict_probabilities(model, img_np, device)
            if img_path not in ensemble_accumulators:
                ensemble_accumulators[img_path] = np.zeros_like(probs, dtype=np.float32)
            ensemble_accumulators[img_path] += probs

        # Free up (GPU) memory
        del model
        torch.cuda.empty_cache()
        gc.collect()

    with open("cache/ensemble_accumulators.pkl", "wb") as f:
        pickle.dump(ensemble_accumulators, f)
        print("Saved ensemble accumulator state to cache.")


# ------------------------------------------
# 4. Build final masks and save predictions
# ------------------------------------------

num_models = 66

for img_path in test_image_paths:
    image_id = extract_image_id(img_path)
    total_value = ensemble_accumulators[img_path]

    # Build mask based on average probabilities
    avg_prob = total_value / num_models
    final_mask = (avg_prob > PIXEL_THRESHOLD).astype(np.float32)

    out_path = os.path.join(PRED_DIR, f"{image_id}.png")
    save_mask(final_mask, out_path)
    print(f"{out_path}", end=" ")

print(f"Predictions saved in: {PRED_DIR}")
