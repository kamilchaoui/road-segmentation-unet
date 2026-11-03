import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGES_DIR = "training/images"
MASKS_DIR = "training/groundtruth"
FILE_EXT = ".png"


def id_to_filename(image_id):
    """Convert '001' to 'satImage_001.png'."""
    return f"satImage_{image_id}{FILE_EXT}"


def load_image(path):
    """Load RGB image as float32 numpy array in [0,1], shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    return img_np


def load_mask(path):
    """Load mask as float32 numpy array in {0,1}, shape (H, W)."""
    m = Image.open(path).convert("L")  # convert to grayscale
    m_np = np.asarray(m, dtype=np.float32)
    m_np = (m_np > 0).astype(np.float32)  # ensure binary mask
    return m_np


def list_image_ids(images_dir=IMAGES_DIR):
    """Return ['001', '002', ...] based on files in training/images."""
    ids = []
    for fname in os.listdir(images_dir):
        if fname.startswith("satImage_") and fname.endswith(FILE_EXT):
            image_id = fname.replace("satImage_", "").replace(FILE_EXT, "")
            ids.append(image_id)
    ids.sort()
    return ids


class RoadSegmentationDataset(Dataset):
    """A PyTorch Dataset for loading road segmentation data.

    Returns:
        image: FloatTensor (3, H, W)
        mask:  FloatTensor (1, H, W)
    """

    def __init__(self, image_ids, images_dir=IMAGES_DIR, masks_dir=MASKS_DIR, transform=None, return_id=False):
        self.image_ids = image_ids
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.return_id = return_id

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        img_path = os.path.join(self.images_dir, id_to_filename(image_id))
        mask_path = os.path.join(self.masks_dir, id_to_filename(image_id))

        image = load_image(img_path)  # numpy (H, W, 3)
        mask = load_mask(mask_path)  # numpy (H, W)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Convert to tensors (CHW for image, 1HW for mask)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        if self.return_id:
            return image_tensor, mask_tensor, image_id

        return image_tensor, mask_tensor

    def __repr__(self):
        return f"RoadSegmentationDataset(num_samples={len(self)}, images_dir={self.images_dir}, masks_dir={self.masks_dir})"
