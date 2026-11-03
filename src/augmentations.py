import random

import numpy as np
from PIL import Image

# ---------------------------
# Basic geometric transforms
# ---------------------------


def random_flip(image, mask, p_horizontal, p_vertical):
    """
    Random horizontal and/or vertical flip.
    image: (H, W, 3), mask: (H, W)
    """
    # Horizontal flip
    if random.random() < p_horizontal:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)

    # Vertical flip
    if random.random() < p_vertical:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)

    # Make arrays contiguous again
    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)

    return image, mask


def random_rotate_90(image, mask, p=0.7):
    """
    Random rotation by 0, 90, 180, or 270 degrees with probability p.
    image: (H, W, 3), mask: (H, W)
    """
    if random.random() >= p:
        return image, mask

    k = random.randint(0, 3)  # number of 90° rotations
    if k > 0:
        image = np.rot90(image, k, axes=(0, 1))
        mask = np.rot90(mask, k, axes=(0, 1))

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

    return image, mask


def small_angle_rotate(image, mask, max_angle=10, p=0.3):
    """
    Random small-angle rotation in range [-max_angle, max_angle] degrees.
    Keeps the same image size.
    image: (H, W, 3), values in [0,1]
    mask:  (H, W), values in {0,1}
    """
    if random.random() >= p:
        return image, mask

    angle = random.uniform(-max_angle, max_angle)

    # Convert to PIL images
    img_uint8 = (image * 255).astype(np.uint8)
    msk_uint8 = (mask * 255).astype(np.uint8)

    img_pil = Image.fromarray(img_uint8)
    msk_pil = Image.fromarray(msk_uint8)

    # Rotate; expand=False keeps original size
    img_rot = img_pil.rotate(angle, resample=Image.BILINEAR, expand=False)
    msk_rot = msk_pil.rotate(angle, resample=Image.NEAREST, expand=False)

    # Back to numpy
    img_np = np.asarray(img_rot, dtype=np.float32) / 255.0
    msk_np = np.asarray(msk_rot, dtype=np.float32)
    msk_np = (msk_np > 127).astype(np.float32)  # ensure binary 0/1

    return img_np, msk_np


# ---------------------------
# Photometric transforms
# ---------------------------


def random_brightness_contrast(image, brightness_limit, contrast_limit, p):
    """
    Random brightness and contrast adjustment on the image only.

    image: (H, W, 3), values in [0, 1]
    brightness_limit: max absolute shift
    contrast_limit: max relative scale
    p: probability of applying the transform
    """
    if random.random() >= p:
        return image

    # Random brightness: add a small value in [-brightness_limit, +brightness_limit]
    brightness_shift = random.uniform(-brightness_limit, brightness_limit)

    # Random contrast: multiply by factor in [1-contrast_limit, 1+contrast_limit]
    contrast_factor = 1.0 + random.uniform(-contrast_limit, contrast_limit)

    image = image * contrast_factor + brightness_shift

    # Clip back to [0,1]
    image = np.clip(image, 0.0, 1.0)
    return image


def random_gaussian_noise(image, mean, std, p):
    """
    Add random Gaussian noise to the image.

    image: (H, W, 3), values in [0, 1]
    mean: Mean of Gaussian noise
    std: Standard deviation of noise
    p: Probability of applying the transform
    """
    if random.random() >= p:
        return image

    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0.0, 1.0)
    return noisy_image


# ---------------------------
# High-level transforms
# ---------------------------


def get_train_transform():
    """
    Returns a callable transform(image, mask) for training.

    This will be passed to RoadSegmentationDataset(transform=...)
    and will be applied ON THE FLY for each sample.
    """

    def transform(image, mask):
        # Geometric augmentations: applied to both image and mask
        image_aug, mask_aug = random_flip(image, mask, p_horizontal=0.7, p_vertical=0.7)

        image_aug, mask_aug = random_rotate_90(image_aug, mask_aug, p=0.7)

        # Small-angle jitter for extra robustness
        image_aug, mask_aug = small_angle_rotate(image_aug, mask_aug, max_angle=10, p=0.3)
        image_aug, mask_aug = small_angle_rotate(image_aug, mask_aug, max_angle=35, p=0.1)

        # Photometric augmentations: applied to image only
        image_aug = random_brightness_contrast(image_aug, brightness_limit=0.15, contrast_limit=0.15, p=0.5)

        image_aug = random_gaussian_noise(image_aug, mean=0.0, std=0.03, p=0.3)

        return image_aug, mask_aug

    return transform


def get_val_transform():
    """
    Returns a callable transform(image, mask) for validation.

    No augmentation here: just return the inputs as-is.
    """

    def transform(image, mask):
        return image, mask

    return transform


def val_transform(image, mask):
    return image, mask


def get_val_transform_mp_safe():
    return val_transform
