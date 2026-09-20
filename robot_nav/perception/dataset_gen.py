"""
Procedural synthetic patch dataset generator for 3-class cell classification.
Classes: 0 = Free, 1 = Obstacle, 2 = Rough Terrain.
"""

import numpy as np
import cv2
from typing import Tuple


def generate_free_patch(patch_size: int = 32) -> np.ndarray:
    """Generate a free space patch (clean light background with subtle grain)."""
    # Base bright gray background (210 to 255)
    base_val = np.random.randint(215, 255)
    patch = np.full((patch_size, patch_size), base_val, dtype=np.uint8)

    # Add minor Gaussian noise
    noise = np.random.normal(0, 3, (patch_size, patch_size)).astype(np.int16)
    patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return patch


def generate_obstacle_patch(patch_size: int = 32) -> np.ndarray:
    """Generate an obstacle patch (dark solid block, thick black line, or corner)."""
    patch = np.full((patch_size, patch_size), np.random.randint(220, 255), dtype=np.uint8)

    obstacle_type = np.random.choice(["solid", "thick_line", "filled_circle", "block"])

    if obstacle_type == "solid":
        patch[:, :] = np.random.randint(0, 40)
    elif obstacle_type == "thick_line":
        pt1 = (np.random.randint(0, patch_size), np.random.randint(0, patch_size))
        pt2 = (np.random.randint(0, patch_size), np.random.randint(0, patch_size))
        thickness = np.random.randint(8, 16)
        cv2.line(patch, pt1, pt2, color=np.random.randint(0, 50), thickness=thickness)
    elif obstacle_type == "filled_circle":
        center = (patch_size // 2, patch_size // 2)
        radius = np.random.randint(8, 15)
        cv2.circle(patch, center, radius, color=np.random.randint(0, 50), thickness=-1)
    else:  # block
        r1, c1 = np.random.randint(0, 8), np.random.randint(0, 8)
        r2, c2 = np.random.randint(20, patch_size), np.random.randint(20, patch_size)
        patch[r1:r2, c1:c2] = np.random.randint(0, 50)

    # Minor noise
    noise = np.random.normal(0, 5, (patch_size, patch_size)).astype(np.int16)
    patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return patch


def generate_rough_patch(patch_size: int = 32) -> np.ndarray:
    """Generate a rough terrain patch (medium gray, speckled texture, hatch lines)."""
    # Mid-gray base
    base_val = np.random.randint(120, 170)
    patch = np.full((patch_size, patch_size), base_val, dtype=np.uint8)

    rough_type = np.random.choice(["speckled", "hatch", "dots"])

    if rough_type == "speckled":
        # High variance salt & pepper / speckle noise
        noise = np.random.normal(0, 35, (patch_size, patch_size)).astype(np.int16)
        patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif rough_type == "hatch":
        # Parallel thin hatch lines
        num_lines = np.random.randint(3, 7)
        for _ in range(num_lines):
            y = np.random.randint(0, patch_size)
            cv2.line(patch, (0, y), (patch_size, y), color=np.random.randint(50, 100), thickness=1)
    else:  # dots
        num_dots = np.random.randint(15, 35)
        for _ in range(num_dots):
            pt = (np.random.randint(0, patch_size), np.random.randint(0, patch_size))
            cv2.circle(patch, pt, radius=1, color=np.random.randint(40, 90), thickness=-1)

    return patch


def augment_patch(patch: np.ndarray) -> np.ndarray:
    """Apply random rotation, blur, and brightness adjustments."""
    patch_size = patch.shape[0]

    # Random rotation
    angle = np.random.choice([0, 90, 180, 270])
    if angle == 90:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        patch = cv2.rotate(patch, cv2.ROTATE_180)
    elif angle == 270:
        patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Optional blur
    if np.random.rand() < 0.3:
        patch = cv2.GaussianBlur(patch, (3, 3), 0)

    return patch


def generate_synthetic_patch_dataset(
    samples_per_class: int = 400, patch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset for free (0), obstacle (1), and rough terrain (2).

    :param samples_per_class: Number of samples per class.
    :param patch_size: Size of square patch.
    :return: (X, y) where X is float32 shape (N, 1, patch_size, patch_size) normalized [0, 1], y is int64 shape (N,).
    """
    patches = []
    labels = []

    for _ in range(samples_per_class):
        # Class 0: Free
        p0 = augment_patch(generate_free_patch(patch_size))
        patches.append(p0)
        labels.append(0)

        # Class 1: Obstacle
        p1 = augment_patch(generate_obstacle_patch(patch_size))
        patches.append(p1)
        labels.append(1)

        # Class 2: Rough Terrain
        p2 = augment_patch(generate_rough_patch(patch_size))
        patches.append(p2)
        labels.append(2)

    X = np.stack(patches, axis=0)[:, np.newaxis, :, :].astype(np.float32) / 255.0
    y = np.array(labels, dtype=np.int64)

    # Shuffle
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    return X[indices], y[indices]
