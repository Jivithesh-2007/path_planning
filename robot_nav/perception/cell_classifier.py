"""
Small CNN classifier (<100k params) and pure CV fallback for cell terrain classification.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import Tuple, Optional


class TerrainCNN(nn.Module):
    """
    3-layer Convolutional Neural Network (<100k parameters) for 3-class cell patch classification:
    Class 0: Free space (cost 1.0)
    Class 1: Obstacle (cost inf / 1e6)
    Class 2: Rough terrain (cost 3.0)
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Conv1: 1 -> 16, output: (16, 16, 16)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Conv2: 16 -> 32, output: (32, 8, 8)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Conv3: 32 -> 64, output: (64, 4, 4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_classifier_model(weights_path: Path | str) -> Optional[TerrainCNN]:
    """Load model weights if existing, else return None."""
    path = Path(weights_path)
    if not path.exists():
        return None
    try:
        model = TerrainCNN()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model
    except Exception:
        return None


def predict_patch_cv_fallback(patch_gray: np.ndarray) -> int:
    """
    Pure CV heuristic fallback classifier if no neural network weights are available.
    :param patch_gray: uint8 2D grayscale array (32x32)
    :return: class label 0 (free), 1 (obstacle), 2 (rough)
    """
    mean_val = float(np.mean(patch_gray))
    std_val = float(np.std(patch_gray))

    # Obstacle: very dark or high fraction of dark pixels
    dark_ratio = float(np.mean(patch_gray < 100))
    if dark_ratio > 0.35 or mean_val < 80:
        return 1
    # Rough terrain: textured/speckled with high standard deviation
    elif std_val > 25.0:
        return 2
    # Free space: clean light background
    else:
        return 0


def classify_map_grid(
    image_gray: np.ndarray,
    model: Optional[TerrainCNN] = None,
    grid_rows: int = 60,
    grid_cols: int = 60,
    free_cost: float = 1.0,
    rough_cost: float = 3.0,
    obstacle_cost: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify downsampled grid cells from a top-down grayscale map image.

    :param image_gray: 2D uint8 grayscale map image.
    :param model: Optional loaded TerrainCNN model.
    :param grid_rows: Number of grid rows.
    :param grid_cols: Number of grid columns.
    :param free_cost: Traversal cost for free space.
    :param rough_cost: Traversal cost for rough terrain.
    :param obstacle_cost: Traversal cost for obstacle.
    :return: (occupancy_grid: uint8, cost_grid: float32)
    """
    h, w = image_gray.shape
    cell_h = h / grid_rows
    cell_w = w / grid_cols

    occupancy = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
    cost_grid = np.full((grid_rows, grid_cols), free_cost, dtype=np.float32)

    patches_batch = []
    cell_indices = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            r_start, r_end = int(r * cell_h), int((r + 1) * cell_h)
            c_start, c_end = int(c * cell_w), int((c + 1) * cell_w)

            patch = image_gray[r_start:r_end, c_start:c_end]
            if patch.size == 0:
                patch = np.full((32, 32), 255, dtype=np.uint8)
            else:
                patch = cv2.resize(patch, (32, 32))

            if model is not None:
                patch_tensor = patch.astype(np.float32) / 255.0
                patches_batch.append(patch_tensor)
                cell_indices.append((r, c))
            else:
                cls_label = predict_patch_cv_fallback(patch)
                occupancy[r, c] = cls_label
                if cls_label == 1:
                    cost_grid[r, c] = obstacle_cost
                elif cls_label == 2:
                    cost_grid[r, c] = rough_cost
                else:
                    cost_grid[r, c] = free_cost

    if model is not None and patches_batch:
        batch_arr = np.stack(patches_batch, axis=0)[:, np.newaxis, :, :]
        tensor_in = torch.from_numpy(batch_arr)
        with torch.no_grad():
            outputs = model(tensor_in)
            preds = torch.argmax(outputs, dim=1).numpy()

        for idx, (r, c) in enumerate(cell_indices):
            cls_label = int(preds[idx])
            occupancy[r, c] = cls_label
            if cls_label == 1:
                cost_grid[r, c] = obstacle_cost
            elif cls_label == 2:
                cost_grid[r, c] = rough_cost
            else:
                cost_grid[r, c] = free_cost

    return occupancy, cost_grid


# Alias for backward compatibility / spec compliance
predict_cost_grid = classify_map_grid

