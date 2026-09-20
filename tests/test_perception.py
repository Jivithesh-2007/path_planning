"""
Tests for perception module: CV map parsing, CNN classifier, dataset generator, and fallback.
"""

from pathlib import Path
import numpy as np
import cv2
import torch
import pytest

from robot_nav.config import SimConfig
from robot_nav.perception.map_from_image import load_map_from_image, detect_start_and_goal
from robot_nav.perception.cell_classifier import TerrainCNN, predict_patch_cv_fallback, classify_map_grid
from robot_nav.perception.dataset_gen import generate_synthetic_patch_dataset
from robot_nav.perception.train import train_terrain_classifier


def test_cnn_parameter_count_constraint():
    model = TerrainCNN(num_classes=3)
    param_count = model.count_parameters()
    assert param_count < 100_000, f"Parameter count {param_count} exceeds 100k limit!"


def test_dataset_generator_shape_and_labels():
    X, y = generate_synthetic_patch_dataset(samples_per_class=10, patch_size=32)
    assert X.shape == (30, 1, 32, 32)
    assert y.shape == (30,)
    assert set(np.unique(y)) == {0, 1, 2}
    assert X.dtype == np.float32
    assert 0.0 <= X.min() <= X.max() <= 1.0


def test_cv_fallback_classifier():
    # Clean white patch -> Free space (0)
    white_patch = np.full((32, 32), 240, dtype=np.uint8)
    assert predict_patch_cv_fallback(white_patch) == 0

    # Solid dark patch -> Obstacle (1)
    dark_patch = np.full((32, 32), 20, dtype=np.uint8)
    assert predict_patch_cv_fallback(dark_patch) == 1

    # Textured patch -> Rough terrain (2)
    textured_patch = np.full((32, 32), 150, dtype=np.uint8)
    noise = np.random.normal(0, 40, (32, 32)).astype(np.int16)
    textured_patch = np.clip(textured_patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    assert predict_patch_cv_fallback(textured_patch) == 2


def test_map_from_image_synthetic_pipeline(tmp_path: Path):
    # Create a synthetic 300x300 BGR map image
    map_img = np.full((300, 300, 3), 255, dtype=np.uint8)

    # Draw black wall (obstacle)
    cv2.rectangle(map_img, (50, 50), (250, 80), (0, 0, 0), -1)

    # Draw green circle for start (BGR: 0, 255, 0)
    cv2.circle(map_img, (30, 30), 12, (0, 255, 0), -1)

    # Draw red circle for goal (BGR: 0, 0, 255)
    cv2.circle(map_img, (270, 270), 12, (0, 0, 255), -1)

    img_file = tmp_path / "synthetic_map.png"
    cv2.imwrite(str(img_file), map_img)

    config = SimConfig()
    config.map.grid_height = 60
    config.map.grid_width = 60

    debug_out = tmp_path / "debug_overlay.png"
    grid, start, goal = load_map_from_image(img_file, config, save_debug_path=debug_out)

    assert grid.shape == (60, 60)
    assert isinstance(start, tuple) and len(start) == 2
    assert isinstance(goal, tuple) and len(goal) == 2
    # Start and goal should be free in grid
    assert grid[start[0], start[1]] == 0
    assert grid[goal[0], goal[1]] == 0

    # Verify debug overlay image was written
    assert debug_out.exists()


def test_train_classifier_short_run(tmp_path: Path):
    weights_file = tmp_path / "test_cnn.pth"
    model = train_terrain_classifier(save_path=weights_file, epochs=1, batch_size=64)
    assert weights_file.exists()
    assert isinstance(model, TerrainCNN)
