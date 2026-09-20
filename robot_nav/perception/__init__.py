"""
Perception subpackage: CV map conversion, CNN patch classifier, synthetic dataset generator, and training pipeline.
"""

from robot_nav.perception.map_from_image import load_map_from_image
from robot_nav.perception.cell_classifier import TerrainCNN, predict_cost_grid
from robot_nav.perception.dataset_gen import generate_synthetic_patch_dataset
from robot_nav.perception.train import train_terrain_classifier

__all__ = [
    "load_map_from_image",
    "TerrainCNN",
    "predict_cost_grid",
    "generate_synthetic_patch_dataset",
    "train_terrain_classifier",
]
