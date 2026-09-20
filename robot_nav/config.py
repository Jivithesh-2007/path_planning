"""
Central configuration module for the robot_nav simulator.
Defines all hyper-parameters, thresholds, and dimensions as clean dataclass structures.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List

@dataclass
class MapConfig:
    grid_width: int = 60
    grid_height: int = 60
    cell_size_m: float = 0.2  # Each cell is 0.2m x 0.2m
    occupied_threshold: float = 0.35  # Fraction of occupied pixels in patch
    free_cost: float = 1.0
    rough_terrain_cost: float = 3.0
    obstacle_cost: float = 1e6  # Infinite / untraversable cost

@dataclass
class PerceptionConfig:
    patch_size: int = 32  # 32x32 pixel patch for cell classifier
    weights_path: Path = field(default_factory=lambda: Path("robot_nav/perception/weights/cnn.pth"))
    # HSV ranges for start (green) and goal (red) detection
    # OpenCV HSV ranges: H: 0-180, S: 0-255, V: 0-255
    green_lower: Tuple[int, int, int] = (35, 50, 50)
    green_upper: Tuple[int, int, int] = (85, 255, 255)
    red_lower1: Tuple[int, int, int] = (0, 70, 50)
    red_upper1: Tuple[int, int, int] = (10, 255, 255)
    red_lower2: Tuple[int, int, int] = (170, 70, 50)
    red_upper2: Tuple[int, int, int] = (180, 255, 255)

@dataclass
class PlannerConfig:
    connectivity: int = 8  # 4- or 8-connectivity
    heuristic_type: str = "octile"  # "octile", "euclidean", or "manhattan"
    tie_break_eps: float = 1.001  # Small factor to break tie in priority queue
    smooth_path: bool = True  # Bresenham line-of-sight shortcutting

@dataclass
class ControlConfig:
    robot_radius: float = 0.25  # Meters
    v_max: float = 1.5  # Max linear velocity m/s
    w_max: float = 2.5  # Max angular velocity rad/s
    a_max: float = 2.0  # Max linear acceleration m/s^2
    alpha_max: float = 4.0  # Max angular acceleration rad/s^2
    lookahead_dist: float = 0.6  # Pure pursuit lookahead distance in meters
    dt: float = 0.05  # Simulation time step in seconds

@dataclass
class SensorConfig:
    range_m: float = 6.0  # Lidar ray max range in meters
    fov_deg: float = 120.0  # Field of view angle in degrees
    num_rays: int = 61  # Number of rays cast across FOV

@dataclass
class RenderConfig:
    screen_width: int = 800
    screen_height: int = 800
    fps: int = 60
    title: str = "2D Autonomous Robot Simulator - Vision, A*, Pure Pursuit & Replanning"

@dataclass
class EvalConfig:
    benchmark_maps: int = 50
    densities: List[float] = field(default_factory=lambda: [0.10, 0.20, 0.30])
    output_dir: Path = field(default_factory=lambda: Path("eval/results"))

@dataclass
class SimConfig:
    map: MapConfig = field(default_factory=MapConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

# Default global config instance
DEFAULT_CONFIG = SimConfig()
