"""
Tests for simulation module: Lidar sensor raycasting, world stepping, dynamic replanning, and runner.
"""

import math
import numpy as np
import pytest

from robot_nav.sim.sensor import LidarSensor
from robot_nav.sim.world import SimulationWorld
from robot_nav.sim.runner import SimulationRunner
from robot_nav.config import SimConfig


def test_lidar_sensor_raycast():
    grid_arr = np.zeros((20, 20), dtype=np.uint8)
    # Place wall at col 10
    grid_arr[:, 10] = 1

    sensor = LidarSensor(range_m=5.0, fov_deg=90.0, num_rays=21)

    # Robot at (1.0, 2.0) facing theta=0 (pointing straight towards col 10)
    # Cell size 0.2m -> col 10 is at x = 2.0m. Distance = 1.0m.
    endpoints, free_cells, obs_cells = sensor.scan(grid_arr, (1.0, 2.0, 0.0), cell_size_m=0.2)

    assert len(endpoints) == 21
    assert len(obs_cells) > 0
    # Wall cells at col 10 should be in obs_cells
    cols = [c for r, c in obs_cells]
    assert 10 in cols


def test_simulation_world_headless_execution():
    config = SimConfig()
    config.map.grid_height = 20
    config.map.grid_width = 20

    gt_grid = np.zeros((20, 20), dtype=np.uint8)
    perc_grid = np.zeros((20, 20), dtype=np.uint8)

    # Add dynamic hidden obstacle in ground truth map along path
    gt_grid[5, 5] = 1

    start = (1, 1)
    goal = (10, 10)

    world = SimulationWorld(gt_grid, perc_grid, start, goal, config=config, planner_name="A*")
    assert world.is_finished is False
    assert len(world.current_path) > 0

    SimulationRunner.run_headless(world, max_steps=400, dt=0.05)

    assert world.is_finished is True
    assert world.total_distance_traveled > 0.0
    assert len(world.traveled_trail) > 10
