"""
Tests for evaluation module: trajectory metrics and benchmark runner.
"""

from pathlib import Path
import numpy as np
import pytest

from robot_nav.sim.world import SimulationWorld
from robot_nav.eval.metrics import compute_trajectory_metrics
from robot_nav.eval.benchmark import run_benchmark, generate_random_solvable_map


def test_trajectory_metrics_computation():
    gt = np.zeros((15, 15), dtype=np.uint8)
    perc = np.zeros((15, 15), dtype=np.uint8)
    world = SimulationWorld(gt, perc, start_cell=(1, 1), goal_cell=(12, 12), planner_name="A*")


    world.step(dt=0.1)

    metrics = compute_trajectory_metrics(world)

    assert metrics["planner"] == "A*"
    assert "path_cost" in metrics
    assert "expanded_nodes" in metrics
    assert "planning_runtime_ms" in metrics
    assert "mean_cte_m" in metrics
    assert "max_cte_m" in metrics


def test_procedural_solvable_map_generator():
    grid, start, goal = generate_random_solvable_map(grid_size=(30, 30), density=0.20, seed=123)
    assert grid.shape == (30, 30)
    assert grid[start] == 0
    assert grid[goal] == 0


def test_short_benchmark_run(tmp_path: Path):
    results = run_benchmark(
        num_maps=2,
        densities=[0.10, 0.20],
        output_dir=tmp_path,
        grid_size=(20, 20),
    )

    # 2 maps * 2 densities * 4 planners = 16 result records
    assert len(results) == 16
    assert (tmp_path / "benchmark_results.csv").exists()
    assert (tmp_path / "path_cost.png").exists()
    assert (tmp_path / "nodes_expanded.png").exists()
    assert (tmp_path / "runtime.png").exists()
