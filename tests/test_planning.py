"""
Tests for planning algorithms (A*, Dijkstra, BFS, Greedy), grid corner-cutting guards, and replanning.
"""

import random
import numpy as np
import pytest

from robot_nav.planning.grid import OccupancyGrid
from robot_nav.planning.astar import astar_search
from robot_nav.planning.baselines import bfs_search, dijkstra_search, greedy_best_first_search
from robot_nav.planning.replan import Replanner


def test_start_equals_goal():
    grid_arr = np.zeros((10, 10), dtype=np.uint8)
    grid = OccupancyGrid(grid_arr)
    start = (3, 3)
    goal = (3, 3)

    res = astar_search(grid, start, goal)
    assert res.cost == 0.0
    assert res.path == [start]
    assert res.smooth_path == [start]


def test_no_path_exists():
    grid_arr = np.zeros((10, 10), dtype=np.uint8)
    # Wall dividing grid vertically
    grid_arr[:, 5] = 1
    grid = OccupancyGrid(grid_arr)

    start = (2, 2)
    goal = (2, 8)

    res = astar_search(grid, start, goal)
    assert res.cost == float("inf")
    assert res.path == []
    assert res.smooth_path == []


def test_diagonal_corner_cutting_forbidden():
    """
    Test that diagonal step is forbidden when passing between two diagonal obstacles:
    0 1
    1 0
    Moving from (0,0) to (1,1) should NOT be allowed directly.
    """
    grid_arr = np.zeros((3, 3), dtype=np.uint8)
    grid_arr[0, 1] = 1
    grid_arr[1, 0] = 1
    grid = OccupancyGrid(grid_arr)

    neighbors = grid.get_neighbors((0, 0), connectivity=8)
    neighbor_coords = [n[0] for n in neighbors]

    # (1, 1) should NOT be in neighbor_coords because (0,1) and (1,0) are obstacles!
    assert (1, 1) not in neighbor_coords


def test_astar_optimality_vs_dijkstra_20_grids():
    """
    Verify A* produces exact same path cost as Dijkstra across 20 random grids.
    """
    random.seed(42)
    np.random.seed(42)

    for i in range(20):
        size = 20
        # 20% random obstacle density
        grid_arr = (np.random.rand(size, size) < 0.2).astype(np.uint8)
        # Ensure rough terrain occasionally
        rough_mask = (np.random.rand(size, size) < 0.15) & (grid_arr == 0)
        grid_arr[rough_mask] = 2

        start = (0, 0)
        goal = (size - 1, size - 1)

        grid_arr[start] = 0
        grid_arr[goal] = 0

        grid = OccupancyGrid(grid_arr)

        res_astar = astar_search(grid, start, goal, connectivity=8, heuristic_type="octile", smooth=False)
        res_dijkstra = dijkstra_search(grid, start, goal, connectivity=8, smooth=False)

        assert pytest.approx(res_astar.cost, rel=1e-5) == res_dijkstra.cost, (
            f"Cost mismatch on map {i}: A*={res_astar.cost}, Dijkstra={res_dijkstra.cost}"
        )


def test_path_validity_traversable():
    """
    Check that all steps in an A* path are traversable and adjacent.
    """
    grid_arr = np.zeros((15, 15), dtype=np.uint8)
    grid_arr[5, 2:13] = 1  # Wall with gaps
    grid = OccupancyGrid(grid_arr)

    start = (1, 1)
    goal = (13, 13)

    res = astar_search(grid, start, goal, connectivity=8)
    assert len(res.path) > 0

    for idx in range(len(res.path) - 1):
        r1, c1 = res.path[idx]
        r2, c2 = res.path[idx + 1]

        assert grid.is_free(r1, c1)
        assert grid.is_free(r2, c2)
        dr, dc = abs(r2 - r1), abs(c2 - c1)
        assert dr <= 1 and dc <= 1 and (dr + dc > 0)


def test_replanning_trigger():
    """
    Verify replanner detects dynamic obstacle on path and recomputes new path.
    """
    grid_arr = np.zeros((10, 10), dtype=np.uint8)
    grid = OccupancyGrid(grid_arr)
    start = (0, 0)
    goal = (0, 9)

    planner = lambda g, s, e: astar_search(g, s, e, smooth=False)
    res = planner(grid, start, goal)
    orig_path = res.path

    replanner = Replanner()

    # Block path by placing obstacle at (0, 4)
    grid.grid[0, 4] = 1
    grid.cost_grid[0, 4] = float("inf")

    did_replan, new_path, event = replanner.check_and_replan(
        grid, start, goal, orig_path, current_path_idx=0, planner_fn=planner
    )

    assert did_replan is True
    assert event is not None
    assert event.reason == "obstacle_detected_on_path"
    assert (0, 4) not in new_path
