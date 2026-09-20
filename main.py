"""
Main CLI Entry Point for 2D Autonomous Robot Navigation Simulator.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from robot_nav.config import SimConfig, DEFAULT_CONFIG
from robot_nav.perception.map_from_image import load_map_from_image
from robot_nav.perception.cell_classifier import load_classifier_model, predict_cost_grid
from robot_nav.perception.train import train_terrain_classifier
from robot_nav.sim.world import SimulationWorld
from robot_nav.sim.runner import SimulationRunner
from robot_nav.eval.benchmark import run_benchmark, generate_random_solvable_map
from maps.generator import generate_sample_map_image


def parse_size(size_str: str) -> tuple[int, int]:
    """Parse '60x60' string into (60, 60) tuple."""
    try:
        parts = size_str.lower().split("x")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid size format '{size_str}'. Expected format like '60x60'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2D Autonomous Robot Simulator - Vision, A* Path Planning, Pure Pursuit & Dynamic Replanning"
    )
    parser.add_argument("--image", type=str, help="Path to input map image (e.g. maps/sample1.png)")
    parser.add_argument("--random", action="store_true", help="Generate a random procedural occupancy map")
    parser.add_argument("--size", type=parse_size, default="60x60", help="Grid size in format HxW (default: 60x60)")
    parser.add_argument(
        "--planner",
        type=str,
        choices=["A*", "BFS", "Dijkstra", "Greedy"],
        default="A*",
        help="Path planning algorithm (default: A*)",
    )
    parser.add_argument("--train", action="store_true", help="Train PyTorch CNN cell classifier (<3 min CPU)")
    parser.add_argument("--benchmark", action="store_true", help="Run 50-map benchmark evaluation across 3 densities")
    parser.add_argument("--headless", action="store_true", help="Run simulation in non-GUI headless mode")

    args = parser.parse_args()

    # Mode 1: Train CNN
    if args.train:
        print("Executing PyTorch CNN cell classifier training...")
        train_terrain_classifier()
        sys.exit(0)

    # Mode 2: Run Benchmark
    if args.benchmark:
        print("Executing multi-planner benchmark evaluation...")
        run_benchmark(num_maps=50, densities=[0.10, 0.20, 0.30])
        sys.exit(0)

    # Mode 3: Simulation (Image-based or Procedural Random)
    config = SimConfig()
    grid_h, grid_w = args.size
    config.map.grid_height = grid_h
    config.map.grid_width = grid_w

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Specified image '{img_path}' not found. Generating sample map image...")
            img_path = generate_sample_map_image(img_path)

        print(f"Loading map from image: {img_path}")
        perceived_grid_arr, start, goal = load_map_from_image(img_path, config=config)

        # Ground truth initially matches perceived grid; add subtle dynamic hidden obstacle for test
        gt_grid_arr = perceived_grid_arr.copy()
        # Add dynamic hidden obstacle in ground truth map near center if free
        mid_r, mid_c = grid_h // 2, grid_w // 2
        if gt_grid_arr[mid_r, mid_c] == 0:
            gt_grid_arr[mid_r, mid_c : mid_c + 3] = 1

    else:  # Default or --random
        print(f"Generating procedural random solvable map ({grid_h}x{grid_w})...")
        gt_grid_arr, start, goal = generate_random_solvable_map(grid_size=(grid_h, grid_w), density=0.20)

        # Perceived map initially omits some hidden obstacles
        perceived_grid_arr = gt_grid_arr.copy()
        # Mask out 2 random obstacles from perceived grid so sensor reveals them dynamically
        obs_coords = np.argwhere(gt_grid_arr == 1)
        if len(obs_coords) > 5:
            hidden_indices = obs_coords[len(obs_coords) // 2 : len(obs_coords) // 2 + 3]
            for hr, hc in hidden_indices:
                if (hr, hc) != start and (hr, hc) != goal:
                    perceived_grid_arr[hr, hc] = 0

    print(f"Map parsed successfully. Grid shape: {perceived_grid_arr.shape} | Start: {start} | Goal: {goal}")
    print(f"Initializing simulation with planner: '{args.planner}'...")

    world = SimulationWorld(
        ground_truth_array=gt_grid_arr,
        perceived_array=perceived_grid_arr,
        start_cell=start,
        goal_cell=goal,
        config=config,
        planner_name=args.planner,
    )

    if args.headless:
        print("Running in headless mode...")
        SimulationRunner.run_headless(world)
        print("\nHeadless simulation completed!")
        print(f"  Reached goal: {world.is_finished}")
        print(f"  Sim time: {world.sim_time:.2f}s")
        print(f"  Traveled distance: {world.total_distance_traveled:.2f}m")
        print(f"  Replans: {len(world.replanner.history) - 1}")
        print(f"  Mean CTE: {np.mean(world.cross_track_errors):.4f}m")
    else:
        print("Starting interactive Pygame GUI window...")
        SimulationRunner.run_interactive(world, config=config)


if __name__ == "__main__":
    main()
