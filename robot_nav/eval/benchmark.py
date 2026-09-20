"""
Benchmark suite evaluating BFS, Dijkstra, Greedy Best-First, and A* across procedurally generated maps.
Generates CSV results and Matplotlib comparative charts.
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

from robot_nav.config import SimConfig, DEFAULT_CONFIG
from robot_nav.planning.grid import OccupancyGrid
from robot_nav.planning.astar import astar_search
from robot_nav.planning.baselines import bfs_search, dijkstra_search, greedy_best_first_search


def generate_random_solvable_map(
    grid_size: Tuple[int, int] = (60, 60),
    density: float = 0.20,
    seed: int = 42,
    rough_ratio: float = 0.15,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Procedurally generate a random solvable occupancy grid map with obstacles and rough terrain.

    :param grid_size: (rows, cols) tuple.
    :param density: Obstacle density fraction (0.1, 0.2, 0.3).
    :param seed: Random seed for reproducibility.
    :param rough_ratio: Fraction of free cells to assign as rough terrain.
    :return: (grid_array: uint8, start: (r, c), goal: (r, c))
    """
    rng = np.random.RandomState(seed)
    h, w = grid_size

    for trial in range(100):
        grid = (rng.rand(h, w) < density).astype(np.uint8)

        # Assign rough terrain to random free cells
        free_mask = (grid == 0)
        rough_mask = (rng.rand(h, w) < rough_ratio) & free_mask
        grid[rough_mask] = 2

        start = (1, 1)
        goal = (h - 2, w - 2)

        grid[start] = 0
        grid[goal] = 0

        # Validate that a path exists using A*
        occ = OccupancyGrid(grid)
        res = astar_search(occ, start, goal, connectivity=8, smooth=False)
        if res.path:
            return grid, start, goal

        # Increment seed for next trial if unsolvable
        rng = np.random.RandomState(seed + trial + 1000)

    # Fallback to empty grid if high density blocks all paths
    grid = np.zeros(grid_size, dtype=np.uint8)
    return grid, (1, 1), (h - 2, w - 2)


def run_benchmark(
    num_maps: int = 50,
    densities: List[float] = [0.10, 0.20, 0.30],
    output_dir: Path | str = "eval/results",
    grid_size: Tuple[int, int] = (60, 60),
) -> List[Dict[str, Any]]:
    """
    Run multi-planner evaluation across procedurally generated maps and obstacle densities.

    :param num_maps: Total maps per density level.
    :param densities: Obstacle density levels to evaluate.
    :param output_dir: Directory path for exporting CSV and charts.
    :param grid_size: Dimensions of generated maps.
    :return: List of metric dictionary records.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    planners = ["BFS", "Dijkstra", "Greedy", "A*"]
    results: List[Dict[str, Any]] = []

    print(f"\n================ Running Planner Benchmark ================")
    print(f"Maps per density: {num_maps} | Densities: {densities} | Grid: {grid_size}")

    total_runs = num_maps * len(densities) * len(planners)
    run_idx = 0

    for density in densities:
        for map_id in range(num_maps):
            seed = int(density * 10000) + map_id * 37
            grid_arr, start, goal = generate_random_solvable_map(grid_size, density, seed=seed)
            occ_grid = OccupancyGrid(grid_arr)

            for planner_name in planners:
                run_idx += 1
                if planner_name == "BFS":
                    res = bfs_search(occ_grid, start, goal, connectivity=8, smooth=True)
                elif planner_name == "Dijkstra":
                    res = dijkstra_search(occ_grid, start, goal, connectivity=8, smooth=True)
                elif planner_name == "Greedy":
                    res = greedy_best_first_search(occ_grid, start, goal, connectivity=8, heuristic_type="octile", smooth=True)
                else:  # A*
                    res = astar_search(occ_grid, start, goal, connectivity=8, heuristic_type="octile", tie_break_eps=1.001, smooth=True)

                rec = {
                    "map_id": map_id,
                    "density": density,
                    "planner": planner_name,
                    "cost": round(res.cost, 3),
                    "nodes_expanded": res.expanded_nodes,
                    "runtime_ms": round(res.runtime_ms, 3),
                    "path_length": len(res.path),
                    "smooth_path_length": len(res.smooth_path),
                    "success": bool(res.path),
                }
                results.append(rec)

    # 1. Export CSV
    csv_file = out_path / "benchmark_results.csv"
    fieldnames = list(results[0].keys())
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Exported benchmark CSV to {csv_file}")

    # 2. Generate Matplotlib Comparative Charts
    generate_benchmark_plots(results, out_path)

    return results


def generate_benchmark_plots(results: List[Dict[str, Any]], out_dir: Path) -> None:
    """Generate and save publication-quality Matplotlib comparative charts."""
    planners = ["BFS", "Dijkstra", "Greedy", "A*"]
    densities = sorted(list(set(r["density"] for r in results)))

    metrics_to_plot = [
        ("cost", "Path Cost", "path_cost.png"),
        ("nodes_expanded", "Nodes Expanded", "nodes_expanded.png"),
        ("runtime_ms", "Runtime (ms)", "runtime.png"),
    ]

    colors = {"BFS": "#e74c3c", "Dijkstra": "#3498db", "Greedy": "#e67e22", "A*": "#2ecc71"}

    for key, title, filename in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(densities))
        bar_width = 0.18

        for idx, p_name in enumerate(planners):
            means = []
            stds = []
            for d in densities:
                vals = [r[key] for r in results if r["planner"] == p_name and r["density"] == d and r["success"]]
                means.append(np.mean(vals) if vals else 0.0)
                stds.append(np.std(vals) if vals else 0.0)

            offset = (idx - 1.5) * bar_width
            ax.bar(
                x + offset,
                means,
                yerr=stds,
                width=bar_width,
                label=p_name,
                color=colors[p_name],
                capsize=3,
                edgecolor="black",
                linewidth=0.8,
            )

        ax.set_title(f"Planner Benchmark Comparison - {title}", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Obstacle Density", fontsize=12, labelpad=8)
        ax.set_ylabel(title, fontsize=12, labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(d*100)}%" for d in densities])
        ax.legend(title="Algorithm", frameon=True, facecolor="#f8f9fa")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_path = out_dir / filename
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Saved benchmark plot: {plot_path}")


if __name__ == "__main__":
    run_benchmark(num_maps=50, densities=[0.10, 0.20, 0.30])
