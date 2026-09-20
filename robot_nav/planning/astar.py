"""
A* search planner on weighted grids with path smoothing.
"""

import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from robot_nav.planning.grid import OccupancyGrid


@dataclass
class PlanResult:
    """Dataclass holding path planning outputs and execution metrics."""

    path: List[Tuple[int, int]]  # Discrete grid path [(r0, c0), (r1, c1), ...]
    smooth_path: List[Tuple[int, int]]  # Shortcut path [(r0, c0), ...]
    expanded_nodes: int  # Number of nodes popped from priority queue
    runtime_ms: float  # Planning duration in milliseconds
    cost: float  # Total g-cost of path
    g_score: Dict[Tuple[int, int], float]  # g-cost map for heatmap visualization
    closed_set: Set[Tuple[int, int]]  # Set of explored nodes


def compute_heuristic(
    node: Tuple[int, int], goal: Tuple[int, int], heuristic_type: str = "octile"
) -> float:
    """
    Compute heuristic distance from node to goal.
    :param node: (r, c)
    :param goal: (r, c)
    :param heuristic_type: 'manhattan', 'euclidean', or 'octile'
    """
    dr = abs(node[0] - goal[0])
    dc = abs(node[1] - goal[1])

    if heuristic_type == "manhattan":
        return float(dr + dc)
    elif heuristic_type == "euclidean":
        return math.sqrt(dr * dr + dc * dc)
    elif heuristic_type == "octile":
        return (dr + dc) + (math.sqrt(2.0) - 2.0) * min(dr, dc)
    else:
        raise ValueError(f"Unknown heuristic_type: {heuristic_type}")


def smooth_path_line_of_sight(
    grid: OccupancyGrid, path: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Smooth a discrete grid path using raycasting / line-of-sight shortcutting.
    :param grid: OccupancyGrid instance
    :param path: Original discrete grid path
    :return: Shortened waypoint path
    """
    if len(path) <= 2:
        return list(path)

    smoothed = [path[0]]
    current_idx = 0

    while current_idx < len(path) - 1:
        # Look ahead as far as possible
        next_idx = len(path) - 1
        for look_ahead in range(len(path) - 1, current_idx, -1):
            r1, c1 = path[current_idx]
            r2, c2 = path[look_ahead]
            if grid.line_of_sight(r1, c1, r2, c2):
                next_idx = look_ahead
                break

        if next_idx == current_idx:
            # Fallback if no line of sight to any future node
            next_idx = current_idx + 1

        smoothed.append(path[next_idx])
        current_idx = next_idx

    return smoothed


def astar_search(
    grid: OccupancyGrid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    connectivity: int = 8,
    heuristic_type: str = "octile",
    tie_break_eps: float = 1.001,
    smooth: bool = True,
) -> PlanResult:
    """
    A* path planning on a weighted occupancy grid.

    :param grid: OccupancyGrid object.
    :param start: Start grid coordinate (r, c).
    :param goal: Goal grid coordinate (r, c).
    :param connectivity: 4 or 8 connectivity.
    :param heuristic_type: 'octile', 'euclidean', or 'manhattan'.
    :param tie_break_eps: Factor multiplying heuristic for deterministic tie breaking.
    :param smooth: Whether to apply line-of-sight path shortcutting.
    :return: PlanResult dataclass.
    """
    start_time = time.perf_counter()

    if not grid.is_free(start[0], start[1]):
        return PlanResult([], [], 0, 0.0, float("inf"), {}, set())
    if not grid.is_free(goal[0], goal[1]):
        return PlanResult([], [], 0, 0.0, float("inf"), {}, set())

    # Handle start == goal case
    if start == goal:
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        return PlanResult(
            path=[start],
            smooth_path=[start],
            expanded_nodes=1,
            runtime_ms=runtime_ms,
            cost=0.0,
            g_score={start: 0.0},
            closed_set={start},
        )

    # Priority Queue entries: (f_score, counter, node)
    counter = 0
    open_pq: List[Tuple[float, int, Tuple[int, int]]] = []

    h0 = compute_heuristic(start, goal, heuristic_type) * tie_break_eps
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    closed_set: Set[Tuple[int, int]] = set()

    heapq.heappush(open_pq, (h0, counter, start))
    expanded_count = 0

    path_found = False

    while open_pq:
        _, _, current = heapq.heappop(open_pq)

        if current in closed_set:
            continue

        closed_set.add(current)
        expanded_count += 1

        if current == goal:
            path_found = True
            break

        curr_g = g_score[current]

        for neighbor, move_cost in grid.get_neighbors(current, connectivity):
            if neighbor in closed_set:
                continue

            tentative_g = curr_g + move_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                parent[neighbor] = current
                h = compute_heuristic(neighbor, goal, heuristic_type) * tie_break_eps
                f = tentative_g + h
                counter += 1
                heapq.heappush(open_pq, (f, counter, neighbor))

    runtime_ms = (time.perf_counter() - start_time) * 1000.0

    if not path_found:
        return PlanResult(
            path=[],
            smooth_path=[],
            expanded_nodes=expanded_count,
            runtime_ms=runtime_ms,
            cost=float("inf"),
            g_score=g_score,
            closed_set=closed_set,
        )

    # Reconstruct path
    curr = goal
    raw_path: List[Tuple[int, int]] = []
    while curr in parent:
        raw_path.append(curr)
        curr = parent[curr]
    raw_path.append(start)
    raw_path.reverse()

    cost = g_score[goal]

    if smooth:
        smoothed_path = smooth_path_line_of_sight(grid, raw_path)
    else:
        smoothed_path = list(raw_path)

    return PlanResult(
        path=raw_path,
        smooth_path=smoothed_path,
        expanded_nodes=expanded_count,
        runtime_ms=runtime_ms,
        cost=cost,
        g_score=g_score,
        closed_set=closed_set,
    )
