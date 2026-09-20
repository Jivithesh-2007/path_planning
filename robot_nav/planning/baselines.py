"""
Baseline path planners (BFS, Dijkstra, Greedy Best-First) matching the A* interface.
"""

from collections import deque
import heapq
import time
from typing import Dict, List, Set, Tuple

from robot_nav.planning.astar import PlanResult, compute_heuristic, smooth_path_line_of_sight
from robot_nav.planning.grid import OccupancyGrid


def bfs_search(
    grid: OccupancyGrid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    connectivity: int = 8,
    smooth: bool = True,
) -> PlanResult:
    """
    Breadth-First Search baseline (unweighted edge steps).
    """
    start_time = time.perf_counter()

    if not grid.is_free(start[0], start[1]) or not grid.is_free(goal[0], goal[1]):
        return PlanResult([], [], 0, 0.0, float("inf"), {}, set())

    if start == goal:
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        return PlanResult([start], [start], 1, runtime_ms, 0.0, {start: 0.0}, {start})

    queue: deque[Tuple[int, int]] = deque([start])
    visited: Set[Tuple[int, int]] = {start}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    expanded_count = 0

    path_found = False

    while queue:
        current = queue.popleft()
        expanded_count += 1

        if current == goal:
            path_found = True
            break

        curr_g = g_score[current]

        for neighbor, move_cost in grid.get_neighbors(current, connectivity):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                g_score[neighbor] = curr_g + move_cost
                queue.append(neighbor)

    runtime_ms = (time.perf_counter() - start_time) * 1000.0

    if not path_found:
        return PlanResult([], [], expanded_count, runtime_ms, float("inf"), g_score, visited)

    raw_path: List[Tuple[int, int]] = []
    curr = goal
    while curr in parent:
        raw_path.append(curr)
        curr = parent[curr]
    raw_path.append(start)
    raw_path.reverse()

    cost = g_score[goal]
    smoothed_path = smooth_path_line_of_sight(grid, raw_path) if smooth else list(raw_path)

    return PlanResult(raw_path, smoothed_path, expanded_count, runtime_ms, cost, g_score, visited)


def dijkstra_search(
    grid: OccupancyGrid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    connectivity: int = 8,
    smooth: bool = True,
) -> PlanResult:
    """
    Dijkstra's Algorithm baseline (A* with h(n) = 0).
    """
    start_time = time.perf_counter()

    if not grid.is_free(start[0], start[1]) or not grid.is_free(goal[0], goal[1]):
        return PlanResult([], [], 0, 0.0, float("inf"), {}, set())

    if start == goal:
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        return PlanResult([start], [start], 1, runtime_ms, 0.0, {start: 0.0}, {start})

    counter = 0
    open_pq: List[Tuple[float, int, Tuple[int, int]]] = []
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    closed_set: Set[Tuple[int, int]] = set()

    heapq.heappush(open_pq, (0.0, counter, start))
    expanded_count = 0
    path_found = False

    while open_pq:
        curr_g, _, current = heapq.heappop(open_pq)

        if current in closed_set:
            continue

        closed_set.add(current)
        expanded_count += 1

        if current == goal:
            path_found = True
            break

        for neighbor, move_cost in grid.get_neighbors(current, connectivity):
            if neighbor in closed_set:
                continue

            tentative_g = curr_g + move_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                parent[neighbor] = current
                counter += 1
                heapq.heappush(open_pq, (tentative_g, counter, neighbor))

    runtime_ms = (time.perf_counter() - start_time) * 1000.0

    if not path_found:
        return PlanResult([], [], expanded_count, runtime_ms, float("inf"), g_score, closed_set)

    raw_path: List[Tuple[int, int]] = []
    curr = goal
    while curr in parent:
        raw_path.append(curr)
        curr = parent[curr]
    raw_path.append(start)
    raw_path.reverse()

    cost = g_score[goal]
    smoothed_path = smooth_path_line_of_sight(grid, raw_path) if smooth else list(raw_path)

    return PlanResult(raw_path, smoothed_path, expanded_count, runtime_ms, cost, g_score, closed_set)


def greedy_best_first_search(
    grid: OccupancyGrid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    connectivity: int = 8,
    heuristic_type: str = "octile",
    smooth: bool = True,
) -> PlanResult:
    """
    Greedy Best-First Search baseline (prioritizes pure heuristic h(n)).
    """
    start_time = time.perf_counter()

    if not grid.is_free(start[0], start[1]) or not grid.is_free(goal[0], goal[1]):
        return PlanResult([], [], 0, 0.0, float("inf"), {}, set())

    if start == goal:
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        return PlanResult([start], [start], 1, runtime_ms, 0.0, {start: 0.0}, {start})

    counter = 0
    open_pq: List[Tuple[float, int, Tuple[int, int]]] = []
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    closed_set: Set[Tuple[int, int]] = set()

    h0 = compute_heuristic(start, goal, heuristic_type)
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
                h = compute_heuristic(neighbor, goal, heuristic_type)
                counter += 1
                heapq.heappush(open_pq, (h, counter, neighbor))

    runtime_ms = (time.perf_counter() - start_time) * 1000.0

    if not path_found:
        return PlanResult([], [], expanded_count, runtime_ms, float("inf"), g_score, closed_set)

    raw_path: List[Tuple[int, int]] = []
    curr = goal
    while curr in parent:
        raw_path.append(curr)
        curr = parent[curr]
    raw_path.append(start)
    raw_path.reverse()

    cost = g_score[goal]
    smoothed_path = smooth_path_line_of_sight(grid, raw_path) if smooth else list(raw_path)

    return PlanResult(raw_path, smoothed_path, expanded_count, runtime_ms, cost, g_score, closed_set)
