"""
Grid representation and neighbor generation with diagonal corner-cutting guards.
"""

import math
import numpy as np
from typing import List, Tuple

class OccupancyGrid:
    """
    Encapsulates 2D occupancy grid and traversal cost map.
    Occupancy values: 0 = free, 1 = obstacle, 2 = rough_terrain.
    """

    def __init__(
        self,
        grid: np.ndarray,
        cost_grid: np.ndarray | None = None,
        free_cost: float = 1.0,
        rough_cost: float = 3.0,
        obstacle_cost: float = float("inf"),
    ) -> None:
        """
        Initialize occupancy grid.
        :param grid: 2D uint8 numpy array (0: free, 1: obstacle, 2: rough_terrain)
        :param cost_grid: Optional 2D float32 array specifying cell traversal costs.
        """
        if grid.ndim != 2:
            raise ValueError(f"Grid must be 2D, got shape {grid.shape}")

        self.grid = grid.astype(np.uint8)
        self.height, self.width = grid.shape
        self.free_cost = free_cost
        self.rough_cost = rough_cost
        self.obstacle_cost = obstacle_cost

        if cost_grid is not None:
            if cost_grid.shape != grid.shape:
                raise ValueError("cost_grid shape must match grid shape")
            self.cost_grid = cost_grid.astype(np.float32)
        else:
            # Generate default cost grid from occupancy
            self.cost_grid = np.full(grid.shape, free_cost, dtype=np.float32)
            self.cost_grid[self.grid == 1] = obstacle_cost
            self.cost_grid[self.grid == 2] = rough_cost

    def in_bounds(self, r: int, c: int) -> bool:
        """Check if grid index (r, c) is within boundaries."""
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, r: int, c: int) -> bool:
        """Check if grid index (r, c) is inside bounds and traversable (not an obstacle)."""
        if not self.in_bounds(r, c):
            return False
        return bool(self.grid[r, c] != 1 and not math.isinf(self.cost_grid[r, c]))

    def get_cost(self, r: int, c: int) -> float:
        """Return traversal cost for cell (r, c)."""
        if not self.in_bounds(r, c):
            return float("inf")
        return float(self.cost_grid[r, c])

    def get_neighbors(
        self, node: Tuple[int, int], connectivity: int = 8
    ) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get valid neighboring cells and edge movement costs.
        Enforces diagonal corner-cutting prevention: diagonal moves are forbidden if
        either adjacent cardinal neighbor is an obstacle.

        :param node: (r, c) tuple.
        :param connectivity: 4 or 8.
        :return: List of ((neighbor_r, neighbor_c), move_cost).
        """
        r, c = node
        neighbors: List[Tuple[Tuple[int, int], float]] = []

        # 4 cardinal directions: Up, Down, Left, Right
        cardinal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in cardinal_dirs:
            nr, nc = r + dr, c + dc
            if self.is_free(nr, nc):
                cost = self.get_cost(nr, nc)
                neighbors.append(((nr, nc), cost))

        if connectivity == 8:
            # 4 diagonal directions: UP-LEFT, UP-RIGHT, DOWN-LEFT, DOWN-RIGHT
            diagonal_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

            for dr, dc in diagonal_dirs:
                nr, nc = r + dr, c + dc
                if self.is_free(nr, nc):
                    # Check diagonal corner-cutting guard:
                    # Both adjacent cardinal cells (r+dr, c) and (r, c+dc) MUST be free!
                    cardinal1_free = self.is_free(r + dr, c)
                    cardinal2_free = self.is_free(r, c + dc)

                    if cardinal1_free and cardinal2_free:
                        # Diagonal movement cost is sqrt(2) * target_cell_cost
                        cost = math.sqrt(2.0) * self.get_cost(nr, nc)
                        neighbors.append(((nr, nc), cost))

        return neighbors

    def line_of_sight(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """
        Check if straight line of sight exists between (r1, c1) and (r2, c2)
        using Bresenham's line algorithm with corner-cutting prevention.
        """
        # Uses integer Bresenham raycasting
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        err = dr - dc

        curr_r, curr_c = r1, c1

        while True:
            if not self.is_free(curr_r, curr_c):
                return False

            if curr_r == r2 and curr_c == c2:
                break

            e2 = 2 * err
            prev_r, prev_c = curr_r, curr_c

            if e2 > -dc:
                err -= dc
                curr_r += sr
            if e2 < dr:
                err += dr
                curr_c += sc

            # If both row and col changed (diagonal step), check corner cutting
            if curr_r != prev_r and curr_c != prev_c:
                if not (self.is_free(prev_r, curr_c) and self.is_free(curr_r, prev_c)):
                    return False

        return True
