"""
Replanning manager for dynamic path validation and replan event logging.
"""

from dataclasses import dataclass
import time
from typing import Callable, List, Optional, Tuple

from robot_nav.planning.astar import PlanResult
from robot_nav.planning.grid import OccupancyGrid


@dataclass
class ReplanEvent:
    """Represents a single dynamic path replan event."""

    timestamp: float  # Unix timestamp of replan event
    robot_cell: Tuple[int, int]  # Robot's grid position at replan
    goal_cell: Tuple[int, int]  # Goal grid coordinate
    reason: str  # Cause e.g. "path_blocked_by_obstacle" or "manual_trigger"
    old_remaining_steps: int  # Remaining steps on invalidated path
    new_path_steps: int  # Step count of newly computed path
    success: bool  # Whether replanning succeeded in finding a path


class Replanner:
    """
    Monitors current path against updated perceived occupancy map and triggers replanning when blocked.
    """

    def __init__(self) -> None:
        self.history: List[ReplanEvent] = []

    def is_path_blocked(
        self, grid: OccupancyGrid, path: List[Tuple[int, int]], start_idx: int = 0
    ) -> bool:
        """
        Check if any waypoint in remaining path intersects an obstacle cell.
        :param grid: Current perceived grid.
        :param path: Current planned path.
        :param start_idx: Index of closest current robot waypoint.
        :return: True if remaining path is blocked by obstacle.
        """
        if not path or start_idx >= len(path):
            return False

        for r, c in path[start_idx:]:
            if not grid.is_free(r, c):
                return True
        return False

    def check_and_replan(
        self,
        grid: OccupancyGrid,
        robot_cell: Tuple[int, int],
        goal_cell: Tuple[int, int],
        current_path: List[Tuple[int, int]],
        current_path_idx: int,
        planner_fn: Callable[[OccupancyGrid, Tuple[int, int], Tuple[int, int]], PlanResult],
        force_replan: bool = False,
        reason: str = "path_blocked",
    ) -> Tuple[bool, List[Tuple[int, int]], Optional[ReplanEvent]]:
        """
        Validate path and trigger replan if blocked or forced.

        :param grid: Updated occupancy grid.
        :param robot_cell: Current robot grid coordinate.
        :param goal_cell: Goal coordinate.
        :param current_path: Active discrete grid path.
        :param current_path_idx: Index of current target waypoint in path.
        :param planner_fn: Function executing path search.
        :param force_replan: Whether to force replanning regardless of blockage.
        :param reason: Reason description.
        :return: (did_replan: bool, updated_path: List[(r, c)], event: ReplanEvent | None)
        """
        blocked = self.is_path_blocked(grid, current_path, current_path_idx)

        if not blocked and not force_replan:
            return False, current_path, None

        actual_reason = reason if force_replan else "obstacle_detected_on_path"
        remaining_steps = max(0, len(current_path) - current_path_idx)

        # Plan new path starting from current robot cell
        res = planner_fn(grid, robot_cell, goal_cell)

        event = ReplanEvent(
            timestamp=time.time(),
            robot_cell=robot_cell,
            goal_cell=goal_cell,
            reason=actual_reason,
            old_remaining_steps=remaining_steps,
            new_path_steps=len(res.smooth_path if res.smooth_path else res.path),
            success=bool(res.path),
        )

        self.history.append(event)
        new_path = res.smooth_path if res.smooth_path else res.path

        return True, new_path, event
