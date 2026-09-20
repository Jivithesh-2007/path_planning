"""
Pure pursuit path tracking controller and cross-track error computation.
"""

import math
from typing import List, Tuple
from robot_nav.control.robot import RobotState, normalize_angle


def grid_to_world_path(
    grid_path: List[Tuple[int, int]], cell_size_m: float = 0.2
) -> List[Tuple[float, float]]:
    """
    Convert discrete grid coordinates (row, col) to continuous world coordinates (x, y).
    :param grid_path: List of (row, col) tuples.
    :param cell_size_m: Meter size per cell.
    :return: List of (x, y) continuous world coordinates.
    """
    world_path = []
    for r, c in grid_path:
        x = (c + 0.5) * cell_size_m
        y = (r + 0.5) * cell_size_m
        world_path.append((x, y))
    return world_path


def point_to_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Compute perpendicular distance from point (px, py) to line segment (ax, ay)-(bx, by)."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab2 = abx * abx + aby * aby
    if ab2 < 1e-9:
        return math.hypot(apx, apy)

    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    proj_x = ax + t * abx
    proj_y = ay + t * aby

    return math.hypot(px - proj_x, py - proj_y)


def compute_cross_track_error(
    robot_pos: Tuple[float, float], waypoints: List[Tuple[float, float]]
) -> Tuple[float, int]:
    """
    Compute minimum cross-track error (perpendicular distance to path) and closest segment index.

    :param robot_pos: (x, y) robot position.
    :param waypoints: List of path waypoints.
    :return: (min_distance, closest_segment_index)
    """
    if not waypoints:
        return 0.0, 0
    if len(waypoints) == 1:
        return math.hypot(robot_pos[0] - waypoints[0][0], robot_pos[1] - waypoints[0][1]), 0

    min_dist = float("inf")
    closest_idx = 0

    for i in range(len(waypoints) - 1):
        ax, ay = waypoints[i]
        bx, by = waypoints[i + 1]
        dist = point_to_segment_distance(robot_pos[0], robot_pos[1], ax, ay, bx, by)

        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    return min_dist, closest_idx


class PurePursuitController:
    """
    Pure pursuit geometric trajectory tracking controller for differential-drive robots.
    """

    def __init__(
        self, lookahead_dist: float = 0.6, v_max: float = 1.5, w_max: float = 2.5
    ) -> None:
        self.lookahead_dist = lookahead_dist
        self.v_max = v_max
        self.w_max = w_max
        self.target_idx = 0

    def compute_command(
        self, robot_state: RobotState, waypoints_world: List[Tuple[float, float]]
    ) -> Tuple[float, float, float, bool]:
        """
        Compute linear (v) and angular (w) velocity steering commands.

        :param robot_state: Current RobotState.
        :param waypoints_world: Path waypoints in world frame.
        :return: (v_cmd, w_cmd, cross_track_error, is_goal_reached)
        """
        if not waypoints_world:
            return 0.0, 0.0, 0.0, True

        rx, ry, rtheta = robot_state.x, robot_state.y, robot_state.theta
        goal_x, goal_y = waypoints_world[-1]
        dist_to_goal = math.hypot(goal_x - rx, goal_y - ry)

        # Check goal arrival tolerance
        if dist_to_goal < 0.2:
            cte, _ = compute_cross_track_error((rx, ry), waypoints_world)
            return 0.0, 0.0, cte, True

        # Find closest segment
        cte, closest_idx = compute_cross_track_error((rx, ry), waypoints_world)

        # Find lookahead target point along path segments
        target_point = waypoints_world[-1]

        # Interpolate along line segments starting from closest_idx
        for i in range(closest_idx, len(waypoints_world) - 1):
            ax, ay = waypoints_world[i]
            bx, by = waypoints_world[i + 1]

            # Vector from A to B
            dx, dy = bx - ax, by - ay
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-6:
                continue

            # Project robot onto segment to find parameter t
            apx, apy = rx - ax, ry - ay
            t_proj = max(0.0, min(1.0, (apx * dx + apy * dy) / (seg_len * seg_len)))

            # From projection point, move lookahead_dist along segment
            t_target = t_proj + (self.lookahead_dist / seg_len)
            if t_target <= 1.0:
                target_point = (ax + t_target * dx, ay + t_target * dy)
                break
            else:
                # Target goes past point B; continue to next segment
                target_point = (bx, by)

        tx, ty = target_point
        dx = tx - rx
        dy = ty - ry

        # Transform target to robot local frame
        local_x = dx * math.cos(rtheta) + dy * math.sin(rtheta)
        local_y = -dx * math.sin(rtheta) + dy * math.cos(rtheta)

        l_dist = math.hypot(local_x, local_y)
        if l_dist < 1e-4:
            return 0.0, 0.0, cte, False

        # Heading angle error alpha
        alpha = math.atan2(local_y, local_x)

        # Pure pursuit curvature kappa = 2 * local_y / l_dist^2
        kappa = (2.0 * local_y) / (l_dist * l_dist)

        # Velocity & steering command
        if abs(alpha) > math.pi / 3.0:
            # Turn in place towards lookahead point if heading error > 60 deg
            v_cmd = 0.1
            w_cmd = self.w_max if alpha > 0 else -self.w_max
        else:
            v_base = min(self.v_max, max(0.3, dist_to_goal * 1.5))
            v_cmd = v_base * max(0.2, math.cos(alpha))
            w_cmd = v_cmd * kappa

        v_cmd = max(0.0, min(self.v_max, v_cmd))
        w_cmd = max(-self.w_max, min(self.w_max, w_cmd))

        return v_cmd, w_cmd, cte, False


