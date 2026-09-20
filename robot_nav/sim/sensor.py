"""
2D LiDAR / FOV Raycasting Sensor.
Simulates directional line-of-sight range measurements and reveals map cells to the robot.
"""

import math
import numpy as np
from typing import List, Set, Tuple


class LidarSensor:
    """
    2D LiDAR sensor with configurable range and Field of View (FOV).
    Performs raycasting on ground truth world grid to discover obstacles and free space.
    """

    def __init__(
        self, range_m: float = 6.0, fov_deg: float = 120.0, num_rays: int = 61
    ) -> None:
        self.range_m = range_m
        self.fov_rad = math.radians(fov_deg)
        self.num_rays = max(3, num_rays)

    def scan(
        self,
        ground_truth_grid: np.ndarray,
        robot_pose: Tuple[float, float, float],
        cell_size_m: float = 0.2,
    ) -> Tuple[List[Tuple[float, float]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Perform 2D FOV raycast scan from current robot pose.

        :param ground_truth_grid: 2D uint8 ground truth map array (1 = obstacle).
        :param robot_pose: (rx, ry, rtheta) in meters and radians.
        :param cell_size_m: Grid cell width in meters.
        :return: (ray_endpoints: List[(x, y)], revealed_free: Set[(r, c)], revealed_obstacles: Set[(r, c)])
        """
        rx, ry, rtheta = robot_pose
        grid_h, grid_w = ground_truth_grid.shape

        start_angle = rtheta - (self.fov_rad / 2.0)
        angle_step = self.fov_rad / (self.num_rays - 1)

        ray_endpoints: List[Tuple[float, float]] = []
        revealed_free: Set[Tuple[int, int]] = set()
        revealed_obstacles: Set[Tuple[int, int]] = set()

        ds = cell_size_m * 0.3  # Step size along ray
        max_steps = int(self.range_m / ds)

        for i in range(self.num_rays):
            ray_angle = start_angle + i * angle_step
            cos_a = math.cos(ray_angle)
            sin_a = math.sin(ray_angle)

            hit_obstacle = False
            endpoint = (rx + self.range_m * cos_a, ry + self.range_m * sin_a)

            for step in range(1, max_steps + 1):
                dist = step * ds
                px = rx + dist * cos_a
                py = ry + dist * sin_a

                r = int(py / cell_size_m)
                c = int(px / cell_size_m)

                # Check grid bounds
                if r < 0 or r >= grid_h or c < 0 or c >= grid_w:
                    endpoint = (px, py)
                    break

                if ground_truth_grid[r, c] == 1:
                    # Ray hit obstacle!
                    revealed_obstacles.add((r, c))
                    endpoint = (px, py)
                    hit_obstacle = True
                    break
                else:
                    revealed_free.add((r, c))

            if not hit_obstacle and step == max_steps:
                endpoint = (rx + self.range_m * cos_a, ry + self.range_m * sin_a)

            ray_endpoints.append(endpoint)

        return ray_endpoints, revealed_free, revealed_obstacles
