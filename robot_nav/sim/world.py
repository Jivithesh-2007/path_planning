"""
Simulation World Manager.
Combines ground truth environment, robot-perceived grid, kinematic robot model, sensor, dynamic replanning, and trail tracking.
"""

import math
import numpy as np
from typing import Callable, List, Optional, Set, Tuple

from robot_nav.config import SimConfig, DEFAULT_CONFIG
from robot_nav.control.robot import DifferentialDriveRobot, RobotState
from robot_nav.control.pure_pursuit import PurePursuitController, grid_to_world_path, compute_cross_track_error
from robot_nav.planning.grid import OccupancyGrid
from robot_nav.planning.astar import astar_search, PlanResult
from robot_nav.planning.replan import Replanner, ReplanEvent
from robot_nav.sim.sensor import LidarSensor


class SimulationWorld:
    """
    Main simulation state manager unifying perception, planning, control, and physical simulation.
    """

    def __init__(
        self,
        ground_truth_array: np.ndarray,
        perceived_array: np.ndarray,
        start_cell: Tuple[int, int],
        goal_cell: Tuple[int, int],
        config: SimConfig = DEFAULT_CONFIG,
        planner_name: str = "A*",
    ) -> None:
        self.config = config
        self.planner_name = planner_name
        self.cell_size = config.map.cell_size_m

        self.start_cell = start_cell
        self.goal_cell = goal_cell

        # Grids
        self.ground_truth_grid = OccupancyGrid(ground_truth_array)
        self.perceived_grid = OccupancyGrid(perceived_array)

        # Initial robot pose at start cell center
        start_x = (start_cell[1] + 0.5) * self.cell_size
        start_y = (start_cell[0] + 0.5) * self.cell_size
        goal_x = (goal_cell[1] + 0.5) * self.cell_size
        goal_y = (goal_cell[0] + 0.5) * self.cell_size

        init_theta = math.atan2(goal_y - start_y, goal_x - start_x)

        self.robot = DifferentialDriveRobot(
            x=start_x,
            y=start_y,
            theta=init_theta,
            v_max=config.control.v_max,
            w_max=config.control.w_max,
            a_max=config.control.a_max,
            alpha_max=config.control.alpha_max,
            radius=config.control.robot_radius,
        )

        self.controller = PurePursuitController(
            lookahead_dist=config.control.lookahead_dist,
            v_max=config.control.v_max,
            w_max=config.control.w_max,
        )

        self.sensor = LidarSensor(
            range_m=config.sensor.range_m,
            fov_deg=config.sensor.fov_deg,
            num_rays=config.sensor.num_rays,
        )

        self.replanner = Replanner()

        # Path & Planning state
        self.plan_result: Optional[PlanResult] = None
        self.current_path: List[Tuple[int, int]] = []
        self.smooth_path: List[Tuple[int, int]] = []
        self.world_path: List[Tuple[float, float]] = []

        # Tracking & metrics history
        self.traveled_trail: List[Tuple[float, float]] = [(start_x, start_y)]
        self.cross_track_errors: List[float] = [0.0]
        self.sim_time: float = 0.0
        self.total_distance_traveled: float = 0.0
        self.is_paused: bool = False
        self.is_finished: bool = False
        self.ray_endpoints: List[Tuple[float, float]] = []

        # Execute initial path plan
        self.replan(reason="initial_plan")

    def _get_planner_fn(self) -> Callable[[OccupancyGrid, Tuple[int, int], Tuple[int, int]], PlanResult]:
        """Return path planner function corresponding to planner_name."""
        from robot_nav.planning.baselines import bfs_search, dijkstra_search, greedy_best_first_search

        if self.planner_name == "BFS":
            return lambda g, s, e: bfs_search(g, s, e, connectivity=self.config.planner.connectivity, smooth=self.config.planner.smooth_path)
        elif self.planner_name == "Dijkstra":
            return lambda g, s, e: dijkstra_search(g, s, e, connectivity=self.config.planner.connectivity, smooth=self.config.planner.smooth_path)
        elif self.planner_name == "Greedy":
            return lambda g, s, e: greedy_best_first_search(g, s, e, connectivity=self.config.planner.connectivity, heuristic_type=self.config.planner.heuristic_type, smooth=self.config.planner.smooth_path)
        else:  # Default A*
            return lambda g, s, e: astar_search(
                g, s, e,
                connectivity=self.config.planner.connectivity,
                heuristic_type=self.config.planner.heuristic_type,
                tie_break_eps=self.config.planner.tie_break_eps,
                smooth=self.config.planner.smooth_path,
            )

    def get_robot_cell(self) -> Tuple[int, int]:
        """Convert robot's continuous position to discrete grid (r, c)."""
        r = int(np.clip(self.robot.state.y / self.cell_size, 0, self.perceived_grid.height - 1))
        c = int(np.clip(self.robot.state.x / self.cell_size, 0, self.perceived_grid.width - 1))
        return (r, c)

    def replan(self, reason: str = "manual_trigger") -> PlanResult:
        """Run path planning from robot's current cell to goal cell."""
        robot_cell = self.get_robot_cell()
        planner_fn = self._get_planner_fn()

        res = planner_fn(self.perceived_grid, robot_cell, self.goal_cell)
        self.plan_result = res
        self.current_path = res.path
        self.smooth_path = res.smooth_path if res.smooth_path else res.path
        self.world_path = grid_to_world_path(self.smooth_path, self.cell_size)

        event = ReplanEvent(
            timestamp=self.sim_time,
            robot_cell=robot_cell,
            goal_cell=self.goal_cell,
            reason=reason,
            old_remaining_steps=len(self.current_path),
            new_path_steps=len(self.smooth_path),
            success=bool(res.path),
        )
        self.replanner.history.append(event)
        return res

    def step(self, dt: float = 0.05) -> None:
        """Advance simulation world by one time step dt."""
        if self.is_paused or self.is_finished:
            return

        self.sim_time += dt
        rx, ry, rtheta = self.robot.get_pose()

        # 1. Lidar Scan & Perception Update
        endpoints, revealed_free, revealed_obs = self.sensor.scan(
            self.ground_truth_grid.grid, (rx, ry, rtheta), self.cell_size
        )
        self.ray_endpoints = endpoints

        # Update perceived occupancy map with newly discovered obstacles
        newly_found_obstacle = False
        for r, c in revealed_obs:
            if self.perceived_grid.grid[r, c] != 1:
                self.perceived_grid.grid[r, c] = 1
                self.perceived_grid.cost_grid[r, c] = float("inf")
                newly_found_obstacle = True

        for r, c in revealed_free:
            if self.perceived_grid.grid[r, c] == 1 and self.ground_truth_grid.grid[r, c] == 0:
                self.perceived_grid.grid[r, c] = 0
                self.perceived_grid.cost_grid[r, c] = self.config.map.free_cost

        # 2. Dynamic Replanning Check
        robot_cell = self.get_robot_cell()
        planner_fn = self._get_planner_fn()

        did_replan, updated_path, event = self.replanner.check_and_replan(
            self.perceived_grid,
            robot_cell,
            self.goal_cell,
            self.smooth_path,
            current_path_idx=0,
            planner_fn=planner_fn,
            force_replan=False,
        )

        if did_replan:
            self.smooth_path = updated_path
            self.world_path = grid_to_world_path(updated_path, self.cell_size)

        # 3. Pure Pursuit Control Execution
        v_cmd, w_cmd, cte, is_goal = self.controller.compute_command(self.robot.state, self.world_path)

        # 4. Kinematic Motion Integration
        prev_x, prev_y = rx, ry
        self.robot.step(v_cmd, w_cmd, dt)
        new_x, new_y, _ = self.robot.get_pose()

        # Update tracking metrics
        step_dist = math.hypot(new_x - prev_x, new_y - prev_y)
        self.total_distance_traveled += step_dist
        self.traveled_trail.append((new_x, new_y))
        self.cross_track_errors.append(cte)

        if is_goal:
            self.is_finished = True
