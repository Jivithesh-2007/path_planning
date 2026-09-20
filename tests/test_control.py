"""
Tests for control module: Differential drive kinematics, pure pursuit tracking, and cross-track error.
"""

import math
import pytest

from robot_nav.control.robot import DifferentialDriveRobot, normalize_angle
from robot_nav.control.pure_pursuit import (
    PurePursuitController,
    grid_to_world_path,
    compute_cross_track_error,
)


def test_normalize_angle():
    assert pytest.approx(normalize_angle(0.0)) == 0.0
    assert pytest.approx(abs(normalize_angle(3 * math.pi))) == math.pi
    assert pytest.approx(abs(normalize_angle(-3 * math.pi))) == math.pi



def test_differential_drive_speed_clamping():
    robot = DifferentialDriveRobot(x=0, y=0, theta=0, v_max=1.0, w_max=2.0, a_max=10.0, alpha_max=10.0)

    # Command speeds exceeding max limit
    state = robot.step(v_cmd=5.0, w_cmd=10.0, dt=0.1)

    assert state.v <= 1.0
    assert state.w <= 2.0


def test_straight_line_kinematics():
    robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0, v_max=2.0, a_max=10.0)

    # Move straight in X direction for 1 second at 1.0 m/s
    for _ in range(20):
        robot.step(v_cmd=1.0, w_cmd=0.0, dt=0.05)

    assert robot.state.x > 0.9 and robot.state.x <= 1.05
    assert pytest.approx(robot.state.y, abs=1e-3) == 0.0
    assert pytest.approx(robot.state.theta, abs=1e-3) == 0.0


def test_cross_track_error_calculation():
    waypoints = [(0.0, 0.0), (10.0, 0.0)]  # Straight line along X axis

    # Robot at (5.0, 0.5) -> cross track error should be 0.5
    cte, idx = compute_cross_track_error((5.0, 0.5), waypoints)
    assert pytest.approx(cte, abs=1e-3) == 0.5
    assert idx == 0


def test_pure_pursuit_path_tracking_convergence():
    """
    Test that PurePursuitController successfully steers robot along an L-shaped path
    and reaches the goal with decreasing cross-track error.
    """
    grid_path = [(0, 0), (0, 5), (5, 5)]
    world_path = grid_to_world_path(grid_path, cell_size_m=0.2)

    robot = DifferentialDriveRobot(x=world_path[0][0], y=world_path[0][1], theta=0.0)
    controller = PurePursuitController(lookahead_dist=0.4, v_max=1.0, w_max=2.0)

    reached = False
    ctes = []

    for _ in range(300):  # 15 seconds sim time at dt=0.05
        v_cmd, w_cmd, cte, is_goal = controller.compute_command(robot.state, world_path)
        ctes.append(cte)
        if is_goal:
            reached = True
            break
        robot.step(v_cmd, w_cmd, dt=0.05)

    assert reached is True, "Robot failed to reach goal via Pure Pursuit!"
    assert min(ctes) < 0.2, "Cross track error did not decrease during path following"
