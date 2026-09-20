"""
Control subpackage: Kinematic differential drive robot model and pure pursuit trajectory tracking.
"""

from robot_nav.control.robot import RobotState, DifferentialDriveRobot
from robot_nav.control.pure_pursuit import PurePursuitController, grid_to_world_path, compute_cross_track_error

__all__ = [
    "RobotState",
    "DifferentialDriveRobot",
    "PurePursuitController",
    "grid_to_world_path",
    "compute_cross_track_error",
]
