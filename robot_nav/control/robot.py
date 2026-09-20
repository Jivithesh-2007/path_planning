"""
Kinematic differential-drive robot model with velocity and acceleration limits.
"""

import math
from dataclasses import dataclass
from typing import Tuple


def normalize_angle(angle: float) -> float:
    """Normalize angle in radians to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class RobotState:
    """State representation of a differential-drive robot in continuous 2D space."""

    x: float  # World X position in meters
    y: float  # World Y position in meters
    theta: float  # Heading orientation in radians [-pi, pi]
    v: float = 0.0  # Current linear velocity m/s
    w: float = 0.0  # Current angular velocity rad/s


class DifferentialDriveRobot:
    """
    Differential-drive robot kinematic simulator.
    Continuous non-holonomic state updates:
    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    theta_dot = w
    """

    def __init__(

        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        v_max: float = 1.5,
        w_max: float = 2.5,
        a_max: float = 2.0,
        alpha_max: float = 4.0,
        radius: float = 0.25,
    ) -> None:
        self.state = RobotState(x=x, y=y, theta=normalize_angle(theta), v=0.0, w=0.0)
        self.v_max = v_max
        self.w_max = w_max
        self.a_max = a_max
        self.alpha_max = alpha_max
        self.radius = radius

    def reset(self, x: float, y: float, theta: float = 0.0) -> None:
        """Reset robot pose and velocities."""
        self.state = RobotState(x=x, y=y, theta=normalize_angle(theta), v=0.0, w=0.0)

    def step(self, v_cmd: float, w_cmd: float, dt: float) -> RobotState:
        """
        Advance robot state by dt given velocity commands.

        :param v_cmd: Commanded linear velocity (m/s).
        :param w_cmd: Commanded angular velocity (rad/s).
        :param dt: Time delta (seconds).
        :return: Updated RobotState.
        """
        # Clamp velocity bounds
        target_v = max(-self.v_max, min(self.v_max, v_cmd))
        target_w = max(-self.w_max, min(self.w_max, w_cmd))

        # Acceleration limiting
        dv_max = self.a_max * dt
        dw_max = self.alpha_max * dt

        dv = max(-dv_max, min(dv_max, target_v - self.state.v))
        dw = max(-dw_max, min(dw_max, target_w - self.state.w))

        self.state.v += dv
        self.state.w += dw

        # Kinematic integration
        self.state.x += self.state.v * math.cos(self.state.theta) * dt
        self.state.y += self.state.v * math.sin(self.state.theta) * dt
        self.state.theta = normalize_angle(self.state.theta + self.state.w * dt)

        return self.state

    def get_pose(self) -> Tuple[float, float, float]:
        """Return (x, y, theta) pose tuple."""
        return (self.state.x, self.state.y, self.state.theta)
