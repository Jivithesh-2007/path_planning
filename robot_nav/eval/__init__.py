"""
Evaluation subpackage: Metrics collection and multi-planner benchmarking.
"""

from robot_nav.eval.metrics import compute_trajectory_metrics
from robot_nav.eval.benchmark import run_benchmark

__all__ = ["compute_trajectory_metrics", "run_benchmark"]
