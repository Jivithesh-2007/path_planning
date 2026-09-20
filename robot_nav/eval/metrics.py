"""
Quantitative metric computation for trajectory, path quality, and planner efficiency.
"""

from typing import Dict, Any
from robot_nav.sim.world import SimulationWorld


def compute_trajectory_metrics(world: SimulationWorld) -> Dict[str, Any]:
    """
    Compute comprehensive summary metrics for a completed simulation run.

    :param world: SimulationWorld instance.
    :return: Dictionary of performance metrics.
    """
    plan_res = world.plan_result

    path_cost = plan_res.cost if plan_res else float("inf")
    expanded_nodes = plan_res.expanded_nodes if plan_res else 0
    runtime_ms = plan_res.runtime_ms if plan_res else 0.0

    mean_cte = float(sum(world.cross_track_errors) / max(1, len(world.cross_track_errors)))
    max_cte = float(max(world.cross_track_errors)) if world.cross_track_errors else 0.0

    replan_count = max(0, len(world.replanner.history) - 1)

    return {
        "planner": world.planner_name,
        "sim_time_s": round(world.sim_time, 3),
        "path_cost": round(path_cost, 3),
        "discrete_steps": len(world.current_path),
        "smooth_waypoints": len(world.smooth_path),
        "distance_traveled_m": round(world.total_distance_traveled, 3),
        "mean_cte_m": round(mean_cte, 4),
        "max_cte_m": round(max_cte, 4),
        "expanded_nodes": expanded_nodes,
        "planning_runtime_ms": round(runtime_ms, 3),
        "replan_count": replan_count,
        "reached_goal": world.is_finished,
    }
