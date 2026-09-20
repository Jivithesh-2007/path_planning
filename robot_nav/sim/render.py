"""
Pygame 60 FPS GUI Renderer.
Displays terrain cost grid, A* node expansion heatmap, path, trail, robot, sensor rays, and HUD overlay.
"""

import math
import pygame
import numpy as np
from typing import Optional, Tuple

from robot_nav.sim.world import SimulationWorld
from robot_nav.config import SimConfig, DEFAULT_CONFIG


class PygameRenderer:
    """
    Rich interactive visualizer using Pygame for real-time simulation rendering.
    """

    def __init__(self, world: SimulationWorld, config: SimConfig = DEFAULT_CONFIG) -> None:
        self.world = world
        self.config = config

        pygame.init()
        pygame.font.init()

        self.width = config.render.screen_width
        self.height = config.render.screen_height

        # Reserve top 70 pixels for HUD banner
        self.hud_height = 75
        self.grid_render_height = self.height - self.hud_height

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(config.render.title)

        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Helvetica", 16, bold=True)
        self.font_hud = pygame.font.SysFont("Helvetica", 14)
        self.font_large = pygame.font.SysFont("Helvetica", 22, bold=True)

        # Scale factors from world meters to screen pixels
        self.cell_w_px = self.width / world.perceived_grid.width
        self.cell_h_px = self.grid_render_height / world.perceived_grid.height

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous world coordinates (meters) to screen pixel coordinates."""
        px = int(x / (self.world.perceived_grid.width * self.world.cell_size) * self.width)
        py = int(
            self.hud_height
            + y / (self.world.perceived_grid.height * self.world.cell_size) * self.grid_render_height
        )
        return (px, py)

    def grid_to_screen_rect(self, r: int, c: int) -> pygame.Rect:
        """Convert grid cell (r, c) to Pygame Rect."""
        x = int(c * self.cell_w_px)
        y = int(self.hud_height + r * self.cell_h_px)
        w = max(1, int(self.cell_w_px))
        h = max(1, int(self.cell_h_px))
        return pygame.Rect(x, y, w, h)

    def render(self) -> None:
        """Draw full simulation frame."""
        self.screen.fill((20, 24, 33))  # Dark sleek background

        # 1. Draw Terrain & Occupancy Grid
        grid_h = self.world.perceived_grid.height
        grid_w = self.world.perceived_grid.width

        for r in range(grid_h):
            for c in range(grid_w):
                rect = self.grid_to_screen_rect(r, c)
                perc_val = self.world.perceived_grid.grid[r, c]
                gt_val = self.world.ground_truth_grid.grid[r, c]

                if perc_val == 1:
                    # Perceived obstacle: dark slate
                    color = (45, 50, 62)
                elif gt_val == 1 and perc_val == 0:
                    # Undiscovered hidden obstacle in ground truth: dark reddish tint
                    color = (85, 45, 45)
                elif perc_val == 2:
                    # Rough terrain: golden warm brown
                    color = (195, 145, 60)
                else:
                    # Free space: clean off-white / light gray
                    color = (235, 238, 242)

                pygame.draw.rect(self.screen, color, rect)

        # Draw subtle grid lines
        for r in range(0, grid_h + 1, 5):
            y = int(self.hud_height + r * self.cell_h_px)
            pygame.draw.line(self.screen, (210, 215, 222), (0, y), (self.width, y), 1)
        for c in range(0, grid_w + 1, 5):
            x = int(c * self.cell_w_px)
            pygame.draw.line(self.screen, (210, 215, 222), (x, self.hud_height), (x, self.height), 1)

        # 2. Draw A* Expanded Nodes Heatmap
        if self.world.plan_result and self.world.plan_result.closed_set:
            expanded_surface = pygame.Surface((self.width, self.grid_render_height), pygame.SRCALPHA)
            for r, c in self.world.plan_result.closed_set:
                rect = pygame.Rect(
                    int(c * self.cell_w_px),
                    int(r * self.cell_h_px),
                    max(1, int(self.cell_w_px)),
                    max(1, int(self.cell_h_px)),
                )
                # Cyan / Soft Blue transparent overlay
                pygame.draw.rect(expanded_surface, (0, 180, 255, 65), rect)
            self.screen.blit(expanded_surface, (0, self.hud_height))

        # 3. Draw Start and Goal
        start_rect = self.grid_to_screen_rect(self.world.start_cell[0], self.world.start_cell[1])
        goal_rect = self.grid_to_screen_rect(self.world.goal_cell[0], self.world.goal_cell[1])
        pygame.draw.rect(self.screen, (40, 200, 80), start_rect)  # Green Start
        pygame.draw.rect(self.screen, (230, 50, 50), goal_rect)  # Red Goal

        # Labels for Start & Goal
        start_px = self.world_to_screen(
            (self.world.start_cell[1] + 0.5) * self.world.cell_size,
            (self.world.start_cell[0] + 0.5) * self.world.cell_size,
        )
        goal_px = self.world_to_screen(
            (self.world.goal_cell[1] + 0.5) * self.world.cell_size,
            (self.world.goal_cell[0] + 0.5) * self.world.cell_size,
        )
        self.screen.blit(self.font_main.render("S", True, (255, 255, 255)), (start_px[0] - 5, start_px[1] - 8))
        self.screen.blit(self.font_main.render("G", True, (255, 255, 255)), (goal_px[0] - 5, goal_px[1] - 8))

        # 4. Draw Planned Path (Bold Blue Line)
        if len(self.world.world_path) >= 2:
            pts = [self.world_to_screen(wx, wy) for wx, wy in self.world.world_path]
            pygame.draw.lines(self.screen, (0, 102, 255), False, pts, 3)
            for pt in pts:
                pygame.draw.circle(self.screen, (0, 70, 200), pt, 4)

        # 5. Draw Traveled Trajectory Trail (Green Line)
        if len(self.world.traveled_trail) >= 2:
            trail_pts = [self.world_to_screen(tx, ty) for tx, ty in self.world.traveled_trail]
            pygame.draw.lines(self.screen, (255, 140, 0), False, trail_pts, 2)

        # 6. Draw Lidar Sensor Rays (Fan of Yellow Rays)
        rx, ry, rtheta = self.world.robot.get_pose()
        robot_px = self.world_to_screen(rx, ry)

        if self.world.ray_endpoints:
            sensor_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            for ex, ey in self.world.ray_endpoints:
                ep_px = self.world_to_screen(ex, ey)
                pygame.draw.line(sensor_surface, (255, 235, 50, 45), robot_px, ep_px, 1)
            self.screen.blit(sensor_surface, (0, 0))

        # 7. Draw Robot Body & Heading Vector
        robot_r_px = int(self.world.robot.radius / (self.world.perceived_grid.width * self.world.cell_size) * self.width)
        robot_r_px = max(6, robot_r_px)

        # Chassis circle
        pygame.draw.circle(self.screen, (30, 30, 35), robot_px, robot_r_px + 2)
        pygame.draw.circle(self.screen, (0, 210, 255), robot_px, robot_r_px)

        # Heading direction arrow
        head_len = robot_r_px * 2.2
        hx = robot_px[0] + head_len * math.cos(rtheta)
        hy = robot_px[1] + head_len * math.sin(rtheta)
        pygame.draw.line(self.screen, (255, 255, 255), robot_px, (hx, hy), 3)

        # 8. Render Top HUD Banner
        hud_rect = pygame.Rect(0, 0, self.width, self.hud_height)
        pygame.draw.rect(self.screen, (15, 18, 26), hud_rect)
        pygame.draw.line(self.screen, (0, 180, 255), (0, self.hud_height - 1), (self.width, self.hud_height - 1), 2)

        # Text strings
        expanded_cnt = self.world.plan_result.expanded_nodes if self.world.plan_result else 0
        plan_time = self.world.plan_result.runtime_ms if self.world.plan_result else 0.0
        curr_cte = self.world.cross_track_errors[-1] if self.world.cross_track_errors else 0.0
        max_cte = max(self.world.cross_track_errors) if self.world.cross_track_errors else 0.0
        replan_cnt = len(self.world.replanner.history) - 1  # Excluding initial plan

        col1 = f"Planner: {self.world.planner_name} | Nodes Exp: {expanded_cnt} ({plan_time:.1f}ms)"
        col2 = f"Time: {self.world.sim_time:.2f}s | Replans: {max(0, replan_cnt)} | Dist: {self.world.total_distance_traveled:.2f}m"
        col3 = f"CTE: {curr_cte:.3f}m (Max: {max_cte:.3f}m)"

        status_str = "PAUSED" if self.world.is_paused else ("FINISHED" if self.world.is_finished else "RUNNING")
        keys_str = "SPACE: Pause | R: Replan | N: New Map | ESC: Quit"

        txt_col1 = self.font_main.render(col1, True, (0, 220, 255))
        txt_col2 = self.font_hud.render(col2, True, (220, 225, 230))
        txt_col3 = self.font_hud.render(col3, True, (255, 180, 60))

        status_color = (255, 200, 0) if self.world.is_paused else ((0, 255, 120) if self.world.is_finished else (100, 255, 100))
        txt_status = self.font_large.render(status_str, True, status_color)
        txt_keys = self.font_hud.render(keys_str, True, (160, 170, 185))

        self.screen.blit(txt_col1, (12, 8))
        self.screen.blit(txt_col2, (12, 28))
        self.screen.blit(txt_col3, (12, 48))

        self.screen.blit(txt_status, (self.width - 150, 8))
        self.screen.blit(txt_keys, (self.width - 340, 48))

        pygame.display.flip()
        self.clock.tick(self.config.render.fps)

    def handle_events(self) -> bool:
        """
        Process Pygame user inputs.

        :return: False if quit requested (ESC or window close), True otherwise.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.world.is_paused = not self.world.is_paused
                elif event.key == pygame.K_r:
                    self.world.replan(reason="user_keypress_R")
        return True
