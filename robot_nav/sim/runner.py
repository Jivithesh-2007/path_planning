"""
Simulation Runner supporting interactive Pygame GUI and fast headless execution mode.
"""

import time
from typing import Optional

from robot_nav.sim.world import SimulationWorld
from robot_nav.sim.render import PygameRenderer
from robot_nav.config import SimConfig, DEFAULT_CONFIG


class SimulationRunner:
    """
    Runner for executing simulation sessions interactively or headless.
    """

    @staticmethod
    def run_headless(
        world: SimulationWorld, max_steps: int = 2000, dt: float = 0.05
    ) -> SimulationWorld:
        """
        Execute simulation in fast headless mode without rendering GUI.

        :param world: SimulationWorld instance.
        :param max_steps: Maximum step iterations before cutoff.
        :param dt: Time delta per step.
        :return: Completed SimulationWorld instance.
        """
        step_cnt = 0
        while not world.is_finished and step_cnt < max_steps:
            world.step(dt=dt)
            step_cnt += 1
        return world

    @staticmethod
    def run_interactive(
        world: SimulationWorld, config: SimConfig = DEFAULT_CONFIG
    ) -> SimulationWorld:
        """
        Execute simulation in 60 FPS interactive Pygame GUI mode.

        :param world: SimulationWorld instance.
        :param config: System configuration.
        :return: Completed SimulationWorld instance.
        """
        renderer = PygameRenderer(world, config)
        running = True

        while running:
            running = renderer.handle_events()
            if not world.is_paused:
                world.step(dt=config.control.dt)
            renderer.render()

        import pygame
        pygame.quit()
        return world
