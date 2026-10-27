"""The execution scene: built identically by the sandbox child and by the prompt builder."""

from __future__ import annotations

import copy

SCENE_FIELDS = ("position", "orientation", "color", "kind")


def build_exec_scene(task, exec_seed: int):
    """Reset a fresh env to (task, exec_seed), home the robot, settle. Returns (env, robot, state)."""
    from core_robot.sim.env import SIM_HZ, TabletopEnv
    from core_robot.sim.primitives import Robot

    env = TabletopEnv()
    task.scene_builder(env, exec_seed)
    robot = Robot(env)
    env.step(int(0.2 * SIM_HZ))
    return env, robot, env.scene_state()


def exec_initial_state(task, exec_seed: int) -> dict:
    """The starting scene_state() that run(robot, scene) will see for this execution seed."""
    env, _, state = build_exec_scene(task, exec_seed)
    env.close()
    return state


def read_only_scene(state: dict) -> dict:
    """Deep copy limited to the public fields (name -> position, orientation, color, kind)."""
    return {name: {k: copy.deepcopy(v[k]) for k in SCENE_FIELDS} for name, v in state.items()}
