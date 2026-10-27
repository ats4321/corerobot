"""Scripted demonstrator: performs a task's ground-truth plan with the robot primitives.

Writes data/demos/<task_id>/seed_<n>/:
  video.mp4               overhead camera, 10 fps of simulated time
  ground_truth_plan.json  the Plan (object names / free_table_space only)
  initial_state.json      scene_state() before acting
  final_state.json        scene_state() after acting
  result.json             goal-check result, layout seed, primitive call log

"free_table_space" destinations are resolved to coordinates here, internally:
line-up spots for t5, original positions for the t7 swap, and the most open table
spot otherwise. Plans themselves never contain coordinates.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from core_robot.config import DEMOS_DIR
from core_robot.demos.tasks import Task
from core_robot.planning.schema import FREE_TABLE_SPACE
from core_robot.sim.env import OBJECTS, SIM_HZ, WORKSPACE_X, WORKSPACE_Y, TabletopEnv
from core_robot.sim.primitives import Robot
from core_robot.sim.render import Recorder

RADIUS = {o.name: o.radius for o in OBJECTS}
LINE_SPACING = 0.10
PAUSE_S = 0.5  # still frames before and after the demonstration


def _grid(step=0.02, margin=0.04):
    xs = np.arange(WORKSPACE_X[0] + margin, WORKSPACE_X[1] - margin + 1e-9, step)
    ys = np.arange(WORKSPACE_Y[0] + margin, WORKSPACE_Y[1] - margin + 1e-9, step)
    return [(float(x), float(y)) for x in xs for y in ys]


def _clearance(pt, scene, ignore=()) -> float:
    return min(
        (math.hypot(pt[0] - v["position"][0], pt[1] - v["position"][1]) - RADIUS[n]
         for n, v in scene.items() if n not in ignore),
        default=1.0,
    )


def free_spot(scene: dict, ignore=(), avoid=()) -> tuple[float, float]:
    """Most open table point: maximizes clearance from objects (and from `avoid` points)."""
    def score(pt):
        c = _clearance(pt, scene, ignore)
        for a in avoid:
            c = min(c, math.hypot(pt[0] - a[0], pt[1] - a[1]) - 0.05)
        return (round(c, 4), -abs(pt[0] - 0.5), -abs(pt[1]))  # tie-break toward table center

    return max(_grid(), key=score)


def line_up_spots(scene: dict, order: list[str]) -> list[tuple[float, float]]:
    """Row of spots, left (+y) to right, for blocks moved in `order`.

    Spot i must be clear of the blocks moved after it (they are still in place).
    Picks the valid row closest to the table center.
    """
    best, best_key = None, None
    for x, cy in _grid(step=0.01, margin=0.05):
        spots = [(x, cy + LINE_SPACING * (1 - i)) for i in range(len(order))]
        if not all(WORKSPACE_Y[0] + 0.04 <= sy <= WORKSPACE_Y[1] - 0.04 for _, sy in spots):
            continue
        ok = all(
            math.hypot(sx - scene[later]["position"][0], sy - scene[later]["position"][1]) >= 0.07
            for i, (sx, sy) in enumerate(spots) for later in order[i + 1:]
        )
        key = math.hypot(x - 0.5, cy)
        if ok and (best_key is None or key < best_key):
            best, best_key = spots, key
    if best is None:
        raise RuntimeError("no free row for line-up")
    return best


def concrete_targets(task: Task, scene: dict) -> list:
    """Per-step place targets: object names, or (x, y) for free_table_space steps."""
    steps = task.steps(scene)
    if task.id == "t5_line_up":
        spots = iter(line_up_spots(scene, [o for o, _ in steps]))
        return [next(spots) if d == FREE_TABLE_SPACE else d for _, d in steps]
    if task.id == "t7_swap_red_blue":
        r0 = tuple(scene["red_block"]["position"][:2])
        b0 = tuple(scene["blue_block"]["position"][:2])
        return [free_spot(scene, avoid=(r0, b0)), r0, b0]
    out = []
    for obj, d in steps:
        out.append(free_spot(scene, ignore=(obj,)) if d == FREE_TABLE_SPACE else d)
    return out


def run_demo(task: Task, seed: int, env: TabletopEnv, out_dir: Path | None = None) -> dict:
    """Build the scene, act out the ground-truth plan, check the goal, optionally save outputs."""
    task.scene_builder(env, seed)
    robot = Robot(env)
    env.step(int(0.2 * SIM_HZ))
    initial = env.scene_state()
    plan = task.ground_truth_plan(initial)
    targets = concrete_targets(task, initial)

    def act():
        env.step(int(PAUSE_S * SIM_HZ))
        for step, target in zip(plan.steps, targets):
            robot.pick(step.object)
            robot.place(step.object, target)
        env.step(int(PAUSE_S * SIM_HZ))

    frames = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with Recorder(env, out_dir / "video.mp4") as rec:
            act()
        frames = rec.frames
    else:
        act()

    final = env.scene_state()
    check = task.check(initial, final)
    result = {
        "task_id": task.id,
        "title": task.title,
        "instruction": task.instruction,
        "seed": seed,
        "layout_seed": env.seed,
        "success": check["success"],
        "check": check,
        "video_frames": frames,
        "call_log": robot.call_log,
    }
    if out_dir is not None:
        _dump(out_dir / "ground_truth_plan.json", plan.model_dump())
        _dump(out_dir / "initial_state.json", initial)
        _dump(out_dir / "final_state.json", final)
        _dump(out_dir / "result.json", result)
    return result


def demo_dir(task_id: str, seed: int) -> Path:
    return DEMOS_DIR / task_id / f"seed_{seed}"


def _dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=_jsonable) + "\n")


def _jsonable(o):
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))
