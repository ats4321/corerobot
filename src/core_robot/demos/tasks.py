"""The 8 benchmark tasks: scene, ground-truth plan, and goal check for each.

A task's success check is exactly core_robot.verify.relations.check_success applied to
its ground-truth plan: every goal atom must hold, plus the global rule that every block
not moved by a step (including destination-only blocks) moves less than 2 cm.
Nothing else is tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from core_robot.planning.schema import FREE_TABLE_SPACE, GoalAtom, Plan, PlanObject, Step
from core_robot.sim.env import TabletopEnv
from core_robot.verify.relations import check_success

MAX_LAYOUT_TRIES = 200
FT = FREE_TABLE_SPACE

StepsFn = Callable[[dict], list[tuple]]  # (object, destination) or (object, destination, note)
GoalFn = Callable[[dict], list[tuple[str, str, str]]]


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    instruction: str
    objects: tuple[str, ...]
    steps: StepsFn  # scene -> [(object, destination[, note])]
    goal: GoalFn  # scene -> [(subject, relation, target)]
    accept: Callable[[dict], bool] = lambda scene: True  # extra layout constraint

    def scene_builder(self, env: TabletopEnv, seed: int) -> dict:
        """Reset env to this task's scene for `seed`; returns scene_state().

        Tries layout seeds seed*1000, seed*1000+1, ... and keeps the first layout that
        satisfies `accept` and where the goal does not already hold. Deterministic.
        """
        for k in range(MAX_LAYOUT_TRIES):
            scene = env.reset(seed * 1000 + k, objects=list(self.objects))
            if self.accept(scene) and not self.check(scene, scene)["success"]:
                return scene
        raise RuntimeError(f"{self.id}: no acceptable layout for seed {seed}")

    def ground_truth_plan(self, scene: dict) -> Plan:
        return Plan(
            objects=[PlanObject(name=n, color=v["color"], kind=v["kind"]) for n, v in scene.items()],
            steps=[
                Step(index=i, action="pick_and_place", object=s[0], destination=s[1],
                     note=s[2] if len(s) > 2 else None)
                for i, s in enumerate(self.steps(scene))
            ],
            goal=[GoalAtom(subject=s, relation=r, target=t) for s, r, t in self.goal(scene)],
        )

    def check(self, initial: dict, final: dict) -> dict:
        """Goal check: check_success(ground-truth plan, initial, final)."""
        return check_success(self.ground_truth_plan(initial), initial, final)


def _xy_dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def blocks_by_mug_distance(scene: dict) -> list[tuple[float, str]]:
    mug = scene["mug"]["position"]
    return sorted((_xy_dist(v["position"], mug), n) for n, v in scene.items() if v["kind"] == "block")


def _nearest_clear_by(margin: float):
    def accept(scene):
        d = blocks_by_mug_distance(scene)
        return d[1][0] - d[0][0] >= margin

    return accept


TASKS: list[Task] = [
    Task(
        id="t1_red_in_red_bin",
        title="Red block into the red bin",
        instruction="Put the red block in the red bin.",
        objects=("red_block", "red_bin", "blue_bin"),
        steps=lambda s: [("red_block", "red_bin")],
        goal=lambda s: [("red_block", "in", "red_bin")],
    ),
    Task(
        id="t2_sort_by_color",
        title="Sort red and blue blocks into matching bins",
        instruction="Put the red block in the red bin and the blue block in the blue bin.",
        objects=("red_block", "blue_block", "red_bin", "blue_bin"),
        steps=lambda s: [("red_block", "red_bin"), ("blue_block", "blue_bin")],
        goal=lambda s: [("red_block", "in", "red_bin"), ("blue_block", "in", "blue_bin")],
    ),
    Task(
        id="t3_stack_green_on_red",
        title="Stack the green block on the red block",
        instruction="Stack the green block on top of the red block.",
        objects=("red_block", "green_block", "blue_block", "yellow_block"),
        steps=lambda s: [("green_block", "red_block")],
        goal=lambda s: [("green_block", "on", "red_block")],
    ),
    Task(
        id="t4_stack_three",
        title="Stack three blocks: blue on green on red",
        instruction="Build a tower: green on red, then blue on green.",
        objects=("red_block", "green_block", "blue_block"),
        steps=lambda s: [("green_block", "red_block"), ("blue_block", "green_block")],
        goal=lambda s: [("green_block", "on", "red_block"), ("blue_block", "on", "green_block")],
    ),
    Task(
        id="t5_line_up",
        title="Line up red, green, blue from left to right",
        instruction="Line up the red, green and blue blocks in a row, left to right.",
        objects=("red_block", "green_block", "blue_block"),
        steps=lambda s: [("red_block", FT), ("green_block", FT), ("blue_block", FT)],
        goal=lambda s: [("red_block", "left_of", "green_block"), ("green_block", "left_of", "blue_block")],
    ),
    Task(
        id="t6_move_named_only",
        title="Move only the green block into the blue bin",
        instruction="Put the green block in the blue bin. Leave the other blocks where they are.",
        objects=("green_block", "red_block", "yellow_block", "blue_bin"),
        steps=lambda s: [("green_block", "blue_bin")],
        goal=lambda s: [("green_block", "in", "blue_bin")],
    ),
    Task(
        id="t7_swap_red_blue",
        title="Swap the red and blue blocks",
        instruction="Swap the positions of the red and blue blocks.",
        objects=("red_block", "blue_block", "green_block"),
        steps=lambda s: [("red_block", FT), ("blue_block", FT), ("red_block", FT)],
        goal=lambda s: [("red_block", "swapped_with", "blue_block")],
    ),
    Task(
        id="t8_nearest_to_mug",
        title="Put the block nearest the mug into the bin",
        instruction="Put the block that is closest to the mug into the red bin.",
        objects=("red_block", "green_block", "blue_block", "mug", "red_bin"),
        steps=lambda s: [(blocks_by_mug_distance(s)[0][1], "red_bin", "the block nearest the mug")],
        goal=lambda s: [(blocks_by_mug_distance(s)[0][1], "in", "red_bin")],
        accept=_nearest_clear_by(0.05),
    ),
]

TASKS_BY_ID = {t.id: t for t in TASKS}
