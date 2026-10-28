"""A task defined by a predicted plan (real clips have no ground-truth task).

The scene holds exactly the plan's objects at seeded random positions; success is judged
against the plan's own goal atoms plus the global rule. That checks that the robot did what
the plan says, not that the plan matches what the person in the clip did.
"""

from __future__ import annotations

from dataclasses import dataclass

from core_robot.planning.schema import Plan
from core_robot.verify.relations import check_success

MAX_LAYOUT_TRIES = 200
JUDGED_AGAINST = ("Success is judged against the plan's own goal atoms (plus the global rule): it checks "
                  "that the robot did what the plan says, not that the plan matches what the person in the "
                  "clip did.")


@dataclass(frozen=True)
class PlanTask:
    name: str
    plan_json: str  # Plan.model_dump_json(); kept as text so the task is hashable and picklable

    @property
    def plan(self) -> Plan:
        return Plan.model_validate_json(self.plan_json)

    @property
    def id(self) -> str:
        return f"real:{self.name}"

    @property
    def objects(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.plan.objects)

    def scene_builder(self, env, seed: int) -> dict:
        """Plan objects at seeded random positions; skips layouts where the goal already holds."""
        for k in range(MAX_LAYOUT_TRIES):
            scene = env.reset(seed * 1000 + k, objects=list(self.objects))
            if not self.check(scene, scene)["success"]:
                return scene
        raise RuntimeError(f"{self.id}: every layout already satisfies the plan's goal")

    def ground_truth_plan(self, scene: dict) -> Plan:
        """The reference for judging is the plan itself."""
        return self.plan

    def check(self, initial: dict, final: dict) -> dict:
        return check_success(self.plan, initial, final)

    def spec(self) -> dict:
        """What the sandbox child needs to rebuild this task."""
        return {"kind": "plan", "name": self.name, "plan_json": self.plan_json}


def task_from_spec(spec: dict) -> PlanTask:
    return PlanTask(spec["name"], spec["plan_json"])
