"""Task success check from simulator state only (never by asking a model).

check(task, scene_state, initial_state, plan) evaluates the plan's goal relations and the
global rule (verify.relations.unmoved_blocks_still). Pass the task's ground-truth plan
for the execution layout when scoring; a model-written plan must not grade itself.
"""

from __future__ import annotations

import math

from core_robot.planning.schema import GoalAtom, Plan
from core_robot.verify.relations import (
    HEIGHT, KIND, LR_MAX_DX, LR_MIN_DY, NEXT_TO_MAX, _is_on, holds, unmoved_blocks_still,
)


def _xy(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def goal_error_cm(atom: GoalAtom, final: dict, initial: dict) -> float:
    """How far a goal relation is from holding, in cm.

    in            horizontal distance from subject to the bin's center
    on            3D distance from subject to its resting spot on top of target
    left_of/right_of  lateral shortfall below 3 cm plus front-back excess above 15 cm
    next_to       center distance beyond 9 cm (or the height gap if the two are stacked)
    swapped_with  the larger distance from either block to the other's start
    """
    s, t = final[atom.subject]["position"], final[atom.target]["position"]
    rel = atom.relation
    if rel == "in":
        err = _xy(s, t)
    elif rel == "on":
        rest_z = t[2] + HEIGHT.get(KIND[atom.target], 0) / 2 + HEIGHT.get(KIND[atom.subject], 0) / 2
        err = math.dist(s, (t[0], t[1], rest_z))
    elif rel in ("left_of", "right_of"):
        dy = (s[1] - t[1]) if rel == "left_of" else (t[1] - s[1])
        err = max(0.0, LR_MIN_DY - dy) + max(0.0, abs(s[0] - t[0]) - LR_MAX_DX)
    elif rel == "next_to":
        stacked = "bin" not in (KIND[atom.subject], KIND[atom.target]) and (
            _is_on(s, t, atom.subject, atom.target)[0] or _is_on(t, s, atom.target, atom.subject)[0]
        )
        err = abs(s[2] - t[2]) if stacked else max(0.0, _xy(s, t) - NEXT_TO_MAX)
    elif rel == "swapped_with":
        err = max(_xy(s, initial[atom.target]["position"]), _xy(t, initial[atom.subject]["position"]))
    else:
        err = float("nan")
    return round(err * 100, 1)


def goal_error_vector_cm(atom: GoalAtom, final: dict, initial: dict) -> list[float] | None:
    """(dx, dy) in cm from where the subject should be to where it is, for relations with a
    single ideal spot: in (bin center), on (target center), swapped_with (target's start).
    None for left_of / right_of / next_to."""
    s = final[atom.subject]["position"]
    if atom.relation in ("in", "on"):
        ideal = final[atom.target]["position"]
    elif atom.relation == "swapped_with":
        ideal = initial[atom.target]["position"]
    else:
        return None
    return [round((s[0] - ideal[0]) * 100, 1), round((s[1] - ideal[1]) * 100, 1)]


def check(task, scene_state: dict | None, initial_state: dict | None, plan: Plan) -> dict:
    """Returns {task_id, success, per_goal: [...], unnamed_ok, unmoved: [...]}.

    per_goal entries: {relation, subject, target, ok, error_cm, detail}.
    unnamed_ok: the global rule holds (no block that no step moves has moved 2 cm or more).
    """
    task_id = getattr(task, "id", str(task))
    if scene_state is None or initial_state is None:
        return {
            "task_id": task_id, "success": False, "unnamed_ok": False, "unmoved": [],
            "per_goal": [
                {"relation": g.relation, "subject": g.subject, "target": g.target, "ok": False,
                 "error_cm": None, "detail": "no final state (run did not finish)"}
                for g in plan.goal
            ],
        }
    per_goal = []
    for atom in plan.goal:
        ok, detail = holds(atom, scene_state, initial_state)
        per_goal.append({
            "relation": atom.relation, "subject": atom.subject, "target": atom.target, "ok": ok,
            "error_cm": goal_error_cm(atom, scene_state, initial_state),
            "error_vec_cm": goal_error_vector_cm(atom, scene_state, initial_state), "detail": detail,
        })
    unmoved = [
        {"block": name, "ok": ok, "moved_cm": round(_xy(scene_state[name]["position"],
                                                        initial_state[name]["position"]) * 100, 1)}
        for name, ok, _ in unmoved_blocks_still(plan, initial_state, scene_state)
    ]
    unnamed_ok = all(u["ok"] for u in unmoved)
    return {
        "task_id": task_id,
        "success": all(g["ok"] for g in per_goal) and unnamed_ok,
        "per_goal": per_goal,
        "unnamed_ok": unnamed_ok,
        "unmoved": unmoved,
    }
