"""Hand-labelled expected plans for real clips, and the plan_matches_clip check.

<clip>.expected_plan.json sits next to the clip and lists the steps (action, object,
destination) and goal atoms a person labelled from watching it. plan_matches_clip is True when
the generated plan has the same steps in the same order with the same destinations, and the
same set of goal atoms. It is separate from the simulator success flag, which only checks that
the robot carried out the plan.
"""

from __future__ import annotations

import json
from pathlib import Path


def expected_plan_path(clip: str | Path) -> Path:
    clip = Path(clip)
    return clip.with_name(f"{clip.stem}.expected_plan.json")


def load_expected(clip: str | Path) -> dict | None:
    p = expected_plan_path(clip)
    return json.loads(p.read_text()) if p.exists() else None


def compare_plan(plan: dict | None, expected: dict) -> dict:
    """{"plan_matches_clip": bool, "differences": [...]} for a generated plan dict (or None)."""
    if plan is None:
        return {"plan_matches_clip": False, "differences": ["no valid plan was generated"]}
    steps = [(s["action"], s["object"], s["destination"]) for s in plan["steps"]]
    want_steps = [(s["action"], s["object"], s["destination"]) for s in expected["steps"]]
    goal = {(g["subject"], g["relation"], g["target"]) for g in plan["goal"]}
    want_goal = {(g["subject"], g["relation"], g["target"]) for g in expected["goal"]}
    diffs = []
    if steps != want_steps:
        extra = [s for s in steps if s not in want_steps]
        missing = [s for s in want_steps if s not in steps]
        if extra:
            diffs.append(f"extra steps: {extra}")
        if missing:
            diffs.append(f"missing steps: {missing}")
        if not extra and not missing:
            diffs.append(f"steps in a different order: {steps} vs expected {want_steps}")
    if goal != want_goal:
        if goal - want_goal:
            diffs.append(f"extra goal atoms: {sorted(goal - want_goal)}")
        if want_goal - goal:
            diffs.append(f"missing goal atoms: {sorted(want_goal - goal)}")
    return {"plan_matches_clip": not diffs, "differences": diffs}
