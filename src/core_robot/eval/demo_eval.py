"""Planner and SAM evaluation over the demonstration set.

Perception and planning see only the demo video. Ground truth (ground_truth_plan.json,
initial/final states) is read only after a plan exists, and only for scoring.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from core_robot.config import DEMOS_DIR, RUNS_DIR
from core_robot.meta_client import ApiError, ReplayMiss
from core_robot.perception.gate import sam_gate
from core_robot.perception.perceive import load_perception, perceive
from core_robot.planning.schema import Plan
from core_robot.planning.scoring import score
from core_robot.planning.video_planner import plan_from_video

# Failures of the API or the harness for one demo: recorded for that demo (and counted as
# a failure against the gate), never allowed to abort the remaining demos.
HARNESS_ERRORS = (ApiError, ReplayMiss, RuntimeError)

PERCEPTION_DIR = RUNS_DIR / "perception"
PLANNER_DIR = RUNS_DIR / "planner"


def demo_ids() -> list[str]:
    return sorted(f"{p.parent.name}/{p.name}" for p in DEMOS_DIR.glob("*/seed_*") if (p / "video.mp4").exists())


def demo_meta(demo_id: str) -> dict:
    """Video length and object count, for the dry-run cost estimate only."""
    d = DEMOS_DIR / demo_id
    return {"name": demo_id, "n_frames": json.loads((d / "result.json").read_text())["video_frames"],
            "n_objects": len(json.loads((d / "initial_state.json").read_text()))}


def perception_for(client, demo_id: str, out_root: Path = PERCEPTION_DIR, reuse: bool = True) -> dict:
    out = out_root / demo_id
    if reuse and (out / "perception.json").exists():
        return load_perception(out / "perception.json")
    return perceive(DEMOS_DIR / demo_id / "video.mp4", client, out)


def load_ground_truth_plan(demo_id: str) -> Plan:
    return Plan.model_validate_json((DEMOS_DIR / demo_id / "ground_truth_plan.json").read_text())


def evaluate_planner(client, ids: list[str], perception_root: Path = PERCEPTION_DIR,
                     out_root: Path = PLANNER_DIR, reasoning_effort: str | None = None, merge: bool = True,
                     over_budget=lambda: False) -> dict:
    """Plan each demo and score it. With merge, rows for other demos already in
    out_root/results.json are kept, so rerunning one demo does not drop the rest."""
    rows = []
    for demo_id in ids:
        if over_budget():  # stop before starting another demo
            break
        try:
            perception = perception_for(client, demo_id, perception_root)
            result = plan_from_video(client, perception["clip"]["path"], perception,
                                     reasoning_effort=reasoning_effort)
        except HARNESS_ERRORS as e:
            rows.append({"demo": demo_id, "named": [], "strict": False, "lenient": False,
                         "errors": ["harness error"], "planner_attempts": 0,
                         "planner_error": f"{type(e).__name__}: {e}", "tokens": [0, 0], "plan": None})
            continue
        gt = load_ground_truth_plan(demo_id)  # scoring only, after the plan exists
        s = score(result["plan"], gt)
        rows.append({
            "demo": demo_id, "named": perception["naming"]["objects"], "strict": s["strict"],
            "lenient": s["lenient"], "errors": s["errors"], "planner_attempts": len(result["attempts"]),
            "planner_error": result["error"], "tokens": list(result["tokens"]),
            "plan": result["plan"].model_dump(exclude_none=True) if result["plan"] else None,
            "attempts": result["attempts"], "reasoning_effort": reasoning_effort,
        })
    previous = out_root / "results.json"
    if merge and previous.exists():
        new = {r["demo"] for r in rows}
        rows = sorted([r for r in json.loads(previous.read_text())["rows"] if r["demo"] not in new] + rows,
                      key=lambda r: r["demo"])
    counts = Counter(e for r in rows for e in r["errors"])
    summary = {"demos": len(rows), "strict": sum(r["strict"] for r in rows),
               "lenient": sum(r["lenient"] for r in rows), "top_errors": counts.most_common(3), "rows": rows}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "results.json").write_text(json.dumps(summary, indent=1))
    return summary


def evaluate_sam(client, ids: list[str], perception_root: Path = PERCEPTION_DIR) -> dict:
    from core_robot.sim.env import TabletopEnv

    env = TabletopEnv()
    project = lambda p: env.project(p, "angled")
    rows = []
    for demo_id in ids:
        try:
            perception = perception_for(client, demo_id, perception_root)
        except HARNESS_ERRORS as e:
            rows.append({"demo": demo_id, "pass": False, "objects": {}, "overlay": None,
                         "error": f"{type(e).__name__}: {e}"})
            continue
        d = DEMOS_DIR / demo_id
        initial = json.loads((d / "initial_state.json").read_text())
        final = json.loads((d / "final_state.json").read_text())
        g = sam_gate(perception, initial, final, project)
        rows.append({"demo": demo_id, "pass": g["pass"], "objects": g["objects"],
                     "overlay": perception.get("overlay"), "error": None})
    env.close()
    summary = {"demos": len(rows), "passed": sum(r["pass"] for r in rows), "rows": rows}
    perception_root.mkdir(parents=True, exist_ok=True)
    (perception_root / "sam_gate.json").write_text(json.dumps(summary, indent=1))
    return summary
