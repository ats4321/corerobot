"""Real-clip pipeline: normalize -> perception -> plan -> scene from the plan -> code -> sandbox,
with condition C repair (up to 3 attempts). Prompts are the frozen simulator prompts.

Each stage is reported through on_stage(name, payload) as it completes, for the CLI and the UI.
Stages: "normalized", "perception", "plan", "attempt" (once per attempt), "outcome".
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

from core_robot.config import (
    PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M, RUNS_DIR, SAM_MIN_FRAMES, SAM_PRICE_PER_FRAME, SAM_PRICE_PER_IMAGE,
)
from core_robot.real.plan_task import JUDGED_AGAINST, PlanTask

REAL_DIR = RUNS_DIR / "real"
REAL_SEED = 300  # outside the demo (0, 1), evaluation (100-104) and pilot (200-204) seeds
MAX_OBJECTS = 7  # the simulator vocabulary: worst case, every object is named

# Measured averages so far (docs/findings_v1.md): used for the expected estimate only.
EXPECTED = {"naming": (1800, 300), "planner": (4900, 2750), "codegen": (1750, 2500), "repair": (2500, 6400)}
# Worst-case bounds, matching the budget guard's reservations.
WORST = {"naming": (5000, 2000), "planner_first": (43000, 6000), "planner_retry": (43000, 16000),
         "codegen": (7000, 12000)}


def _usd(t_in: float, t_out: float) -> float:
    return t_in * PRICE_INPUT_PER_M / 1e6 + t_out * PRICE_OUTPUT_PER_M / 1e6


def slug(path: str | Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).stem)[:60] or "clip"


def estimate_cost(duration_s: float, n_objects: int = MAX_OBJECTS, expected_objects: int = 4) -> dict:
    """Dry-run estimate for one clip of this (normalized) length."""
    frames = max(SAM_MIN_FRAMES, math.ceil(duration_s * 5) + 1)  # 5 fps clip, billed minimum 50
    sam_one = frames * SAM_PRICE_PER_FRAME
    sam_worst_per_obj = 2 * sam_one + (4 + 8) * SAM_PRICE_PER_IMAGE  # video, probes, retry video, keyframes
    worst = {
        "naming": _usd(*WORST["naming"]),
        "sam": n_objects * sam_worst_per_obj,
        "planner": _usd(*WORST["planner_first"]) + _usd(*WORST["planner_retry"]),
        "codegen": 3 * _usd(*WORST["codegen"]),
    }
    expected = {
        "naming": _usd(*EXPECTED["naming"]),
        "sam": expected_objects * sam_one,
        "planner": _usd(*EXPECTED["planner"]),
        "codegen": _usd(*EXPECTED["codegen"]) + _usd(*EXPECTED["repair"]),
    }
    return {"frames_billed_per_sam_video": frames, "expected": expected, "expected_total": sum(expected.values()),
            "worst": worst, "worst_total": sum(worst.values()),
            "assumptions": f"expected: {expected_objects} objects, one SAM video each, one planner call, "
                           f"attempt 1 plus one repair; worst: all {n_objects} objects with every SAM fallback, "
                           "planner retried at 16k output, three code generations at the 12k output cap"}


def run_pipeline(clip: str | Path, client, out_root: Path = REAL_DIR, seed: int = REAL_SEED,
                 on_stage: Callable[[str, dict], None] | None = None, record: bool = True,
                 footage: str = "real") -> dict:
    """Run the whole pipeline on one clip. Returns the summary also written to summary.json.

    footage: perception mode ("real" for camera footage; "sim" is the simulator-demo path).
    If <clip>.expected_plan.json exists, the summary gets plan_matches_clip (separate from the
    simulator success flag)."""
    from core_robot.codegen.muse_generator import MuseGenerator
    from core_robot.perception.perceive import perceive, summary_text
    from core_robot.planning.video_planner import plan_from_video
    from core_robot.prompt_freeze import git_commit, prompt_hash
    from core_robot.real.normalize import normalize_clip
    from core_robot.repair.loop import run_with_repair

    stage = on_stage or (lambda name, payload: None)
    name = slug(clip)
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"clip": str(clip), "name": name, "seed": seed, "prompt_hash": prompt_hash(),
                     "git_commit": git_commit(), "judged_against": JUDGED_AGAINST, "footage_mode": footage}

    def save():
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    from core_robot.meta_client import ApiError, BudgetExceeded, ReplayMiss

    def stopped(where: str, e: Exception) -> dict:
        summary["outcome"] = {"success": False, "reason": f"stopped during {where}: {type(e).__name__}: {e}"}
        stage("outcome", summary["outcome"])
        save()
        return summary

    norm = normalize_clip(clip, out / "normalized.mp4")
    summary["normalized"] = norm
    stage("normalized", norm)

    try:
        perception = perceive(norm["path"], client, out / "perception", footage=footage)
    except (ApiError, BudgetExceeded, ReplayMiss) as e:
        return stopped("perception", e)
    summary["perception"] = {"named": perception["naming"]["objects"], "naming_error": perception["naming"]["error"],
                             "overlay": perception["overlay"], "tracks": summary_text(perception),
                             "coverage": {n: round(t["coverage"], 2) for n, t in perception["tracks"].items()},
                             "phrases": {n: t.get("concept") for n, t in perception["tracks"].items()},
                             "chosen": {n: {"from_phrase": t.get("concept"), "object_id": t.get("object_id"),
                                            "frames": [min(t["frames"]), max(t["frames"])] if t["frames"] else None,
                                            "n_frames": len(t["frames"])} for n, t in perception["tracks"].items()},
                             "real_mode": perception.get("real_mode")}
    stage("perception", summary["perception"])

    try:
        planned = plan_from_video(client, perception["clip"]["path"], perception)
    except (ApiError, BudgetExceeded, ReplayMiss) as e:
        return stopped("planning", e)
    plan = planned["plan"]
    summary["plan"] = plan.model_dump(exclude_none=True) if plan else None
    summary["planner_attempts"] = planned["attempts"]
    from core_robot.real.expected import compare_plan, expected_plan_path, load_expected

    expected = load_expected(clip)
    if expected is not None:
        summary["plan_vs_clip"] = {**compare_plan(summary["plan"], expected),
                                   "expected_plan": str(expected_plan_path(clip))}
    if plan is not None:
        (out / "plan.json").write_text(json.dumps(summary["plan"], indent=1))
    stage("plan", {"plan": summary["plan"], "error": planned["error"], "plan_vs_clip": summary.get("plan_vs_clip")})
    if plan is None:
        summary["outcome"] = {"success": False, "reason": f"no valid plan: {planned['error']}"}
        stage("outcome", summary["outcome"])
        save()
        return summary

    task = PlanTask(name, plan.model_dump_json())
    meta = {"run_source": "live", "plan_source": "predicted_real", "environment": "clean",
            "prompt_hash": summary["prompt_hash"], "git_commit": summary["git_commit"]}
    log = run_with_repair(task, plan, seed, "C", MuseGenerator(client), out_dir=out / "repair", record=record,
                          meta=meta, on_attempt=lambda a: stage("attempt", a))
    last = log["attempts"][-1] if log["attempts"] else {}
    summary["outcome"] = {
        "success": log["success"], "attempts": log["attempts_used"], "judged_against": JUDGED_AGAINST,
        "goal": [f"{g['subject']} {g['relation']} {g['target']}: {'ok' if g['ok'] else 'NO'}"
                 for g in (last.get("outcome") or {}).get("per_goal", [])],
        "report": last.get("report"), "video": last.get("video"), "run_log": log.get("log_path"),
    }
    stage("outcome", summary["outcome"])
    save()
    return summary
