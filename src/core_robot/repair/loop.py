"""Generate -> run in sandbox -> check -> (on failure) feed back -> retry.

Conditions:
  A  one attempt, no feedback
  B  up to 3 attempts; the text failure report is passed back via generate_fn(feedback=...)
  C  as B, plus the path of the final angled-camera frame via generate_fn(image_path=...)

Every attempt runs in a fresh sandbox process that resets the simulator to the same
execution seed, so attempts never inherit the previous attempt's end state. Success is
judged against the task's ground-truth plan for the execution layout, not the plan given
to the generator. One JSON log per run records each attempt's code, report and outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from core_robot.codegen.sandbox import DEFAULT_TIMEOUT_S, run_code
from core_robot.config import RUNS_DIR
from core_robot.planning.schema import Plan
from core_robot.sim.env import DEMO_CAMERA
from core_robot.verify import checker
from core_robot.verify.report import build_failure_report

MAX_ATTEMPTS = {"A": 1, "B": 3, "C": 3}


class GenerateFn(Protocol):
    def __call__(self, *, task, plan: Plan, attempt: int, feedback: str | None = None,
                 image_path: str | None = None) -> str: ...


def run_dir(task_id: str, condition: str, exec_seed: int) -> Path:
    return RUNS_DIR / "repair" / task_id / condition / f"seed_{exec_seed}"


def run_with_repair(
    task,
    plan: Plan,
    exec_seed: int,
    condition: str,
    generate_fn: GenerateFn,
    out_dir: str | Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    record: bool = True,
) -> dict:
    """Run one (task, exec_seed, condition). Stops at the first success. Returns the run log."""
    if condition not in MAX_ATTEMPTS:
        raise ValueError(f"condition must be one of {sorted(MAX_ATTEMPTS)}, got {condition!r}")
    out = Path(out_dir) if out_dir else run_dir(task.id, condition, exec_seed)
    out.mkdir(parents=True, exist_ok=True)
    log = {
        "task_id": task.id, "condition": condition, "exec_seed": exec_seed,
        "plan": plan.model_dump(exclude_none=True), "success": False, "attempts_used": 0, "attempts": [],
    }
    feedback = image_path = None
    for attempt in range(1, MAX_ATTEMPTS[condition] + 1):
        code = generate_fn(task=task, plan=plan, attempt=attempt, feedback=feedback, image_path=image_path)
        result = run_code(code, task.id, exec_seed, out / f"attempt_{attempt}", timeout_s=timeout_s, record=record)
        gt_plan = task.ground_truth_plan(result["initial_state"]) if result["initial_state"] else plan
        chk = checker.check(task, result["final_state"], result["initial_state"], gt_plan)
        report = build_failure_report(result, chk)
        log["attempts"].append({
            "attempt": attempt,
            "feedback_given": feedback,
            "image_given": image_path,
            "code": code,
            "report": report,
            "outcome": {
                "success": chk["success"], "ran_ok": result["ok"], "exception": result["exception"],
                "rejected": result["rejected"], "timed_out": result["timed_out"],
                "sandbox_error": result["sandbox_error"],
                "per_goal": chk["per_goal"], "unnamed_ok": chk["unnamed_ok"], "unmoved": chk["unmoved"],
            },
            "call_log": result["call_log"],
            "initial_state": result["initial_state"],
            "final_state": result["final_state"],
            "video": result["video"],
            "frames": result["frames"],
        })
        log["attempts_used"] = attempt
        log["success"] = chk["success"]
        _write(out / "run.json", log)
        if chk["success"]:
            break
        if condition in ("B", "C"):
            feedback = report
        if condition == "C":
            image_path = (result["frames"] or {}).get(DEMO_CAMERA)
    log["log_path"] = str(out / "run.json")
    return log


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)
