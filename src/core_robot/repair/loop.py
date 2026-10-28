"""Generate -> run in sandbox -> check -> (on failure) feed back -> retry.

Conditions:
  A  one attempt, no feedback
  B  up to 3 attempts; the text failure report is passed back via generate_fn(feedback=...)
  C  as B, plus the path of the final angled-camera frame via generate_fn(image_path=...)

Every attempt runs in a fresh sandbox process that resets the simulator to the same
execution seed, so attempts never inherit the previous attempt's end state. Success is
judged against the task's ground-truth plan for the execution layout, not the plan given
to the generator. One JSON log per run records each attempt's code, report and outcome;
"finished" is set only once the run is complete (used to resume interrupted evaluations).
An exception raised by generate_fn (API error, replay miss) ends the run as a harness error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from core_robot.codegen.sandbox import DEFAULT_TIMEOUT_S, run_code
from core_robot.codegen.scene import exec_initial_state, read_only_scene
from core_robot.config import RUNS_DIR
from core_robot.planning.schema import Plan
from core_robot.sim.env import DEMO_CAMERA
from core_robot.verify import checker
from core_robot.verify.report import build_failure_report

MAX_ATTEMPTS = {"A": 1, "B": 3, "C": 3}


class GenerateFn(Protocol):
    """Returns the code as a string, or {"code": str, "tokens_in": int, "tokens_out": int}."""

    def __call__(self, *, task, plan: Plan, attempt: int, scene: dict, feedback: str | None = None,
                 image_path: str | None = None) -> str | dict: ...


def _normalize(generated) -> tuple[str, int, int, dict]:
    """(code, tokens_in, tokens_out, extra fields such as request_key) from a generator's return."""
    if isinstance(generated, str):
        return generated, 0, 0, {}
    extra = {k: generated[k] for k in ("request_key", "status", "incomplete_reason") if k in generated}
    return generated["code"], int(generated.get("tokens_in", 0)), int(generated.get("tokens_out", 0)), extra


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
    meta: dict | None = None,
    calibration_offset_cm: float | None = None,
    on_attempt=None,
) -> dict:
    """Run one (task, exec_seed, condition). Stops at the first success. Returns the run log.

    meta: extra fields (e.g. run_source, plan_source) stored in the log from the first write.
    on_attempt: optional callback, called with each finished attempt's record.
    """
    if condition not in MAX_ATTEMPTS:
        raise ValueError(f"condition must be one of {sorted(MAX_ATTEMPTS)}, got {condition!r}")
    out = Path(out_dir) if out_dir else run_dir(task.id, condition, exec_seed)
    out.mkdir(parents=True, exist_ok=True)
    log = {
        "task_id": task.id, "condition": condition, "exec_seed": exec_seed,
        "plan": plan.model_dump(exclude_none=True), "success": False, "attempts_used": 0, "attempts": [],
        "tokens_in": 0, "tokens_out": 0, "finished": False, **(meta or {}),
    }
    feedback = image_path = None
    # The same read-only starting scene that run(robot, scene) receives in the sandbox.
    scene = read_only_scene(exec_initial_state(task, exec_seed))
    for attempt in range(1, MAX_ATTEMPTS[condition] + 1):
        try:
            code, t_in, t_out, gen_info = _normalize(
                generate_fn(task=task, plan=plan, attempt=attempt, scene=scene, feedback=feedback,
                            image_path=image_path)
            )
        except Exception as e:  # generator/API/replay failure: a harness error, not the model's code
            log["attempts"].append({
                "attempt": attempt, "feedback_given": feedback, "image_given": image_path, "code": None,
                "report": None, "generator_error": f"{type(e).__name__}: {e}",
                "outcome": {"success": False, "generator_error": True},
            })
            log["attempts_used"] = attempt
            break
        log["tokens_in"] += t_in
        log["tokens_out"] += t_out
        result = run_code(code, task.id, exec_seed, out / f"attempt_{attempt}", timeout_s=timeout_s, record=record,
                          calibration_offset_cm=calibration_offset_cm,
                          task_spec=task.spec() if hasattr(task, "spec") else None)
        gt_plan = task.ground_truth_plan(result["initial_state"]) if result["initial_state"] else plan
        chk = checker.check(task, result["final_state"], result["initial_state"], gt_plan)
        report = build_failure_report(result, chk)
        log["attempts"].append({
            "attempt": attempt,
            "feedback_given": feedback,
            "image_given": image_path,
            "code": code,
            "tokens_in": t_in,
            "tokens_out": t_out,
            "generation": gen_info,
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
        if on_attempt is not None:  # e.g. the UI showing each attempt as it finishes
            on_attempt(log["attempts"][-1])
        if chk["success"]:
            break
        if condition in ("B", "C"):
            feedback = report
        if condition == "C":
            image_path = (result["frames"] or {}).get(DEMO_CAMERA)
    log["finished"] = True
    log["log_path"] = str(out / "run.json")
    _write(out / "run.json", log)
    return log


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)
