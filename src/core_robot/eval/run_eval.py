"""Evaluation matrix: tasks x exec_seeds() x conditions A/B/C, with an injected generator.

    python -m core_robot.eval.run_eval --generator fake [--workers 4] [--limit N] [--dry-run]

Each run goes through repair.loop.run_with_repair and leaves one run.json. The matrix is
resumable (finished runs are skipped), parallel (threads; the work happens in sandbox
subprocesses), and deterministic (results.csv is rebuilt from run logs in sorted order;
only wall_clock_s varies between identical runs). Fake-generator output goes only to
data/runs/fake/; live output to data/runs/eval/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from core_robot.config import (
    FAKE_RUNS_DIR, LIVE_RUNS_DIR, PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M, exec_seeds, load_settings,
)
from core_robot.demos.tasks import TASKS, TASKS_BY_ID
from core_robot.repair.loop import MAX_ATTEMPTS, run_with_repair
from core_robot.sim.env import TabletopEnv

CONDITIONS = ("A", "B", "C")
PLAN_SOURCES = ("ground_truth", "predicted")
CATEGORIES = ("wrong object", "wrong order", "grasp failure", "code error", "timeout", "harness error", "other")
CSV_FIELDS = [
    "task", "seed", "condition", "success", "attempts", "plan_source", "run_source", "failure_category",
    "wall_clock_s", "tokens_in", "tokens_out", "est_cost_usd",
]
ASSUMED_TOKENS_IN = 8000
ASSUMED_TOKENS_OUT = 1500


@dataclass(frozen=True, order=True)
class RunSpec:
    task_id: str
    seed: int
    condition: str


# ---- cost ----

def token_cost(tokens_in: int, tokens_out: int) -> float:
    """USD for one or more Muse Spark calls at standard-mode prices."""
    return tokens_in * PRICE_INPUT_PER_M / 1e6 + tokens_out * PRICE_OUTPUT_PER_M / 1e6


def estimate_codegen_cost(
    runs: list[RunSpec], tokens_in: int = ASSUMED_TOKENS_IN, tokens_out: int = ASSUMED_TOKENS_OUT
) -> dict:
    """Upper bound for code generation: every run uses its condition's maximum attempts."""
    generations = sum(MAX_ATTEMPTS[r.condition] for r in runs)
    per_gen = token_cost(tokens_in, tokens_out)
    return {
        "runs": len(runs), "generations": generations, "tokens_in_per_generation": tokens_in,
        "tokens_out_per_generation": tokens_out, "usd_per_generation": per_gen,
        "usd_total": generations * per_gen,
    }


# ---- matrix ----

def planned_runs(task_ids=None, seeds=None, conditions=CONDITIONS, limit: int | None = None) -> list[RunSpec]:
    task_ids = task_ids or [t.id for t in TASKS]
    seeds = seeds if seeds is not None else exec_seeds()
    runs = sorted(RunSpec(t, s, c) for t in task_ids for s in seeds for c in conditions)
    return runs[:limit] if limit else runs


def run_path(root: Path, spec: RunSpec) -> Path:
    return root / spec.task_id / spec.condition / f"seed_{spec.seed}"


def is_finished(root: Path, spec: RunSpec) -> bool:
    p = run_path(root, spec) / "run.json"
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text()).get("finished"))
    except json.JSONDecodeError:
        return False


def plan_for(task, seed: int, plan_source: str):
    if plan_source != "ground_truth":
        raise NotImplementedError(f"plan_source={plan_source!r} is not implemented yet")
    env = TabletopEnv()
    try:
        return task.ground_truth_plan(task.scene_builder(env, seed))
    finally:
        env.close()


def run_one(spec: RunSpec, generator, root: Path, plan_source: str, record: bool, timeout_s: float) -> str:
    """Execute one run unless it is already finished. Returns "done" or "skipped"."""
    if is_finished(root, spec):
        return "skipped"
    task = TASKS_BY_ID[spec.task_id]
    out = run_path(root, spec)
    t0 = time.monotonic()
    meta = {"plan_source": plan_source, "run_source": generator.run_source, "generator": generator.name}
    log = run_with_repair(task, plan_for(task, spec.seed, plan_source), spec.seed, spec.condition, generator,
                          out_dir=out, timeout_s=timeout_s, record=record, meta=meta)
    log["wall_clock_s"] = round(time.monotonic() - t0, 2)
    tmp = out / "run.json.tmp"
    tmp.write_text(json.dumps(log, indent=2, default=str))
    tmp.replace(out / "run.json")
    return "done"


def run_matrix(runs, generator, root: Path, plan_source="ground_truth", workers=4, record=True,
               timeout_s=60.0) -> Counter:
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(lambda s: run_one(s, generator, root, plan_source, record, timeout_s), runs))
    write_results_csv(root)
    return Counter(outcomes)


# ---- labelling ----

def failure_category(log: dict) -> str:
    """Label a failed run from its last attempt. Returns "" for a success.

    Precedence: harness error > timeout > code error > grasp failure > wrong object >
    wrong order > other.
      harness error  generator/API/replay failure, or the sandbox itself failed
      timeout        the run exceeded the time limit
      code error     rejected by static checks, or the code raised an exception
      grasp failure  a pick failed with "grasp missed" or "unreachable"
      wrong object   picked a block the ground-truth plan does not move, disturbed a block
                     that must stay, or never picked a block the ground-truth plan moves
      wrong order    picked exactly the ground-truth plan's objects, in a different order
    """
    if log.get("success"):
        return ""
    if not log.get("attempts"):
        return "harness error"
    last = log["attempts"][-1]
    out = last.get("outcome", {})
    if last.get("generator_error") or out.get("generator_error") or out.get("sandbox_error"):
        return "harness error"
    if out.get("timed_out"):
        return "timeout"
    if out.get("rejected") or out.get("exception"):
        return "code error"
    calls = last.get("call_log") or []
    if any(c["primitive"] == "pick" and not c["ok"] and
           (c["reason"].startswith("grasp missed") or c["reason"] == "unreachable") for c in calls):
        return "grasp failure"
    gt_objects = _ground_truth_objects(log, last)
    picked = [c["args"]["obj"] for c in calls if c["primitive"] == "pick" and c["ok"]]
    disturbed = any(not u["ok"] for u in out.get("unmoved", []))
    if disturbed or set(picked) - set(gt_objects) or set(gt_objects) - set(picked):
        return "wrong object"
    if sorted(picked) == sorted(gt_objects) and picked != gt_objects:
        return "wrong order"
    return "other"


def _ground_truth_objects(log: dict, attempt: dict) -> list[str]:
    task = TASKS_BY_ID[log["task_id"]]
    return [s.object for s in task.ground_truth_plan(attempt["initial_state"]).steps]


# ---- results.csv ----

def row_from_log(log: dict) -> dict:
    t_in, t_out = int(log.get("tokens_in", 0)), int(log.get("tokens_out", 0))
    return {
        "task": log["task_id"], "seed": log["exec_seed"], "condition": log["condition"],
        "success": int(bool(log["success"])), "attempts": log["attempts_used"],
        "plan_source": log.get("plan_source", ""), "run_source": log.get("run_source", ""),
        "failure_category": failure_category(log), "wall_clock_s": log.get("wall_clock_s", ""),
        "tokens_in": t_in, "tokens_out": t_out, "est_cost_usd": round(token_cost(t_in, t_out), 6),
    }


def write_results_csv(root: Path) -> Path:
    rows = []
    for p in sorted(root.glob("*/*/seed_*/run.json")):
        log = json.loads(p.read_text())
        if log.get("finished"):
            rows.append(row_from_log(log))
    rows.sort(key=lambda r: (r["task"], r["condition"], int(r["seed"])))
    path = root / "results.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


# ---- CLI ----

def output_root(generator: str, plan_source: str, run_name: str | None = None) -> tuple[Path, str]:
    """(root directory, run_source). Fake output only ever goes under data/runs/fake/."""
    run_source = "fake" if generator == "fake" else "live"
    base = FAKE_RUNS_DIR if run_source == "fake" else LIVE_RUNS_DIR
    return base / (run_name or f"{generator}_{plan_source}"), run_source


def make_generator(name: str, settings):
    if name == "fake":
        from core_robot.eval.fake_generator import FakeGenerator
        return FakeGenerator()
    if name == "muse":
        if not settings.api_key and not settings.replay:
            raise SystemExit("MODEL_API_KEY is not set: add it to .env, or set CORE_ROBOT_REPLAY=1 to use the cache only")
        from core_robot.codegen.muse_generator import MuseGenerator
        from core_robot.meta_client import MetaClient
        return MuseGenerator(MetaClient(settings=settings))
    raise SystemExit(f"unknown --generator {name!r}")


def resolve_tasks(spec: str | None) -> list[str] | None:
    """"t4,t7" or full ids -> task ids. None means all tasks."""
    if not spec:
        return None
    ids = []
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        matches = [t.id for t in TASKS if t.id == part or t.id.split("_", 1)[0] == part]
        if len(matches) != 1:
            raise SystemExit(f"--tasks: {part!r} matches {matches or 'no task'}; use e.g. t4 or t4_stack_three")
        ids.append(matches[0])
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--generator", default="fake", choices=["fake", "muse"])
    ap.add_argument("--plan-source", default="ground_truth", choices=PLAN_SOURCES)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tasks", default=None, help="comma-separated task ids or prefixes, e.g. t4,t7")
    ap.add_argument("--seeds", default=None, help="comma-separated execution seeds (default: exec_seeds())")
    ap.add_argument("--conditions", default="A,B,C", help="comma-separated subset of A,B,C")
    ap.add_argument("--limit", type=int, default=None, help="only the first N runs (sorted order)")
    ap.add_argument("--dry-run", action="store_true", help="print planned runs and estimated cost; run nothing")
    ap.add_argument("--run-name", default=None, help="subdirectory name (default: <generator>_<plan_source>)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args(argv)

    settings = load_settings()
    replay = settings.replay
    root, run_source = output_root(args.generator, args.plan_source, args.run_name)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    if not set(conditions) <= set(CONDITIONS):
        raise SystemExit(f"--conditions must be a subset of {CONDITIONS}")
    runs = planned_runs(resolve_tasks(args.tasks), seeds, conditions, limit=args.limit)
    todo = [r for r in runs if not is_finished(root, r)]

    if args.dry_run:
        est = estimate_codegen_cost(todo)
        print(f"output: {root}  (run_source={run_source}, replay={'on' if replay else 'off'})")
        for r in todo:
            print(f"  {r.task_id:24s} seed {r.seed}  condition {r.condition}  (max {MAX_ATTEMPTS[r.condition]} attempts)")
        print(f"{len(todo)} runs to do ({len(runs) - len(todo)} already finished)")
        print(
            f"code generation upper bound: {est['generations']} generations x "
            f"({est['tokens_in_per_generation']} in + {est['tokens_out_per_generation']} out tokens) = "
            f"{est['generations']} x ${est['usd_per_generation']:.6f} = ${est['usd_total']:.2f}"
            + ("  [fake generator: actual cost $0]" if run_source == "fake" else "")
        )
        return 0

    if args.plan_source == "predicted":
        raise SystemExit("--plan-source predicted is not implemented yet")
    generator = make_generator(args.generator, settings)
    t0 = time.monotonic()
    counts = run_matrix(todo, generator, root, args.plan_source, args.workers, not args.no_video, args.timeout)
    from core_robot.eval.report import write_report
    md, png = write_report(root)
    print(f"{counts.get('done', 0)} runs executed, {len(runs) - len(todo)} skipped, "
          f"{time.monotonic() - t0:.0f} s -> {root / 'results.csv'}, {md}, {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
