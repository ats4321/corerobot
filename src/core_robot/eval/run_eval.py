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
from core_robot.planning.schema import Plan
from core_robot.sim.env import TabletopEnv

CONDITIONS = ("A", "B", "C")
PLAN_SOURCES = ("ground_truth", "predicted")
CATEGORIES = ("wrong object", "wrong order", "grasp failure", "code error", "timeout", "harness error", "other")
CSV_FIELDS = [
    "task", "seed", "condition", "environment", "success", "attempts", "plan_source", "plan_strict", "plan_lenient",
    "run_source", "failure_category", "wall_clock_s", "tokens_in", "tokens_out", "est_cost_usd",
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
    """A run is done when its log is finished and it did not end in a harness error
    (generator/API/replay failure or sandbox crash); those are redone on resume."""
    p = run_path(root, spec) / "run.json"
    if not p.exists():
        return False
    try:
        log = json.loads(p.read_text())
    except json.JSONDecodeError:
        return False
    return bool(log.get("finished")) and failure_category(log) != "harness error"


class NoPredictedPlan(RuntimeError):
    """The planner produced no valid plan for this task's demo (never replaced by ground truth)."""


PREDICTED_PLAN_DEMO_SEED = 0


def predicted_plan(task, planner_results: Path | None = None) -> Plan:
    """The cached predicted plan from the task's seed-0 demo (scripts/eval_planner.py)."""
    return predicted_plan_row(task, planner_results)[0]


def predicted_plan_row(task, planner_results: Path | None = None) -> tuple[Plan, dict]:
    """(plan, planner result row) for the task's seed-0 demo; the row carries its strict/lenient score."""
    from core_robot.eval.demo_eval import PLANNER_DIR

    path = planner_results or PLANNER_DIR / "results.json"
    demo = f"{task.id}/seed_{PREDICTED_PLAN_DEMO_SEED}"
    if not path.exists():
        raise NoPredictedPlan(f"no planner results at {path}; run scripts/eval_planner.py")
    rows = {r["demo"]: r for r in json.loads(path.read_text())["rows"]}
    row = rows.get(demo)
    if row is None:
        raise NoPredictedPlan(f"no planner result for demo {demo}")
    if not row.get("plan"):
        raise NoPredictedPlan(f"the planner produced no valid plan for demo {demo}: {row.get('planner_error')}")
    return Plan.model_validate(row["plan"]), row


def plan_for(task, seed: int, plan_source: str, planner_results: Path | None = None) -> Plan:
    if plan_source == "predicted":
        return predicted_plan(task, planner_results)
    env = TabletopEnv()
    try:
        return task.ground_truth_plan(task.scene_builder(env, seed))
    finally:
        env.close()


def run_one(spec: RunSpec, generator, root: Path, plan_source: str, record: bool, timeout_s: float,
            over_budget=lambda: False) -> str:
    """Execute one run unless it is already finished. Returns "done", "skipped" or "over_budget"."""
    if is_finished(root, spec):
        return "skipped"
    if over_budget():
        return "over_budget"
    task = TASKS_BY_ID[spec.task_id]
    out = run_path(root, spec)
    t0 = time.monotonic()
    meta = {"plan_source": plan_source, "run_source": generator.run_source, "generator": generator.name,
            "environment": "clean"}
    if plan_source == "predicted":
        try:
            _, prow = predicted_plan_row(task)
            meta.update(plan_demo=prow["demo"], plan_strict=int(bool(prow.get("strict"))),
                        plan_lenient=int(bool(prow.get("lenient"))))
        except NoPredictedPlan:
            pass
    try:
        plan = plan_for(task, spec.seed, plan_source)
    except NoPredictedPlan as e:
        out.mkdir(parents=True, exist_ok=True)
        log = {"task_id": task.id, "condition": spec.condition, "exec_seed": spec.seed, "plan": None,
               "success": False, "attempts_used": 0, "attempts": [], "tokens_in": 0, "tokens_out": 0,
               "finished": True, "plan_error": str(e), **meta, "wall_clock_s": 0.0}
        (out / "run.json").write_text(json.dumps(log, indent=2))
        return "done"
    log = run_with_repair(task, plan, spec.seed, spec.condition, generator,
                          out_dir=out, timeout_s=timeout_s, record=record, meta=meta)
    log["wall_clock_s"] = round(time.monotonic() - t0, 2)
    tmp = out / "run.json.tmp"
    tmp.write_text(json.dumps(log, indent=2, default=str))
    tmp.replace(out / "run.json")
    return "done"


def run_matrix(runs, generator, root: Path, plan_source="ground_truth", workers=4, record=True,
               timeout_s=60.0, over_budget=lambda: False) -> Counter:
    """Run (or resume) the given runs. over_budget() is checked before each run starts; once
    it returns True no new runs start (runs already started finish)."""
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(
            lambda s: run_one(s, generator, root, plan_source, record, timeout_s, over_budget), runs))
    write_results_csv(root)
    return Counter(outcomes)


def attempt_one_cache_status(runs, plan_source: str, settings) -> tuple[int, int]:
    """(runs whose attempt-1 generation is not cached yet, maximum repair generations for B/C)."""
    from core_robot.codegen.muse_generator import MuseGenerator
    from core_robot.codegen.scene import exec_initial_state, read_only_scene
    from core_robot.meta_client import MetaClient

    gen = MuseGenerator(MetaClient(settings=settings))
    new = 0
    for r in runs:
        task = TASKS_BY_ID[r.task_id]
        try:
            plan = plan_for(task, r.seed, plan_source)
        except NoPredictedPlan:
            continue
        key = gen.request_key(plan=plan, scene=read_only_scene(exec_initial_state(task, r.seed)))
        new += not gen.client.is_cached(key)
    repairs = sum(MAX_ATTEMPTS[r.condition] - 1 for r in runs)
    return new, repairs


def measured_codegen_tokens(log_path: Path) -> tuple[float, float, int] | None:
    """Average (in, out) tokens of past live code-generation calls (text-only Muse calls)."""
    if not Path(log_path).exists():
        return None
    calls = [json.loads(l) for l in Path(log_path).read_text().splitlines() if l.strip()]
    calls = [c for c in calls if c.get("model", "").startswith("muse") and c.get("kind") == "text"
             and not c.get("cache_hit") and c.get("usage")]
    if not calls:
        return None
    return (sum(c["usage"].get("input_tokens", 0) for c in calls) / len(calls),
            sum(c["usage"].get("output_tokens", 0) for c in calls) / len(calls), len(calls))


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
    if not log.get("attempts"):  # includes "no valid predicted plan" (log["plan_error"])
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
        "environment": log.get("environment", "clean"),
        "success": int(bool(log["success"])), "attempts": log["attempts_used"],
        "plan_source": log.get("plan_source", ""), "plan_strict": log.get("plan_strict", ""),
        "plan_lenient": log.get("plan_lenient", ""), "run_source": log.get("run_source", ""),
        "failure_category": failure_category(log), "wall_clock_s": log.get("wall_clock_s", ""),
        "tokens_in": t_in, "tokens_out": t_out, "est_cost_usd": round(token_cost(t_in, t_out), 6),
    }


def write_results_csv(root: Path) -> Path:
    rows = []
    for p in sorted(root.glob("*/*/seed_*/run.json")):
        log = json.loads(p.read_text())
        if log.get("finished"):
            rows.append(row_from_log(log))
    rows.sort(key=lambda r: (r["environment"], r["task"], r["condition"], int(r["seed"])))
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
    ap.add_argument("--budget", type=float, default=None,
                    help="USD: stop starting new runs once actual API spend since this command started exceeds it")
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
        measured = measured_codegen_tokens(settings.log_path)
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
        if run_source == "live":
            from core_robot.codegen.muse_generator import DEFAULT_MAX_OUTPUT_TOKENS

            if settings.api_key or settings.replay:
                new1, repairs = attempt_one_cache_status(todo, args.plan_source, settings)
                per = token_cost(*(measured[:2])) if measured else token_cost(ASSUMED_TOKENS_IN, ASSUMED_TOKENS_OUT)
                print(f"cache check: attempt 1 needs a new generation in {new1} of {len(todo)} runs "
                      f"(the rest are cached); conditions B/C add at most {repairs} repair generations")
                print(f"expected new spend: ({new1} + up to {repairs}) x ${per:.4f} = ${new1 * per:.2f} "
                      f"to ${(new1 + repairs) * per:.2f}")

            if measured:
                m_in, m_out, n = measured
                print(f"measured so far: {n} code-generation calls averaged {m_in:.0f} in + {m_out:.0f} out tokens "
                      f"-> {est['generations']} x ${token_cost(m_in, m_out):.4f} = "
                      f"${est['generations'] * token_cost(m_in, m_out):.2f}")
            worst = est["generations"] * token_cost(max(measured[0] if measured else 0, 3000), DEFAULT_MAX_OUTPUT_TOKENS)
            print(f"worst case (every call uses the {DEFAULT_MAX_OUTPUT_TOKENS}-token output cap): ${worst:.2f}"
                  + (f"; --budget {args.budget:.2f} stops new runs after that much spend" if args.budget else ""))
        return 0

    generator = make_generator(args.generator, settings)
    t0 = time.monotonic()
    started = time.time()
    over_budget = lambda: False
    if args.budget is not None:
        from core_robot.perception.costs import actual_spend

        over_budget = lambda: actual_spend(settings.log_path, started)["usd"] > args.budget
    counts = run_matrix(todo, generator, root, args.plan_source, args.workers, not args.no_video, args.timeout,
                        over_budget)
    if counts.get("over_budget"):
        print(f"BUDGET: {counts['over_budget']} runs not started (spend passed ${args.budget:.2f})")
    from core_robot.eval.report import write_report
    md, png = write_report(root)
    print(f"{counts.get('done', 0)} runs executed, {len(runs) - len(todo)} skipped, "
          f"{time.monotonic() - t0:.0f} s -> {root / 'results.csv'}, {md}, {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
