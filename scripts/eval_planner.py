"""Planner evaluation on the demonstration set (live API; cached).

Run:  .venv/bin/python scripts/eval_planner.py [--demos t4_stack_three/seed_0,...] [--dry-run]
For each demo: perception (object naming + SAM tracks), video -> plan with Muse Spark, then
scoring against the ground-truth plan (strict for the gate, lenient beside it).
Gate: strict match on at least 6 of 10 demos.
"""

from __future__ import annotations

import argparse
import sys

from core_robot.config import load_settings
from core_robot.eval.demo_eval import demo_ids, demo_meta, evaluate_planner
from core_robot.perception.costs import estimate_perception_planning

GATE = 6


def print_estimate(ids):
    from core_robot.eval.demo_eval import PERCEPTION_DIR
    from core_robot.eval.run_eval import measured_codegen_tokens
    from core_robot.perception.costs import measured_planner_tokens, muse_cost
    from core_robot.planning.video_planner import FIRST_MAX_OUTPUT_TOKENS, RETRY_MAX_OUTPUT_TOKENS
    from core_robot.config import load_settings

    cached = [d for d in ids if (PERCEPTION_DIR / d / "perception.json").exists()]
    if len(cached) == len(ids):
        m = measured_planner_tokens(load_settings().log_path)
        t_in, t_out = (m[0], m[1]) if m else (20000, 3000)
        per = muse_cost(t_in, t_out)
        worst = muse_cost(t_in, FIRST_MAX_OUTPUT_TOKENS) + muse_cost(t_in, RETRY_MAX_OUTPUT_TOKENS)
        print(f"perception: cached for all {len(ids)} demos ($0)")
        print(f"planner: {len(ids)} calls x ${per:.4f} (measured avg {t_in:.0f} in + {t_out:.0f} out tokens"
              f"{'' if m else ', assumed'}) = ${len(ids) * per:.2f} expected")
        print(f"planner worst case (every call truncated at {FIRST_MAX_OUTPUT_TOKENS}, retried at "
              f"{RETRY_MAX_OUTPUT_TOKENS}): {len(ids)} x ${worst:.4f} = ${len(ids) * worst:.2f}")
        return
    e = estimate_perception_planning([demo_meta(d) for d in ids])
    print(f"perception + planner estimate for {e['demos']} demos:")
    print(f"  SAM video: {e['sam_video_frames_billed']} frames x $0.0002 = ${e['sam_video_usd']:.2f}")
    print(f"  SAM keyframe fallback (worst case, every object): {e['sam_fallback_images_worst']} images x $0.0025"
          f" = ${e['sam_fallback_usd_worst']:.2f}")
    print(f"  object naming: {e['naming_calls']} calls ~ ${e['naming_usd']:.2f} (assumes 3000 in + 800 out tokens)")
    print(f"  planner: {e['planner_calls'][0]}-{e['planner_calls'][1]} calls ~ ${e['planner_usd'][0]:.2f}-"
          f"${e['planner_usd'][1]:.2f} (assumes 20000 in + 3000 out tokens per video call; unmeasured)")
    print(f"  total: expected ~${e['expected_usd']:.2f}, worst case ${e['worst_usd']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default=None, help="comma-separated demo ids (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, help="experiment only: reasoning.effort for planner calls")
    ap.add_argument("--label", default=None, help="write results to data/runs/planner/experiments/<label>/")
    ap.add_argument("--budget", type=float, default=None,
                    help="USD: start no new demo once actual spend since this command started exceeds it")
    args = ap.parse_args()
    ids = args.demos.split(",") if args.demos else demo_ids()
    if args.dry_run:
        print_estimate(ids)
        return 0
    settings = load_settings()
    if not settings.api_key and not settings.replay:
        print("MODEL_API_KEY is not set (or use CORE_ROBOT_REPLAY=1)")
        return 2
    from core_robot.meta_client import MetaClient

    from core_robot.eval.demo_eval import PLANNER_DIR

    out = PLANNER_DIR / "experiments" / args.label if args.label else PLANNER_DIR
    if args.reasoning_effort and not args.label:
        print("--reasoning-effort is an experiment: give it a --label so it does not overwrite the main results")
        return 2

    from core_robot.meta_client import BudgetGuard

    guard = BudgetGuard(args.budget) if args.budget is not None else None
    over_budget = (lambda: guard.remaining() <= 0) if guard else (lambda: False)
    s = evaluate_planner(MetaClient(settings=settings, guard=guard), ids, out_root=out, reasoning_effort=args.reasoning_effort,
                         over_budget=over_budget)
    if len(s["rows"]) and over_budget():
        print(f"BUDGET: spend passed ${args.budget:.2f}; demos after that were not started")
    print(f"results: {out / 'results.json'}" + (f"  [EXPERIMENT {args.label}: reasoning_effort={args.reasoning_effort}]"
                                                  if args.label else ""))
    print(f"{'demo':32s} {'strict':6s} {'lenient':7s} errors")
    for r in s["rows"]:
        err = ", ".join(r["errors"]) or "-"
        if r["planner_error"]:
            err += f"  [planner: {r['planner_error'][:80]}]"
        print(f"{r['demo']:32s} {'yes' if r['strict'] else 'no':6s} {'yes' if r['lenient'] else 'no':7s} {err}")
    print(f"\nstrict {s['strict']}/{s['demos']}, lenient {s['lenient']}/{s['demos']}")
    print("most common error types:", ", ".join(f"{e} ({n})" for e, n in s["top_errors"]) or "none")
    ok = s["strict"] >= GATE
    print(f"GATE planner strict >= {GATE}/10: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
