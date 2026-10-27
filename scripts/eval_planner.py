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

    s = evaluate_planner(MetaClient(settings=settings), ids)
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
