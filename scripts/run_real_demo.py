"""Run the full pipeline on one real demonstration clip (live API; cached).

    .venv/bin/python scripts/run_real_demo.py data/real/clip.mp4 --dry-run
    .venv/bin/python scripts/run_real_demo.py data/real/clip.mp4 [--budget 0.5] [--seed 300]

Normalize -> SAM perception -> plan (frozen simulator prompts) -> a simulator scene with exactly
the plan's objects -> generated code -> sandbox, with condition C repair (up to 3 attempts).
Success is judged against the plan's own goal atoms. Outputs: data/runs/real/<clip>/.
The budget guard reserves every call's worst-case cost, so --budget is a hard cap.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from core_robot.config import load_settings
from core_robot.real.pipeline import REAL_SEED, estimate_cost


def print_estimate(clip: Path) -> None:
    from core_robot.real.normalize import normalize_clip

    with tempfile.TemporaryDirectory() as tmp:
        norm = normalize_clip(clip, Path(tmp) / "n.mp4")
    e = estimate_cost(norm["duration_s"])
    print(f"clip {clip.name}: {norm['source_duration_s']} s"
          f"{' (trimmed to 20 s)' if norm['trimmed'] else ''}, normalized {norm['size'][0]}x{norm['size'][1]}, "
          f"{norm['duration_s']} s; SAM bills {e['frames_billed_per_sam_video']} frames per video request")
    for k in ("naming", "sam", "planner", "codegen"):
        print(f"  {k:8s} expected ${e['expected'][k]:.3f}   worst ${e['worst'][k]:.3f}")
    print(f"  total    expected ${e['expected_total']:.2f}   worst ${e['worst_total']:.2f}")
    print(f"  ({e['assumptions']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip")
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=REAL_SEED)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    clip = Path(args.clip)
    if not clip.is_file():
        print(f"no such clip: {clip}")
        return 2
    print_estimate(clip)
    if args.dry_run:
        return 0

    from core_robot.eval.run_eval import check_prompt_freeze
    from core_robot.meta_client import BudgetGuard, MetaClient
    from core_robot.real.pipeline import run_pipeline

    check_prompt_freeze(allow_change=False)  # real clips use the frozen prompts, unchanged
    settings = load_settings()
    if not settings.api_key and not settings.replay:
        print("MODEL_API_KEY is not set (or use CORE_ROBOT_REPLAY=1)")
        return 2
    guard = BudgetGuard(args.budget)

    def show(name, payload):
        if name == "normalized":
            print(f"\n[normalized] {payload['path']} ({payload['size'][0]}x{payload['size'][1]}, {payload['duration_s']} s)")
        elif name == "perception":
            print(f"\n[perception] named: {payload['named']}  phrases: {payload['phrases']}")
            print(f"  coverage: {payload['coverage']}\n  overlay: {payload['overlay']}")
        elif name == "plan":
            print("\n[plan]", json.dumps(payload["plan"], indent=1) if payload["plan"] else f"none: {payload['error']}")
        elif name == "attempt":
            o = payload.get("outcome", {})
            print(f"\n[attempt {payload['attempt']}] {'success' if o.get('success') else 'failed'}; "
                  f"code: {Path(payload.get('video') or '').parent / 'code.py'}; video: {payload.get('video')}")
            if payload.get("report") and not o.get("success"):
                print("  " + payload["report"].replace("\n", "\n  "))
        elif name == "outcome":
            print(f"\n[outcome] {'SUCCESS' if payload['success'] else 'FAILED'}"
                  f"{' after ' + str(payload.get('attempts')) + ' attempt(s)' if payload.get('attempts') else ''}")
            print(f"  {payload.get('judged_against') or payload.get('reason')}")
            for g in payload.get("goal", []):
                print(f"  - {g}")

    s = run_pipeline(clip, MetaClient(settings=settings, guard=guard), seed=args.seed, on_stage=show)
    print(f"\nspend this clip: ${guard.spent:.3f} (budget ${args.budget:.2f}); summary: "
          f"data/runs/real/{s['name']}/summary.json")
    return 0 if s.get("outcome", {}).get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
