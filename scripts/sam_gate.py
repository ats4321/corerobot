"""SAM gate on the demonstration set: every block and bin found and tracked (live API; cached).

Run:  .venv/bin/python scripts/sam_gate.py [--demos ...]
Reuses perception from scripts/eval_planner.py when present (data/runs/perception/).
Gate: at least 8 of 10 demos pass (definition in core_robot.perception.gate).
"""

from __future__ import annotations

import argparse
import sys

from core_robot.config import load_settings
from core_robot.eval.demo_eval import demo_ids, evaluate_sam

GATE = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default=None)
    args = ap.parse_args()
    ids = args.demos.split(",") if args.demos else demo_ids()
    settings = load_settings()
    if not settings.api_key and not settings.replay:
        print("MODEL_API_KEY is not set (or use CORE_ROBOT_REPLAY=1)")
        return 2
    from core_robot.meta_client import MetaClient

    s = evaluate_sam(MetaClient(settings=settings), ids)
    for r in s["rows"]:
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['demo']}   overlay: {r['overlay']}")
        for name, o in r["objects"].items():
            tag = "" if o["required"] else " (optional)"
            print(f"      {name}{tag}: {'ok' if o['pass'] else o['reason']}  coverage {o['coverage']:.0%}"
                  f"  source {o.get('source')}")
    ok = s["passed"] >= GATE
    print(f"\n{s['passed']}/{s['demos']} demos pass")
    print(f"GATE SAM >= {GATE}/10: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
