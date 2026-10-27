"""Actual API spend from the call log (cache hits are free).

Run:  .venv/bin/python scripts/spend.py [--since UNIX_TS] [--budget USD]
Exits 3 if --budget is given and spend since --since exceeds it.
"""

from __future__ import annotations

import argparse
import sys

from core_robot.config import load_settings
from core_robot.perception.costs import actual_spend


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=0.0)
    ap.add_argument("--budget", type=float, default=None)
    args = ap.parse_args()
    s = actual_spend(load_settings().log_path, args.since)
    est = f" ({s['sam_frames_estimated']} SAM videos without a frame count, counted at the minimum)" \
        if s.get("sam_frames_estimated") else ""
    print(f"spend: ${s['usd']:.3f}  [Muse {s['muse_calls']} calls, {s['tokens_in']} in / {s['tokens_out']} out tokens;"
          f" SAM {s['sam_calls']} calls, {s['sam_frames']} frames, {s['sam_images']} images{est};"
          f" {s['cache_hits']} cache hits; {s['errors']} failed calls]")
    if args.budget is not None and s["usd"] > args.budget:
        print(f"BUDGET EXCEEDED: ${s['usd']:.2f} > ${args.budget:.2f}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
