"""Generate the demonstration set: all 8 tasks at seed 0, plus t2 and t5 at seed 1.

Run:  .venv/bin/python scripts/make_demos.py
Writes data/demos/<task_id>/seed_<n>/ and prints a pass/fail line per demo.
"""

from __future__ import annotations

import sys

from core_robot.demos.demonstrator import demo_dir, run_demo
from core_robot.demos.tasks import TASKS, TASKS_BY_ID
from core_robot.sim.env import TabletopEnv

DEMO_SET = [(t.id, 0) for t in TASKS] + [("t2_sort_by_color", 1), ("t5_line_up", 1)]


def main() -> int:
    env = TabletopEnv()
    passed = 0
    for task_id, seed in DEMO_SET:
        out = demo_dir(task_id, seed)
        r = run_demo(TASKS_BY_ID[task_id], seed, env, out)
        passed += r["success"]
        atoms = "; ".join(f"{g['atom']}={'ok' if g['ok'] else 'NO'}" for g in r["check"]["goal"])
        still = [s["block"] for s in r["check"]["global_rule"] if not s["ok"]]
        print(
            f"{'PASS' if r['success'] else 'FAIL'}  {task_id} seed {seed} (layout {r['layout_seed']}): "
            f"{atoms}{'; moved: ' + ','.join(still) if still else ''}  [{r['video_frames']} frames -> {out}]"
        )
    env.close()
    print(f"\n{passed}/{len(DEMO_SET)} demos pass their goal check")
    return 0 if passed == len(DEMO_SET) else 1


if __name__ == "__main__":
    sys.exit(main())
