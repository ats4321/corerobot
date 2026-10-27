"""Hand-written baseline: put red_block into red_bin for 10 seeds, judged from scene_state().

Run:  .venv/bin/python scripts/handwritten_demo.py [--video-seed 0] [--gui]
Success = block center inside the red bin's inner footprint and below its rim.
Writes data/runs/handwritten_demo_seed<N>.mp4 (overhead camera) for one seed.
"""

from __future__ import annotations

import argparse
import sys

from core_robot.config import RUNS_DIR
from core_robot.sim.env import TabletopEnv, in_bin
from core_robot.sim.primitives import Robot
from core_robot.sim.render import Recorder


def run(robot: Robot):
    robot.pick("red_block")
    robot.place("red_block", "red_bin")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--video-seed", type=int, default=0)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    env = TabletopEnv(gui=args.gui)
    wins = 0
    for seed in range(args.seeds):
        env.reset(seed)
        robot = Robot(env)
        if seed == args.video_seed:
            video = RUNS_DIR / f"handwritten_demo_seed{seed}.mp4"
            with Recorder(env, video) as rec:
                run(robot)
            print(f"  video: {video} ({rec.frames} frames)")
        else:
            run(robot)
        ok = in_bin(env.scene_state(), "red_block", "red_bin")
        wins += ok
        calls = ", ".join(f"{c['primitive']}={'ok' if c['ok'] else c['reason']}" for c in robot.call_log)
        print(f"seed {seed}: {'SUCCESS' if ok else 'FAIL'}  ({calls})")
    env.close()
    print(f"\n{wins}/{args.seeds} succeeded")
    return 0 if wins >= 0.9 * args.seeds else 1


if __name__ == "__main__":
    sys.exit(main())
