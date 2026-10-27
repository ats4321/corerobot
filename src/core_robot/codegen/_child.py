"""Sandbox child process: build the scene, run generated code, write result.json.

Invoked only by codegen.sandbox.run_code as `python -m core_robot.codegen._child job.json`.
"""

from __future__ import annotations

import builtins
import copy
import io
import json
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import imageio.v2 as imageio

from core_robot.codegen.sandbox import SAFE_BUILTINS
from core_robot.codegen.scene import build_exec_scene, read_only_scene
from core_robot.demos.tasks import TASKS_BY_ID
from core_robot.sim.env import DEMO_CAMERA, SIM_HZ
from core_robot.sim.primitives import Robot
from core_robot.sim.render import Recorder

FRAME_CAMERAS = (DEMO_CAMERA, "side")


def make_robot_api(robot: Robot, calls_path: Path):
    """Object exposing only the four primitives; results are copies, calls are streamed to disk."""

    def logged(fn):
        def call(*args, **kwargs):
            result = fn(*args, **kwargs)
            with open(calls_path, "a") as f:
                f.write(json.dumps(robot.call_log[-1], default=str) + "\n")
            return copy.deepcopy(result)
        return call

    methods = {
        "pick": logged(robot.pick),
        "place": logged(robot.place),
        "move_above": logged(robot.move_above),
        "open_gripper": logged(robot.open_gripper),
    }
    api_cls = type("Robot", (), {"__slots__": (), **{k: staticmethod(v) for k, v in methods.items()},
                                 "__repr__": lambda self: "<Robot: pick, place, move_above, open_gripper>"})
    return api_cls()


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    out = Path(job["out_dir"])
    env, robot, initial = build_exec_scene(TASKS_BY_ID[job["task_id"]], job["exec_seed"])

    safe = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    namespace = {"__builtins__": safe, "__name__": "generated"}
    stdout = io.StringIO()
    exception = None
    video = out / "video.mp4" if job["record"] else None

    def execute():
        exec(compile(job["code"], "<generated>", "exec"), namespace)
        namespace["run"](make_robot_api(robot, out / "calls.jsonl"), read_only_scene(initial))

    try:
        with redirect_stdout(stdout):
            if video:
                with Recorder(env, video):
                    try:
                        execute()
                    finally:
                        env.step(int(0.5 * SIM_HZ))
            else:
                execute()
    except BaseException as e:  # report everything the generated code raises
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<generated>"]
        where = f" (line {frames[-1].lineno}: {frames[-1].line or ''})" if frames else ""
        exception = f"{type(e).__name__}: {e}{where}"

    final = env.scene_state()
    frame_paths = {}
    for cam in FRAME_CAMERAS:
        path = out / f"final_{cam}.png"
        imageio.imwrite(path, env.render(cam))
        frame_paths[cam] = str(path)
    result = {
        "ok": exception is None,
        "exception": exception,
        "call_log": robot.call_log,
        "initial_state": initial,
        "final_state": final,
        "video": str(video) if video else None,
        "frames": frame_paths,
        "stdout": stdout.getvalue()[-5000:],
    }
    (out / "result.json").write_text(json.dumps(result, default=str))
    env.close()


if __name__ == "__main__":
    main(sys.argv[1])
