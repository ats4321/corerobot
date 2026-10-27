"""Record a simulation run to mp4 from one camera: one frame per 1/10 s of simulated time."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio

from core_robot.sim.env import TabletopEnv

FRAME_PERIOD = 0.1  # seconds of simulated time per video frame


class Recorder:
    """Context manager that captures frames while the env steps.

    with Recorder(env, "run.mp4"):
        robot.pick("red_block")
        ...
    The video plays at real time (10 fps). `frames` counts frames written.
    """

    def __init__(self, env: TabletopEnv, path: str | Path, camera: str = "overhead"):
        self.env = env
        self.path = Path(path)
        self.camera = camera
        self.frames = 0
        self._next_t = 0.0
        self._writer = None

    def __enter__(self) -> "Recorder":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(
            self.path, fps=round(1 / FRAME_PERIOD), codec="libx264", pixelformat="yuv420p",
            macro_block_size=8, output_params=["-movflags", "+faststart"],
        )
        self._next_t = self.env.sim_time
        self._capture()
        self.env.step_hooks.append(self._on_step)
        return self

    def __exit__(self, *exc):
        self.env.step_hooks.remove(self._on_step)
        self._writer.close()

    def _on_step(self, env: TabletopEnv):
        if env.sim_time + 1e-9 >= self._next_t:
            self._capture()

    def _capture(self):
        self._writer.append_data(self.env.render(self.camera))
        self.frames += 1
        self._next_t += FRAME_PERIOD
