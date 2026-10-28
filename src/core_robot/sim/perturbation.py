"""Opt-in execution perturbation: "calibration_offset" (off by default).

The gripper's release position is shifted by a constant offset in every place() of a run,
as if the arm were miscalibrated. The magnitude m is fixed (cm); the direction comes from
a dedicated random stream seeded by the execution seed, so it is reproducible per seed and
independent of the scene layout. The robot does not know: place() still reports success
and the position it believes it released at (the commanded point). The checker, failure
report, logs and videos use the true simulator state. Code-generation prompts and the API
reference never mention it.
"""

from __future__ import annotations

import math

import numpy as np

_STREAM = 0xCA1B  # separates this stream from the layout RNG


def calibration_offset(exec_seed: int, magnitude_cm: float) -> tuple[float, float]:
    """(dx, dy) in meters: magnitude_cm in a seeded direction."""
    theta = float(np.random.default_rng([int(exec_seed), _STREAM]).uniform(0.0, 2.0 * math.pi))
    m = magnitude_cm / 100.0
    return m * math.cos(theta), m * math.sin(theta)
