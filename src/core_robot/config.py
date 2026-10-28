"""Project-wide settings: model IDs, API endpoint, credentials, and data paths.

The API key is read from the MODEL_API_KEY environment variable, which may be
supplied through a `.env` file at the repo root. The key is never printed: it is
excluded from the Settings repr.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
DEMOS_DIR = DATA_DIR / "demos"
CACHE_DIR = DATA_DIR / "cache"
RUNS_DIR = DATA_DIR / "runs"

MUSE_MODEL = "muse-spark-1.3"
SAM_MODEL = "sam-3.1"
BASE_URL = "https://api.meta.ai/v1"

FAKE_RUNS_DIR = RUNS_DIR / "fake"  # anything produced by the fake generator lives here only
LIVE_RUNS_DIR = RUNS_DIR / "eval"

# Muse Spark 1.3 standard-mode prices, USD per 1M tokens (dev.meta.ai/docs/pricing-rate-limits).
PRICE_INPUT_PER_M = 1.25
PRICE_OUTPUT_PER_M = 4.25
# SAM 3.1: per image, and per video frame with a 50-frame minimum per video request.
SAM_PRICE_PER_IMAGE = 2.50 / 1000
SAM_PRICE_PER_FRAME = 0.20 / 1000
SAM_MIN_FRAMES = 50

# Execution (evaluation) seeds are disjoint from demonstration seeds (0, 1, ...), so a
# robot is never evaluated on the exact layout it was shown.
EXEC_SEED_BASE = 100


# Pilot seeds for choosing the perturbation magnitude; never used for evaluation.
PILOT_SEEDS = [200, 201, 202, 203, 204]

# calibration_offset perturbation magnitude (cm), frozen after the pilot (docs/perturbation_pilot.md).
# None = not frozen yet: perturbed runs on evaluation seeds are refused.
CALIBRATION_OFFSET_CM: float | None = 4.0


def exec_seeds(n: int = 5) -> list[int]:
    """Seeds for evaluation layouts: EXEC_SEED_BASE .. EXEC_SEED_BASE + n - 1."""
    return list(range(EXEC_SEED_BASE, EXEC_SEED_BASE + n))


@dataclass(frozen=True)
class Settings:
    base_url: str = BASE_URL
    muse_model: str = MUSE_MODEL
    sam_model: str = SAM_MODEL
    api_key: str | None = field(default=None, repr=False)
    replay: bool = False
    cache_dir: Path = CACHE_DIR / "model"
    log_path: Path = RUNS_DIR / "model_calls.jsonl"
    timeout_s: float = 600.0
    max_attempts: int = 5


def load_settings() -> Settings:
    """Build Settings from the environment, loading `.env` at the repo root first."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    return Settings(
        api_key=os.environ.get("MODEL_API_KEY") or None,
        replay=os.environ.get("CORE_ROBOT_REPLAY", "") == "1",
    )
