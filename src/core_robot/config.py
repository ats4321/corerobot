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
