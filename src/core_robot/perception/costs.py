"""Cost estimate (dry run) and actual spend for perception, planning and code generation."""

from __future__ import annotations

import json
import math
from pathlib import Path

from core_robot.config import (
    PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M, SAM_MIN_FRAMES, SAM_PRICE_PER_FRAME, SAM_PRICE_PER_IMAGE,
)

# Token assumptions for calls whose real size is unknown until the first live run.
NAMING_TOKENS = (3000, 800)  # one image + vocabulary
PLANNER_TOKENS = (20000, 3000)  # a ~10-40 s clip + track summary; video token count is a guess
KEYFRAMES = 8


def muse_cost(t_in: int, t_out: int) -> float:
    return t_in * PRICE_INPUT_PER_M / 1e6 + t_out * PRICE_OUTPUT_PER_M / 1e6


def resampled_frames(n_source: int, step: int = 2) -> int:
    n = math.ceil(n_source / step)
    return n + (0 if (n_source - 1) % step == 0 else 1)


def estimate_perception_planning(demos: list[dict]) -> dict:
    """demos: [{"name", "n_frames" (source), "n_objects"}]. Returns expected and worst-case USD.

    expected: one SAM video request per object, one naming call, one planner call per demo.
    worst: every object also falls back to keyframes, and every planner call is retried.
    """
    sam_frames = sum(d["n_objects"] * max(SAM_MIN_FRAMES, resampled_frames(d["n_frames"])) for d in demos)
    sam_video = sam_frames * SAM_PRICE_PER_FRAME
    fallback_images = sum(d["n_objects"] for d in demos) * KEYFRAMES
    naming = len(demos) * muse_cost(*NAMING_TOKENS)
    planner_one = len(demos) * muse_cost(*PLANNER_TOKENS)
    return {
        "demos": len(demos), "sam_video_frames_billed": sam_frames, "sam_video_usd": sam_video,
        "sam_fallback_images_worst": fallback_images, "sam_fallback_usd_worst": fallback_images * SAM_PRICE_PER_IMAGE,
        "naming_calls": len(demos), "naming_usd": naming, "planner_calls": (len(demos), 2 * len(demos)),
        "planner_usd": (planner_one, 2 * planner_one),
        "expected_usd": sam_video + naming + planner_one,
        "worst_usd": sam_video + fallback_images * SAM_PRICE_PER_IMAGE + naming + 2 * planner_one,
    }


def actual_spend(log_path: str | Path, since: float = 0.0) -> dict:
    """Spend from the MetaClient call log (cache hits are free)."""
    out = {"muse_calls": 0, "tokens_in": 0, "tokens_out": 0, "sam_calls": 0, "sam_frames": 0, "sam_images": 0,
           "cache_hits": 0, "errors": 0}
    p = Path(log_path)
    if p.exists():
        for line in p.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("ts", 0) < since:
                continue
            if rec.get("cache_hit"):
                out["cache_hits"] += 1
                continue
            if rec.get("error"):
                out["errors"] += 1
                continue
            usage = rec.get("usage") or {}
            if rec.get("model", "").startswith("sam"):
                out["sam_calls"] += 1
                if rec.get("kind") == "video":
                    if "video_frames_processed" in usage:
                        out["sam_frames"] += int(usage["video_frames_processed"])
                    else:  # not reported: count the billing minimum and flag it
                        out["sam_frames"] += SAM_MIN_FRAMES
                        out["sam_frames_estimated"] = out.get("sam_frames_estimated", 0) + 1
                else:
                    out["sam_images"] += 1
            elif rec.get("model", "").startswith("muse"):
                out["muse_calls"] += 1
                out["tokens_in"] += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
                out["tokens_out"] += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    out["usd"] = (muse_cost(out["tokens_in"], out["tokens_out"]) + out["sam_frames"] * SAM_PRICE_PER_FRAME
                  + out["sam_images"] * SAM_PRICE_PER_IMAGE)
    return out
