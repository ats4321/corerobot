"""UI live path: run the real-clip pipeline on an uploaded video, stage by stage.

events() is a generator of (stage, payload) pairs the page renders as they arrive. It never
makes a live API call unless the user ticked the confirmation box; with "Replay from cache"
on it uses cached responses only (no spend, no confirmation needed). Spending is capped by the
budget guard (worst-case reservations), so the shown budget is a hard limit.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

DEFAULT_BUDGET = 0.5


def estimate_text(video_path: str | None) -> str:
    """Dry-run estimate for an uploaded clip (normalizes it locally; no API call)."""
    if not video_path:
        return "Upload a video to see the cost estimate."
    import tempfile

    from core_robot.real.normalize import normalize_clip
    from core_robot.real.pipeline import estimate_cost

    try:
        with tempfile.TemporaryDirectory() as tmp:
            norm = normalize_clip(video_path, Path(tmp) / "n.mp4")
    except Exception as e:
        return f"Could not read this video: {e}"
    e = estimate_cost(norm["duration_s"])
    return (f"Clip: {norm['source_duration_s']} s{' (trimmed to 20 s)' if norm['trimmed'] else ''}, normalized to "
            f"{norm['size'][0]}x{norm['size'][1]}. Estimated cost: **${e['expected_total']:.2f} expected, "
            f"${e['worst_total']:.2f} worst case** ({e['assumptions']}).")


def _default_client(replay: bool, budget: float):
    from dataclasses import replace

    from core_robot.config import load_settings
    from core_robot.meta_client import BudgetGuard, MetaClient

    settings = load_settings()
    if replay:
        return MetaClient(settings=replace(settings, replay=True))
    if not settings.api_key:
        raise RuntimeError("MODEL_API_KEY is not set (add it to .env)")
    return MetaClient(settings=settings, guard=BudgetGuard(budget))


def events(video_path: str | None, replay: bool, confirmed: bool, budget: float = DEFAULT_BUDGET,
           client_factory: Callable | None = None, out_root: Path | None = None) -> Iterator[tuple[str, dict]]:
    """Yield ("error", {...}) or the pipeline's stages in order, ending with "outcome" or "error"."""
    if not video_path:
        yield "error", {"message": "Upload a video first."}
        return
    if not replay and not confirmed:
        yield "error", {"message": f"Tick the confirmation box to allow live API spend (up to ${budget:.2f})."}
        return
    from core_robot.eval.run_eval import check_prompt_freeze
    from core_robot.real.pipeline import REAL_DIR, run_pipeline

    try:
        check_prompt_freeze(allow_change=False)
        client = (client_factory or _default_client)(replay, budget)
    except (SystemExit, RuntimeError) as e:
        yield "error", {"message": str(e)}
        return

    q: queue.Queue = queue.Queue()
    done = object()

    def work():
        try:
            run_pipeline(video_path, client, out_root=out_root or REAL_DIR,
                         on_stage=lambda name, payload: q.put((name, payload)))
        except Exception as e:  # shown on the page, never a crash
            q.put(("error", {"message": f"{type(e).__name__}: {e}"}))
        finally:
            q.put(done)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is done:
            break
        yield item
    guard = getattr(client, "guard", None)
    if guard is not None:
        yield "spend", {"spent": round(guard.spent, 4), "budget": guard.budget}


def run_events_with_timeout(gen: Iterator, timeout_s: float = 900) -> list:
    """Collect events (tests and scripts); stops after timeout_s."""
    out, t0 = [], time.monotonic()
    for ev in gen:
        out.append(ev)
        if time.monotonic() - t0 > timeout_s:
            break
    return out
