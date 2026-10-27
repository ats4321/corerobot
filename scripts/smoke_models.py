"""Smoke-test the Meta Model API: list models, Muse Spark on image and video, SAM 3.1 on image and video.

Run:  .venv/bin/python scripts/smoke_models.py
Each check prints OK or FAIL with the reason, plus the request shape that was used.
For video, inline base64 (input_video.video_url) is tried first, then Files API upload (file_id).
Responses are cached in data/cache/model/, so reruns are free. Never prints the API key.
"""

from __future__ import annotations

import sys

from core_robot.config import SAMPLES_DIR, load_settings
from core_robot.meta_client import ApiError, MetaClient

IMAGE = SAMPLES_DIR / "table_objects.png"
CLIP = SAMPLES_DIR / "red_block_move.mp4"
SHORT_CLIP = SAMPLES_DIR / "red_block_move_3s.mp4"

results: list[tuple[str, bool, str]] = []


def check(name: str, shape: str, fn):
    try:
        detail = fn()
        results.append((name, True, shape))
        print(f"OK    {name}  [{shape}]\n      {detail}")
        return True
    except ApiError as e:
        msg = f"HTTP {e.status}: {e.message[:400]}"
    except Exception as e:  # report anything else verbatim
        msg = f"{type(e).__name__}: {str(e)[:400]}"
    results.append((name, False, f"{shape} -> {msg}"))
    print(f"FAIL  {name}  [{shape}]\n      {msg}")
    return False


def sam_summary(r: dict) -> str:
    frames = r["frames"]
    if not frames:
        return "0 matches (empty output_text)"
    first = min(frames)
    objs = frames[first]
    boxes = {oid: o["box"] for oid, o in objs.items()}
    return f"{len(frames)} frame(s) with objects; frame {first}: {len(objs)} object(s) boxes={boxes}"


def main() -> int:
    settings = load_settings()
    if not settings.api_key and not settings.replay:
        print("FAIL  MODEL_API_KEY is not set (add it to .env at the repo root)")
        return 1
    missing = [p for p in (IMAGE, CLIP, SHORT_CLIP) if not p.exists()]
    if missing:
        print(f"FAIL  missing samples {missing}; run scripts/make_placeholder_samples.py")
        return 1
    c = MetaClient(settings=settings)

    def models():
        ids = c.list_models()
        for want in (settings.muse_model, settings.sam_model):
            if want not in ids:
                raise RuntimeError(f"{want} not in model list {ids}")
        return f"{len(ids)} models: {ids}"

    check("list models", "GET /v1/models", models)

    check(
        "muse image",
        "POST /v1/responses muse-spark-1.3 [input_text, input_image(image_url=data:image/png;base64)]",
        lambda: c.muse("Describe this image in one or two sentences. List the objects and colors.", images=[IMAGE])["text"][:300],
    )

    video_prompt = "Describe what happens in this 10-second video in two sentences."
    ok = check(
        "muse video",
        "POST /v1/responses muse-spark-1.3 [input_text, input_video(video_url=data:video/mp4;base64)]",
        lambda: c.muse(video_prompt, video=CLIP)["text"][:300],
    )
    if not ok:
        check(
            "muse video (file fallback)",
            "POST /v1/files purpose=user_data, then /v1/responses [input_text, input_video(file_id)]",
            lambda: c.muse(video_prompt, video=CLIP, video_mode="file")["text"][:300],
        )

    check(
        "sam image",
        'POST /v1/responses sam-3.1 [input_text("red block"), input_image(image_url=data:image/png;base64)]',
        lambda: sam_summary(c.sam_image(IMAGE, "red block")),
    )

    ok = check(
        "sam video",
        'POST /v1/responses sam-3.1 [input_text("red block"), input_video(video_url=data:video/mp4;base64)] 3 s clip',
        lambda: sam_summary(c.sam_video(SHORT_CLIP, "red block")),
    )
    if not ok:
        check(
            "sam video (file fallback)",
            'POST /v1/files, then /v1/responses sam-3.1 [input_text("red block"), input_video(file_id)]',
            lambda: sam_summary(c.sam_video(SHORT_CLIP, "red block", video_mode="file")),
        )

    print("\nSUMMARY")
    for name, ok, shape in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}: {shape}")
    required = ["list models", "muse image", "muse video", "sam image", "sam video"]
    passed = {n.split(" (")[0] for n, ok, _ in results if ok}
    return 0 if all(r in passed for r in required) else 1


if __name__ == "__main__":
    sys.exit(main())
