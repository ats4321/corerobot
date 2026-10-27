"""SAM 3.1 tracking of one named object per request, with a keyframe fallback.

segment_video sends one concept prompt (e.g. "red cube") with the whole (resampled) clip.
If that request fails, or the chosen object has a mask on fewer than MIN_COVERAGE of the
frames, the object is re-segmented on evenly spaced keyframes as single images, and the
track with the better coverage is kept.

A concept can match several objects (e.g. "red cube" might also catch a red bin), so one
SAM object id is selected per name: among ids seen on at least half as many frames as the
most-seen id, blocks take the smallest median box, bins the largest, and the mug the most
frames.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median

from core_robot.meta_client import ApiError
from core_robot.perception.frames import keyframe_indices, read_frames
from core_robot.sim.env import OBJECTS

KIND = {o.name: o.kind for o in OBJECTS}
CONCEPTS = {
    "red_block": "red cube", "green_block": "green cube", "blue_block": "blue cube",
    "yellow_block": "yellow cube", "red_bin": "red bin", "blue_bin": "blue bin", "mug": "white mug",
}
MIN_COVERAGE = 0.8
SAM_METADATA = {"mask_encoding": "one_bit"}


def _area(box) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def select_object(frames: dict[int, dict[int, dict]], kind: str) -> int | None:
    """Pick one SAM object id from {frame: {object_id: record}} for an object of `kind`."""
    seen: dict[int, list[float]] = {}
    for objs in frames.values():
        for oid, rec in objs.items():
            seen.setdefault(oid, []).append(_area(rec["box"]))
    if not seen:
        return None
    most = max(len(v) for v in seen.values())
    pool = [oid for oid, v in seen.items() if len(v) >= most / 2]
    if kind == "block":
        return min(pool, key=lambda o: (median(seen[o]), -len(seen[o]), o))
    if kind == "bin":
        return max(pool, key=lambda o: (median(seen[o]), len(seen[o]), -o))
    return max(pool, key=lambda o: (len(seen[o]), -o))


def _track(name: str, frames: dict[int, dict[int, dict]], attempted: list[int], source: str,
           n_frames: int) -> dict:
    oid = select_object(frames, KIND[name])
    recs = {f: objs[oid] for f, objs in frames.items() if oid is not None and oid in objs}
    hits = [f for f in attempted if f in recs]
    return {
        "name": name, "concept": CONCEPTS[name], "source": source, "object_id": oid,
        "candidates": len({o for objs in frames.values() for o in objs}),
        "frames": {int(f): {"box": list(recs[f]["box"]), "mask": recs[f].get("mask")} for f in sorted(recs)},
        "attempted": attempted, "coverage": len(hits) / len(attempted) if attempted else 0.0,
        "n_frames": n_frames, "error": None,
    }


def segment_video(client, video: str | Path, name: str, n_frames: int) -> dict:
    """Track `name` through the whole clip with one SAM video request."""
    try:
        r = client.sam_video(Path(video), CONCEPTS[name], metadata=SAM_METADATA)
    except ApiError as e:
        return {"name": name, "concept": CONCEPTS[name], "source": "video", "object_id": None, "candidates": 0,
                "frames": {}, "attempted": list(range(n_frames)), "coverage": 0.0, "n_frames": n_frames,
                "error": f"HTTP {e.status}: {e.message[:300]}"}
    return _track(name, r["frames"], list(range(n_frames)), "video", n_frames)


def segment_keyframes(client, video: str | Path, name: str, n_frames: int, out_dir: str | Path,
                      k: int = 8) -> dict:
    """Track `name` on k evenly spaced keyframes, one SAM image request each."""
    import imageio.v2 as imageio

    frames_rgb, _ = read_frames(video)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx = keyframe_indices(n_frames, k)
    found: dict[int, dict[int, dict]] = {}
    errors = []
    for i in idx:
        png = out / f"keyframe_{i:04d}.png"
        if not png.exists():
            imageio.imwrite(png, frames_rgb[i])
        try:
            r = client.sam_image(png, CONCEPTS[name], metadata=SAM_METADATA)
        except ApiError as e:
            errors.append(f"frame {i}: HTTP {e.status}")
            continue
        if 0 in r["frames"]:
            found[i] = r["frames"][0]
    t = _track(name, found, idx, "keyframes", n_frames)
    t["error"] = "; ".join(errors) or None
    return t


def track_object(client, video: str | Path, name: str, n_frames: int, keyframe_dir: str | Path,
                 k: int = 8) -> dict:
    """Video tracking, falling back to keyframes when it fails or covers < MIN_COVERAGE."""
    t = segment_video(client, video, name, n_frames)
    if t["error"] is None and t["coverage"] >= MIN_COVERAGE:
        return t
    reason = t["error"] or f"video coverage {t['coverage']:.2f} < {MIN_COVERAGE}"
    kf = segment_keyframes(client, video, name, n_frames, keyframe_dir, k)
    best = kf if kf["coverage"] > t["coverage"] else t
    best["fallback"] = {"reason": reason, "video_coverage": t["coverage"], "keyframe_coverage": kf["coverage"],
                        "used": best["source"]}
    return best
