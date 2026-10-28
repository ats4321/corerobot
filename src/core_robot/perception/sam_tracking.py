"""SAM 3.1 tracking of one named object per request, with phrase probing and a keyframe fallback.

1. segment_video sends the object's default concept phrase (e.g. "red cube") with the clip.
2. If that track covers fewer than MIN_COVERAGE of the frames, alternative phrases for the
   object are probed on the first frame (one SAM image request each). The first phrase that
   returns an instance that is plausible for the object (size, on-table, color; see
   perception.plausibility, which uses no ground truth) is used for a second video request,
   keeping only plausible-sized instances.
3. If coverage is still below MIN_COVERAGE, the best phrase is re-segmented on evenly spaced
   keyframes as single images. The track with the best coverage is kept.

Every request and what it returned is recorded in track["attempts"], so a missing object can
be told apart: "not named" (naming step) vs "named, but SAM returned no instance".

A phrase can match several objects (e.g. "red cube" also matches a red bin), so one SAM
object id is selected per name: among ids seen on at least half as many frames as the
most-seen id, blocks take the smallest median box, bins the largest, the mug the most frames.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median

from core_robot.meta_client import ApiError
from core_robot.perception.frames import keyframe_indices, read_frames
from core_robot.perception.plausibility import plausible, size_range
from core_robot.sim.env import OBJECTS

KIND = {o.name: o.kind for o in OBJECTS}
PHRASES = {
    "red_block": ["red cube"], "green_block": ["green cube"], "blue_block": ["blue cube"],
    "yellow_block": ["yellow cube"],
    "red_bin": ["red bin", "red box", "red container", "red tray", "red square tray"],
    "blue_bin": ["blue bin", "blue box", "blue container", "blue tray", "blue square tray"],
    "mug": ["white mug", "white cup", "white cylinder"],
}
CONCEPTS = {name: phrases[0] for name, phrases in PHRASES.items()}  # default phrase per object
MIN_COVERAGE = 0.8
SAM_METADATA = {"mask_encoding": "one_bit"}


def _area(box) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _size_ok(name: str, box) -> bool:
    wmin, wmax, hmin, hmax = size_range(KIND[name])
    w, h = box[2] - box[0] + 1, box[3] - box[1] + 1
    return 0.7 * wmin <= w <= 1.4 * wmax and 0.7 * hmin <= h <= 1.4 * hmax


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


def _instances(frames: dict[int, dict[int, dict]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for objs in frames.values():
        for oid in objs:
            counts[oid] = counts.get(oid, 0) + 1
    return counts


def _track(name: str, phrase: str, frames: dict[int, dict[int, dict]], attempted: list[int], source: str,
           n_frames: int) -> dict:
    oid = select_object(frames, KIND[name])
    recs = {f: objs[oid] for f, objs in frames.items() if oid is not None and oid in objs}
    hits = [f for f in attempted if f in recs]
    return {
        "name": name, "concept": phrase, "source": source, "object_id": oid,
        "candidates": len(_instances(frames)),
        "frames": {int(f): {"box": list(recs[f]["box"]), "mask": recs[f].get("mask")} for f in sorted(recs)},
        "attempted": attempted, "coverage": len(hits) / len(attempted) if attempted else 0.0,
        "n_frames": n_frames, "error": None,
    }


def _record(t: dict, mode: str) -> dict:
    return {"mode": mode, "phrase": t["concept"], "instances": t["candidates"], "frames_with_object": len(t["frames"]),
            "attempted": len(t["attempted"]), "coverage": round(t["coverage"], 3), "error": t["error"]}


def segment_video(client, video: str | Path, name: str, n_frames: int, phrase: str | None = None,
                  size_filter: bool = False) -> dict:
    """Track `name` through the clip with one SAM video request for `phrase` (default concept)."""
    phrase = phrase or CONCEPTS[name]
    try:
        r = client.sam_video(Path(video), phrase, metadata=SAM_METADATA)
    except ApiError as e:
        t = _track(name, phrase, {}, list(range(n_frames)), "video", n_frames)
        t["error"] = f"HTTP {e.status}: {e.message[:300]}"
        return t
    frames = r["frames"]
    if size_filter:
        frames = {f: {o: rec for o, rec in objs.items() if _size_ok(name, rec["box"])} for f, objs in frames.items()}
        frames = {f: objs for f, objs in frames.items() if objs}
    return _track(name, phrase, frames, list(range(n_frames)), "video", n_frames)


def probe_phrases(client, first_frame: str | Path, frame_rgb, name: str, phrases: list[str]) -> dict:
    """Try phrases on the first frame; accept the first with a plausible instance."""
    tried = []
    for phrase in phrases:
        try:
            r = client.sam_image(Path(first_frame), phrase, metadata=SAM_METADATA)
        except ApiError as e:
            tried.append({"phrase": phrase, "error": f"HTTP {e.status}", "instances": []})
            continue
        found = []
        for oid, rec in r["frames"].get(0, {}).items():
            ok, why = plausible(name, rec["box"], rec.get("mask"), frame_rgb)
            found.append({"id": oid, "box": list(rec["box"]), "plausible": ok, "why_not": why})
        tried.append({"phrase": phrase, "instances": found})
        if any(f["plausible"] for f in found):
            return {"phrase": phrase, "tried": tried}
    return {"phrase": None, "tried": tried}


def segment_keyframes(client, video: str | Path, name: str, n_frames: int, out_dir: str | Path,
                      k: int = 8, phrase: str | None = None) -> dict:
    """Track `name` on k evenly spaced keyframes, one SAM image request each."""
    import imageio.v2 as imageio

    phrase = phrase or CONCEPTS[name]
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
            r = client.sam_image(png, phrase, metadata=SAM_METADATA)
        except ApiError as e:
            errors.append(f"frame {i}: HTTP {e.status}")
            continue
        if 0 in r["frames"]:
            found[i] = r["frames"][0]
    t = _track(name, phrase, found, idx, "keyframes", n_frames)
    t["error"] = "; ".join(errors) or None
    return t


def track_object(client, video: str | Path, name: str, n_frames: int, keyframe_dir: str | Path,
                 k: int = 8, first_frame: str | Path | None = None) -> dict:
    """Default phrase on video; if coverage < MIN_COVERAGE, probe phrases on the first frame and
    retry the video with an accepted phrase; then keyframes. Keeps the best-covered track."""
    t = segment_video(client, video, name, n_frames)
    attempts = [_record(t, "video")]
    if t["error"] is None and t["coverage"] >= MIN_COVERAGE:
        t["attempts"] = attempts
        return t
    reason = t["error"] or f"video coverage {t['coverage']:.2f} < {MIN_COVERAGE}"
    best, probe = t, None
    alternatives = PHRASES[name][1:]
    if alternatives and first_frame is not None:
        frame_rgb = read_frames(video)[0][0]
        probe = probe_phrases(client, first_frame, frame_rgb, name, alternatives)
        attempts += [{"mode": "probe-image", "phrase": p["phrase"], "instances": len(p["instances"]),
                      "plausible": sum(i["plausible"] for i in p["instances"])} for p in probe["tried"]]
        if probe["phrase"]:
            t2 = segment_video(client, video, name, n_frames, probe["phrase"], size_filter=True)
            attempts.append(_record(t2, "video"))
            if t2["coverage"] > best["coverage"]:
                best = t2
    if best["coverage"] < MIN_COVERAGE:
        kf = segment_keyframes(client, video, name, n_frames, keyframe_dir, k, phrase=best["concept"])
        attempts.append(_record(kf, "keyframes"))
        if kf["coverage"] > best["coverage"]:
            best = kf
    best["attempts"] = attempts
    best["probe"] = probe
    best["fallback"] = {"reason": reason, "used": f"{best['source']} '{best['concept']}'"}
    return best
