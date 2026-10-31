"""Real-footage perception mode (footage="real"). The simulator path does not use this module.

Differences from the simulator path (perception.sam_tracking / perception.plausibility):
  * Plausibility uses image-relative measures only: box area as a fraction of the frame, box
    aspect ratio, and mask colour. No simulator camera, table projection or bin size.
  * Fragment merging: instances of one SAM request with comparable sizes (median areas within
    MERGE_SIZE_RATIO) whose frame ranges overlap or are separated by at most MERGE_GAP frames, and
    which overlap in space (intersection over the smaller box >= MERGE_OVERLAP on a shared frame,
    or between one piece's last box and the other's first), are merged into one track (e.g. a cube
    split into pieces by a ghosted hand). Every merge is reported.
  * Assignment by class from the naming step, pooled per colour: every instance returned by any
    SAM request for that colour is a candidate. Bins (and the mug) take the largest plausible
    instance, blocks the smallest; an instance is assigned to at most one name. So if "red cube"
    returns both the red box and the red cube, the box goes to red_bin and the cube to red_block.
    Never selected by track length.
Requests: one SAM video request per named object with its default phrase (the same requests as
the simulator path). Only an object left without any candidate triggers extra requests.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median

import numpy as np

from core_robot.meta_client import ApiError
from core_robot.perception.frames import read_frames
from core_robot.perception.overlay import decode_mask
from core_robot.perception.plausibility import color_matches
from core_robot.perception.sam_tracking import CONCEPTS, PHRASES, SAM_METADATA
from core_robot.sim.env import OBJECTS

SPEC = {o.name: o for o in OBJECTS}
AREA_FRACTION = {"block": (0.001, 0.04), "bin": (0.03, 0.45), "mug": (0.005, 0.2)}  # of the frame area
ASPECT = (0.4, 2.5)  # box width / height
MERGE_OVERLAP = 0.5  # intersection over the smaller box, on a shared frame or across the gap
MERGE_GAP = 3  # frames
MERGE_SIZE_RATIO = 3.0  # pieces of one object have comparable sizes (keeps a cube out of its box)
DUPLICATE_IOU = 0.5  # an instance this close to an already assigned one is the same object


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    inter = ix * iy
    ua = (a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - inter
    return inter / ua if ua else 0.0


def _overlap_small(a, b) -> float:
    """Intersection over the smaller box's area."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    return ix * iy / min(_area(a), _area(b))


def _area(box) -> float:
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def _median_box(frames: dict) -> list:
    boxes = [r["box"] for r in frames.values()]
    return [median(b[i] for b in boxes) for i in range(4)]


def instances(frames_by_id: dict[int, dict[int, dict]]) -> dict[str, dict]:
    """{frame: {id: rec}} -> {"<id>": {"frames": {frame: rec}}} (one entry per SAM object id)."""
    out: dict[str, dict] = {}
    for f, objs in frames_by_id.items():
        for oid, rec in objs.items():
            out.setdefault(str(oid), {"frames": {}})["frames"][int(f)] = rec
    return out


def merge_fragments(insts: dict[str, dict]) -> tuple[dict[str, dict], list[dict]]:
    """Merge same-request fragments of one object (see the module docstring for the rule).

    On frames where both pieces exist, the record of the piece seen on more frames is kept.
    Returns (instances keyed "a+b" for merged ones, list of merge reports).
    """
    items = {k: {"frames": dict(v["frames"])} for k, v in insts.items()}
    merges = []
    changed = True
    while changed:
        changed = False
        keys = sorted(items, key=lambda k: (min(items[k]["frames"]), k))
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                fa, fb = items[a]["frames"], items[b]["frames"]
                if min(fb) > max(fa):
                    gap = min(fb) - max(fa)
                elif min(fa) > max(fb):
                    gap = min(fa) - max(fb)
                else:
                    gap = 0  # frame ranges overlap
                if gap > MERGE_GAP:
                    continue
                ma = median(_area(r["box"]) for r in fa.values())
                mb = median(_area(r["box"]) for r in fb.values())
                if max(ma, mb) / min(ma, mb) > MERGE_SIZE_RATIO:
                    continue
                shared = set(fa) & set(fb)
                if shared:
                    overlap = max(_overlap_small(fa[k]["box"], fb[k]["box"]) for k in shared)
                elif min(fb) > max(fa):
                    overlap = _overlap_small(fa[max(fa)]["box"], fb[min(fb)]["box"])
                else:
                    overlap = _overlap_small(fb[max(fb)]["box"], fa[min(fa)]["box"])
                if overlap < MERGE_OVERLAP:
                    continue
                primary, secondary = (fa, fb) if len(fa) >= len(fb) else (fb, fa)
                merges.append({"merged": [a, b], "frames": [[min(fa), max(fa)], [min(fb), max(fb)]],
                               "gap": gap, "shared_frames": len(shared), "overlap": round(overlap, 3),
                               "size_ratio": round(max(ma, mb) / min(ma, mb), 2)})
                del items[a], items[b]
                items[f"{a}+{b}"] = {"frames": {**secondary, **primary}}
                changed = True
                break
            if changed:
                break
    return items, merges


def plausible_real(name: str, box, mask: str | None, frame_rgb: np.ndarray) -> tuple[bool, list[str]]:
    """Image-relative plausibility: area fraction, aspect ratio, mask colour."""
    kind, color = SPEC[name].kind, SPEC[name].color
    h_img, w_img = frame_rgb.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2 - x1 + 1, y2 - y1 + 1
    frac = w * h / (w_img * h_img)
    lo, hi = AREA_FRACTION[kind]
    reasons = []
    if not lo <= frac <= hi:
        reasons.append(f"box covers {frac:.4f} of the frame; a {kind} should cover {lo}-{hi}")
    if not ASPECT[0] <= w / h <= ASPECT[1]:
        reasons.append(f"aspect ratio {w / h:.2f} outside {ASPECT[0]}-{ASPECT[1]}")
    raster = decode_mask(mask)
    if raster is not None:
        hh, ww = min(raster.shape[0], h_img - y1), min(raster.shape[1], w_img - x1)
        px = frame_rgb[y1:y1 + hh, x1:x1 + ww][raster[:hh, :ww]]
        if len(px) and not color_matches(color, np.median(px, axis=0)):
            reasons.append(f"masked pixels are not {color}")
    return not reasons, reasons


def assign(named: list[str], calls: dict[str, dict], frames_rgb: list[np.ndarray], n_frames: int) -> dict:
    """Choose one track per named object.

    calls: {phrase: {"color": c, "frames": {frame: {id: rec}}}} for every SAM video request made.
    Returns {"tracks": {name: track}, "candidates": {...}, "merges": {phrase: [...]}}.
    """
    pool, merges = [], {}
    for phrase, call in calls.items():
        merged, m = merge_fragments(instances(call["frames"]))
        merges[phrase] = m
        for key, inst in merged.items():
            fr = inst["frames"]
            pool.append({"phrase": phrase, "id": key, "color": call["color"], "frames": fr,
                         "median_box": _median_box(fr), "area": median(_area(r["box"]) for r in fr.values()),
                         "span": [min(fr), max(fr)]})
    order = sorted(named, key=lambda n: (SPEC[n].kind != "bin", SPEC[n].kind != "mug", n))  # bins, mug, then blocks
    taken: list[dict] = []
    tracks, report = {}, {}
    for name in order:
        kind, color = SPEC[name].kind, SPEC[name].color
        cands = []
        for c in pool:
            if c["color"] != color or c in taken:
                continue
            if any(_iou(c["median_box"], t["median_box"]) > DUPLICATE_IOU for t in taken):
                continue
            f0 = min(c["frames"])
            rec = c["frames"][f0]
            ok, why = plausible_real(name, rec["box"], rec.get("mask"), frames_rgb[f0])
            cands.append({**{k: c[k] for k in ("phrase", "id", "span", "area")}, "plausible": ok, "why_not": why,
                          "_c": c})
        ok_cands = [c for c in cands if c["plausible"]]
        report[name] = [{k: v for k, v in c.items() if k != "_c"} for c in cands]
        if not ok_cands:
            tracks[name] = None
            continue
        pick = (min if kind == "block" else max)(ok_cands, key=lambda c: c["area"])["_c"]
        taken.append(pick)
        fr = pick["frames"]
        tracks[name] = {
            "name": name, "concept": pick["phrase"], "source": "video", "object_id": pick["id"],
            "candidates": len(cands), "frames": {int(f): {"box": list(r["box"]), "mask": r.get("mask")}
                                                 for f, r in sorted(fr.items())},
            "attempted": list(range(n_frames)), "coverage": len(fr) / n_frames, "n_frames": n_frames,
            "error": None, "selection": f"{'smallest' if kind == 'block' else 'largest'} plausible {color} instance "
                                        f"across this colour's SAM results (from '{pick['phrase']}', id {pick['id']})",
        }
    return {"tracks": tracks, "candidates": report, "merges": merges}


def track_all_real(client, video: str | Path, named: list[str], n_frames: int, first_frame: str | Path,
                   keyframe_dir: str | Path) -> tuple[dict, dict]:
    """Real-mode tracks for every named object. Returns (tracks, report)."""
    frames_rgb, _ = read_frames(video)
    calls, attempts = {}, {}
    for name in named:
        phrase = CONCEPTS[name]
        try:
            r = client.sam_video(Path(video), phrase, metadata=SAM_METADATA)
            calls[phrase] = {"color": SPEC[name].color, "frames": r["frames"]}
            attempts[name] = [{"mode": "video", "phrase": phrase, "instances": len(instances(r["frames"]))}]
        except ApiError as e:
            attempts[name] = [{"mode": "video", "phrase": phrase, "error": f"HTTP {e.status}"}]
    result = assign(named, calls, frames_rgb, n_frames)
    # Objects with no candidate at all: probe the remaining phrases on the first frame, then
    # segment the video with the first phrase that yields a plausible instance.
    for name in [n for n, t in result["tracks"].items() if t is None]:
        for phrase in PHRASES[name][1:]:
            try:
                img = client.sam_image(Path(first_frame), phrase, metadata=SAM_METADATA)
            except ApiError:
                continue
            found = img["frames"].get(0, {})
            plaus = [plausible_real(name, rec["box"], rec.get("mask"), frames_rgb[0])[0] for rec in found.values()]
            attempts[name].append({"mode": "probe-image", "phrase": phrase, "instances": len(found),
                                   "plausible": sum(plaus)})
            if any(plaus):
                r = client.sam_video(Path(video), phrase, metadata=SAM_METADATA)
                calls[phrase] = {"color": SPEC[name].color, "frames": r["frames"]}
                attempts[name].append({"mode": "video", "phrase": phrase, "instances": len(instances(r["frames"]))})
                result = assign(named, calls, frames_rgb, n_frames)
                break
    tracks = {}
    for name in named:
        t = result["tracks"].get(name)
        if t is None:
            t = {"name": name, "concept": CONCEPTS[name], "source": "video", "object_id": None, "candidates": 0,
                 "frames": {}, "attempted": list(range(n_frames)), "coverage": 0.0, "n_frames": n_frames,
                 "error": None}
        t["attempts"] = attempts.get(name, [])
        tracks[name] = t
    return tracks, {"candidates": result["candidates"], "merges": result["merges"]}
