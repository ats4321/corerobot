"""SAM gate: did perception find and track every block and bin? (Fixed before any results.)

An object passes when all hold:
  * it was named and SAM returned a track for it;
  * it has a mask on at least 80% of the frames the track attempted;
  * the box on the first frame contains the object's true start position, and the box on
    the last frame contains its true end position, both projected into the angled demo
    camera image. If that frame has no mask, the nearest frame within 3 frames is used.
A demo passes when every block and bin passes. The mug is optional.
Ground truth (initial/final states) is used only here, for scoring.
"""

from __future__ import annotations

MIN_COVERAGE = 0.8
FRAME_TOLERANCE = 3
REQUIRED_KINDS = ("block", "bin")


def _frame_near(frames: dict, target: int, direction: int) -> int | None:
    """The frame with a mask closest to `target`, searching up to FRAME_TOLERANCE frames inward."""
    for d in range(FRAME_TOLERANCE + 1):
        f = target + direction * d
        if f in frames:
            return f
    return None


def _contains(box, pt) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _sam_attempts_text(track: dict) -> str:
    attempts = track.get("attempts") or [{"mode": track.get("source", "video"), "phrase": track.get("concept"),
                                          "instances": track.get("candidates", 0)}]
    return "; ".join(f"{a['mode']} '{a['phrase']}': {a.get('instances', 0)} instance(s)" for a in attempts)


def check_object(track: dict | None, start_px, end_px, n_frames: int, named: bool = True) -> dict:
    if not named:
        return {"pass": False, "reason": "not named by the object-naming step", "coverage": 0.0, "named": False}
    if not track or not track.get("frames"):
        detail = _sam_attempts_text(track) if track else "no SAM request"
        return {"pass": False, "reason": f"named, but SAM returned no instance ({detail})", "coverage": 0.0,
                "named": True}
    frames = track["frames"]
    f0 = _frame_near(frames, 0, +1)
    f1 = _frame_near(frames, n_frames - 1, -1)
    start_ok = f0 is not None and _contains(frames[f0]["box"], start_px)
    end_ok = f1 is not None and _contains(frames[f1]["box"], end_px)
    cov_ok = track["coverage"] >= MIN_COVERAGE
    reasons = [r for r, bad in (
        (f"coverage {track['coverage']:.0%} < {MIN_COVERAGE:.0%}", not cov_ok),
        ("no mask within 3 frames of the start" if f0 is None else "start box misses the true start", not start_ok),
        ("no mask within 3 frames of the end" if f1 is None else "end box misses the true end", not end_ok),
    ) if bad]
    return {"pass": not reasons, "reason": "; ".join(reasons), "coverage": round(track["coverage"], 3), "named": True,
            "phrase": track.get("concept"),
            "start_frame": f0, "end_frame": f1, "source": track.get("source"),
            "start_px": [round(v, 1) for v in start_px], "end_px": [round(v, 1) for v in end_px]}


def sam_gate(perception: dict, initial_state: dict, final_state: dict, project) -> dict:
    """project(world_xyz) -> (col, row) in the demo video's pixels (env.project on the angled camera)."""
    n = perception["clip"]["n_frames"]
    objects = {}
    for name, s0 in initial_state.items():
        if s0["kind"] not in REQUIRED_KINDS and name not in perception["tracks"]:
            continue
        r = check_object(perception["tracks"].get(name), project(s0["position"]),
                         project(final_state[name]["position"]), n, named=name in perception["naming"]["objects"])
        r["required"] = s0["kind"] in REQUIRED_KINDS
        objects[name] = r
    return {"pass": all(o["pass"] for o in objects.values() if o["required"]), "objects": objects}
