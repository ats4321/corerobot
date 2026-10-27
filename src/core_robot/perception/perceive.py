"""Perception for one demonstration video: name the objects, track each with SAM 3.1,
render an overlay, and summarize the tracks as text for the planner.

Output (out_dir): video_5fps.mp4, first_frame.png, keyframes/, overlay.mp4, perception.json.
Nothing here reads the demo's ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from core_robot.perception.frames import resample_video, save_frame
from core_robot.perception.naming import name_objects
from core_robot.perception.overlay import render_overlay
from core_robot.perception.sam_tracking import track_object

MOVE_PX = 12  # a box center that shifts more than this (pixels) counts as moved


def perceive(video: str | Path, client, out_dir: str | Path, fps: float = 5.0, keyframes: int = 8,
             overlay: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clip = resample_video(video, out / "video_5fps.mp4", fps)
    first = save_frame(clip["path"], 0, out / "first_frame.png")
    naming = name_objects(client, first)
    tracks = {
        name: track_object(client, clip["path"], name, clip["n_frames"], out / "keyframes" / name, keyframes)
        for name in naming["objects"]
    }
    result = {
        "video": str(video), "clip": clip, "first_frame": str(first), "naming": naming, "tracks": tracks,
        "overlay": str(render_overlay(clip["path"], tracks, out / "overlay.mp4", clip["fps"])) if overlay else None,
    }
    (out / "perception.json").write_text(json.dumps(result, indent=1))
    return result


def load_perception(path: str | Path) -> dict:
    """Read perception.json, restoring integer frame keys."""
    p = json.loads(Path(path).read_text())
    for t in p["tracks"].values():
        t["frames"] = {int(k): v for k, v in t["frames"].items()}
    return p


def _center(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _inside(pt, box) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def track_summary(perception: dict) -> dict[str, dict]:
    """Per object: first/last seen frame and box, whether it moved, and when."""
    out = {}
    for name, t in perception["tracks"].items():
        frames = sorted(t["frames"])
        if not frames:
            out[name] = {"seen": False}
            continue
        start, end = t["frames"][frames[0]]["box"], t["frames"][frames[-1]]["box"]
        c0 = _center(start)
        moving = [f for f in frames if max(abs(a - b) for a, b in zip(_center(t["frames"][f]["box"]), c0)) > MOVE_PX]
        out[name] = {
            "seen": True, "coverage": round(t["coverage"], 2), "first_frame": frames[0], "last_frame": frames[-1],
            "start_box": start, "end_box": end, "moved": bool(moving) and
            max(abs(a - b) for a, b in zip(_center(end), c0)) > MOVE_PX,
            "first_move_frame": moving[0] if moving else None,
        }
    return out


def summary_text(perception: dict) -> str:
    """Plain-text track summary for the planner prompt (pixel coordinates, image left = robot's left)."""
    s = track_summary(perception)
    n = perception["clip"]["n_frames"]
    w, h = perception["clip"]["size"]
    lines = [f"Tracked objects ({n} frames at {perception['clip']['fps']:g} fps, image {w}x{h} px; "
             "boxes are [x1, y1, x2, y2] in pixels; image left is the robot's left):"]
    ends = {k: v["end_box"] for k, v in s.items() if v.get("seen")}
    for name, v in s.items():
        if not v["seen"]:
            lines.append(f"- {name}: named but not found by segmentation")
            continue
        c = _center(v["end_box"])
        within = [o for o, b in ends.items() if o != name and _inside(c, b)]
        moved = f"moved, starting around frame {v['first_move_frame']}" if v["moved"] else "did not move"
        lines.append(
            f"- {name}: start box {v['start_box']} (frame {v['first_frame']}), end box {v['end_box']} "
            f"(frame {v['last_frame']}); {moved}; mask coverage {v['coverage']:.0%}"
            + (f"; final center lies within the final box of: {', '.join(within)}" if within else "")
        )
    return "\n".join(lines)
