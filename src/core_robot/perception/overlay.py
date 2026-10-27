"""Overlay video: each tracked object's mask (when decodable) and box drawn on the clip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from core_robot.perception.frames import read_frames, write_video
from core_robot.sim.env import OBJECTS

RGB = {"red": (230, 40, 40), "green": (40, 190, 70), "blue": (40, 90, 230), "yellow": (240, 200, 20),
       "white": (250, 250, 250)}
COLOR = {o.name: RGB[o.color] for o in OBJECTS}


def decode_mask(mask: str | None) -> np.ndarray | None:
    """'H,W,<marker>payload' -> bool array (H, W), or None if absent or undecodable."""
    if not mask:
        return None
    try:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster

        h, w, payload = mask.split(",", 2)
        enc = "one_bit" if payload.startswith("!") else "lossless"
        raster = decode_mask_to_raster(SegmentationMask(encoding=enc, payload=payload, width=int(w), height=int(h)))
        return np.frombuffer(raster, dtype=np.uint8).reshape(int(h), int(w)).astype(bool)
    except Exception:
        return None


def paint_mask(img: np.ndarray, box, raster: np.ndarray, rgb, alpha: float = 0.45) -> None:
    """Blend a mask raster into img; the raster covers the record's box (origin = box top-left)."""
    x1, y1 = int(box[0]), int(box[1])
    h = min(raster.shape[0], img.shape[0] - y1)
    w = min(raster.shape[1], img.shape[1] - x1)
    if h <= 0 or w <= 0:
        return
    region = img[y1:y1 + h, x1:x1 + w]
    m = raster[:h, :w]
    region[m] = (region[m] * (1 - alpha) + np.array(rgb) * alpha).astype(np.uint8)


def render_overlay(video: str | Path, tracks: dict[str, dict], out: str | Path, fps: float) -> Path:
    frames, _ = read_frames(video)
    drawn = []
    for i, frame in enumerate(frames):
        img = frame.copy()
        for name, t in tracks.items():
            rec = t["frames"].get(i)
            if rec and (raster := decode_mask(rec.get("mask"))) is not None:
                paint_mask(img, rec["box"], raster, COLOR[name])
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        for name, t in tracks.items():
            rec = t["frames"].get(i)
            if rec:
                x1, y1, x2, y2 = rec["box"]
                d.rectangle([x1, y1, x2, y2], outline=COLOR[name], width=2)
                d.text((x1 + 2, max(0, y1 - 12)), name, fill=COLOR[name])
        d.text((6, 6), f"frame {i}", fill=(20, 20, 20))
        drawn.append(np.asarray(pil))
    return write_video(drawn, out, fps)
