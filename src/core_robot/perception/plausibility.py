"""Is a SAM instance plausibly the named object? Image evidence and camera geometry only.

Used to accept a concept phrase during probing. It never looks at where the object really
is (no ground-truth state): it checks that the box has the pixel size an object of this
kind can have anywhere on the table (from the known camera and object dimensions), that
its center lies on the table, and that the masked pixels have the object's color.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from matplotlib.path import Path as Polygon

from core_robot.perception.overlay import decode_mask
from core_robot.sim.env import (
    BIN_INNER, BIN_WALL_H, BIN_WALL_T, BLOCK_SIZE, MUG_HEIGHT, MUG_RADIUS, OBJECTS, WORKSPACE_X, WORKSPACE_Y,
)

SPEC = {o.name: o for o in OBJECTS}
DIMS = {  # (x, y, z) extent in meters
    "block": (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
    "bin": (BIN_INNER + 2 * BIN_WALL_T, BIN_INNER + 2 * BIN_WALL_T, BIN_WALL_H),
    "mug": (2 * MUG_RADIUS, 2 * MUG_RADIUS, MUG_HEIGHT),
}
SIZE_SLACK = (0.7, 1.4)  # accept boxes from 70% of the smallest to 140% of the largest projection
TABLE_MARGIN = 0.06  # meters beyond the object workspace


@lru_cache(maxsize=None)
def size_range(kind: str, camera: str = "angled") -> tuple[float, float, float, float]:
    """(w_min, w_max, h_min, h_max) in pixels for an object of `kind` anywhere in the workspace."""
    from core_robot.sim.env import TabletopEnv

    env = TabletopEnv()
    dx, dy, dz = DIMS[kind]
    ws, hs = [], []
    for x in np.linspace(*WORKSPACE_X, 6):
        for y in np.linspace(*WORKSPACE_Y, 7):
            pts = [env.project((x + sx * dx / 2, y + sy * dy / 2, z), camera)
                   for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, dz)]
            us, vs = [p[0] for p in pts], [p[1] for p in pts]
            ws.append(max(us) - min(us))
            hs.append(max(vs) - min(vs))
    env.close()
    return min(ws), max(ws), min(hs), max(hs)


@lru_cache(maxsize=None)
def table_polygon(camera: str = "angled") -> Polygon:
    from core_robot.sim.env import TabletopEnv

    env = TabletopEnv()
    x0, x1 = WORKSPACE_X[0] - TABLE_MARGIN, WORKSPACE_X[1] + TABLE_MARGIN
    y0, y1 = WORKSPACE_Y[0] - TABLE_MARGIN, WORKSPACE_Y[1] + TABLE_MARGIN
    poly = Polygon([env.project(p, camera) for p in ((x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0))])
    env.close()
    return poly


def color_matches(color: str, rgb) -> bool:
    r, g, b = (float(v) for v in rgb)
    return {
        "red": r > 100 and r > 1.5 * g and r > 1.5 * b,
        "green": g > 90 and g > 1.3 * r and g > 1.2 * b,
        "blue": b > 100 and b > 1.3 * r and b > 1.2 * g,
        "yellow": r > 140 and g > 120 and b < 0.6 * min(r, g),
        "white": min(r, g, b) > 150 and max(r, g, b) - min(r, g, b) < 60,
    }[color]


def plausible(name: str, box, mask: str | None, frame_rgb: np.ndarray) -> tuple[bool, list[str]]:
    """(ok, reasons it is not plausible). frame_rgb is the image the box was found in."""
    kind, color = SPEC[name].kind, SPEC[name].color
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2 - x1 + 1, y2 - y1 + 1
    wmin, wmax, hmin, hmax = size_range(kind)
    reasons = []
    if not (SIZE_SLACK[0] * wmin <= w <= SIZE_SLACK[1] * wmax and SIZE_SLACK[0] * hmin <= h <= SIZE_SLACK[1] * hmax):
        reasons.append(f"box {w}x{h}px outside the {kind} size range "
                       f"{SIZE_SLACK[0] * wmin:.0f}-{SIZE_SLACK[1] * wmax:.0f} x {SIZE_SLACK[0] * hmin:.0f}-{SIZE_SLACK[1] * hmax:.0f}px")
    if not table_polygon().contains_point(((x1 + x2) / 2, (y1 + y2) / 2)):
        reasons.append("box center is off the table")
    raster = decode_mask(mask)
    if raster is None:
        reasons.append("mask not decodable")
    else:
        hh, ww = min(raster.shape[0], frame_rgb.shape[0] - y1), min(raster.shape[1], frame_rgb.shape[1] - x1)
        px = frame_rgb[y1:y1 + hh, x1:x1 + ww][raster[:hh, :ww]]
        if len(px) == 0:
            reasons.append("empty mask")
        else:
            med = np.median(px, axis=0)
            if not color_matches(color, med):
                reasons.append(f"masked pixels are not {color} (median RGB {[int(v) for v in med]})")
    return not reasons, reasons
