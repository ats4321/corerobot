"""Generate synthetic placeholder media in data/samples/ (seeded, deterministic).

  table_objects.png       640x480 still: red block, green block, blue mug on a table
  red_block_move.mp4      10 s, 10 fps, 640x360: a red block slides across a table
  red_block_move_3s.mp4   first 3 s of the same clip, for cheaper SAM video calls

These are drawn with PIL, not filmed. Replace with real footage when available.
"""

from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from core_robot.config import SAMPLES_DIR

TABLE = (176, 132, 92)
WALL = (222, 222, 215)


def table_scene(w: int, h: int, rng: np.random.Generator) -> Image.Image:
    img = Image.new("RGB", (w, h), WALL)
    d = ImageDraw.Draw(img)
    top = int(h * 0.3)
    d.rectangle([0, top, w, h], fill=TABLE)
    for _ in range(40):  # faint wood grain
        y = int(rng.integers(top, h))
        d.line([0, y, w, y + int(rng.integers(-3, 4))], fill=(160, 118, 80), width=1)
    return img


def block(d: ImageDraw.ImageDraw, x: int, y: int, s: int, color, shade):
    d.rectangle([x, y, x + s, y + s], fill=color, outline=shade, width=2)
    d.polygon([(x, y), (x + s // 4, y - s // 4), (x + s + s // 4, y - s // 4), (x + s, y)], fill=shade)
    d.polygon([(x + s, y), (x + s + s // 4, y - s // 4), (x + s + s // 4, y + s - s // 4), (x + s, y + s)], fill=shade)


def mug(d: ImageDraw.ImageDraw, x: int, y: int, r: int, hgt: int, color, shade):
    d.rectangle([x - r, y, x + r, y + hgt], fill=color)
    d.ellipse([x - r, y + hgt - r // 3, x + r, y + hgt + r // 3], fill=color)
    d.ellipse([x - r, y - r // 3, x + r, y + r // 3], fill=shade)
    d.arc([x + r - 6, y + hgt // 4, x + r + 22, y + 3 * hgt // 4], -90, 90, fill=color, width=6)


def make_image(path):
    rng = np.random.default_rng(0)
    img = table_scene(640, 480, rng)
    d = ImageDraw.Draw(img)
    block(d, 150, 280, 70, (210, 30, 30), (140, 15, 15))
    block(d, 330, 300, 60, (40, 170, 60), (20, 110, 35))
    mug(d, 500, 250, 40, 80, (40, 70, 200), (25, 40, 130))
    img.save(path)


def make_video(path, seconds: float, fps: int = 10):
    rng = np.random.default_rng(1)
    base = table_scene(640, 360, rng)
    n = int(seconds * fps)
    with imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=8,
        output_params=["-movflags", "+faststart"],
    ) as w:
        for i in range(int(10 * fps))[:n]:
            img = base.copy()
            d = ImageDraw.Draw(img)
            block(d, 470, 230, 50, (40, 170, 60), (20, 110, 35))
            t = i / (10 * fps - 1)
            block(d, int(40 + t * 380), 180 + int(40 * t), 50, (210, 30, 30), (140, 15, 15))
            w.append_data(np.asarray(img))


def main():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    make_image(SAMPLES_DIR / "table_objects.png")
    make_video(SAMPLES_DIR / "red_block_move.mp4", 10)
    make_video(SAMPLES_DIR / "red_block_move_3s.mp4", 3)
    for p in sorted(SAMPLES_DIR.glob("*.*")):
        print(f"{p.name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
