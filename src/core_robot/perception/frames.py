"""Video frame sampling for perception."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def read_frames(video: str | Path) -> tuple[list[np.ndarray], float]:
    reader = imageio.get_reader(str(video))
    fps = float(reader.get_meta_data().get("fps", 10.0))
    frames = [np.asarray(f) for f in reader]
    reader.close()
    return frames, fps


def write_video(frames: list[np.ndarray], path: str | Path, fps: float) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=8,
                            output_params=["-movflags", "+faststart"]) as w:
        for f in frames:
            w.append_data(f)
    return path


def resample_video(src: str | Path, dst: str | Path, fps: float = 5.0) -> dict:
    """Keep every k-th frame so the clip plays at about `fps` (cheaper SAM requests).

    Returns {"path", "fps", "n_frames", "size": (w, h), "source_index": [source frame per frame]}.
    """
    frames, src_fps = read_frames(src)
    step = max(1, round(src_fps / fps))
    idx = list(range(0, len(frames), step))
    if idx[-1] != len(frames) - 1:
        idx.append(len(frames) - 1)  # always keep the final frame (the end state)
    kept = [frames[i] for i in idx]
    write_video(kept, dst, src_fps / step)
    h, w = kept[0].shape[:2]
    return {"path": str(dst), "fps": src_fps / step, "n_frames": len(kept), "size": (w, h), "source_index": idx}


def keyframe_indices(n_frames: int, k: int = 8) -> list[int]:
    """k evenly spaced frame indices including the first and the last."""
    if n_frames <= k:
        return list(range(n_frames))
    return sorted({round(i * (n_frames - 1) / (k - 1)) for i in range(k)})


def save_frame(video: str | Path, index: int, png: str | Path) -> Path:
    frames, _ = read_frames(video)
    png = Path(png)
    png.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(png, frames[index])
    return png
