"""Normalize a real demonstration clip: upright, at most 20 s, longest side 960 px, H.264 < 20 MB.

Uses the ffmpeg binary bundled with imageio-ffmpeg. ffmpeg applies the clip's rotation
metadata (phone videos) while re-encoding, so the output frames are upright and carry no
rotation flag.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

MAX_SECONDS = 20.0
MAX_SIDE = 960
MAX_BYTES = 20 * 1024 * 1024
CRF_STEPS = (23, 28, 33, 38)  # quality steps tried until the file is small enough


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: str | Path) -> dict:
    """Duration (s), stored size (w, h) and rotation (degrees) parsed from ffmpeg's header dump."""
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    size = re.search(r"Video: .*?(\d{2,5})x(\d{2,5})", out)
    rot = re.search(r"rotation of (-?[\d.]+) degrees", out) or re.search(r"rotate\s*:\s*(-?\d+)", out)
    if not (dur and size):
        raise ValueError(f"not a readable video: {path}")
    h, m, s = dur.groups()
    return {"duration_s": int(h) * 3600 + int(m) * 60 + float(s), "stored_size": (int(size[1]), int(size[2])),
            "rotation_deg": float(rot[1]) if rot else 0.0}


def normalize_clip(src: str | Path, dst: str | Path) -> dict:
    """Write the normalized clip to dst. Returns what was done and the output's properties."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = probe(src)
    # Longest side capped at MAX_SIDE (after rotation is applied); even dimensions for H.264.
    scale = (f"scale='if(gt(iw,ih),min({MAX_SIDE},iw),-2)':'if(gt(iw,ih),-2,min({MAX_SIDE},ih))',"
             "scale=trunc(iw/2)*2:trunc(ih/2)*2")
    for crf in CRF_STEPS:
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-t", str(MAX_SECONDS),
               "-vf", scale, "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-an", str(dst)]
        subprocess.run(cmd, check=True, capture_output=True)
        if dst.stat().st_size < MAX_BYTES:
            break
    else:
        raise ValueError(f"could not get {src.name} under {MAX_BYTES // (1024 * 1024)} MB")
    out = probe(dst)
    return {
        "source": str(src), "path": str(dst), "source_duration_s": round(info["duration_s"], 2),
        "trimmed": info["duration_s"] > MAX_SECONDS + 0.05, "source_rotation_deg": info["rotation_deg"],
        "source_stored_size": list(info["stored_size"]), "size": list(out["stored_size"]),
        "duration_s": round(out["duration_s"], 2), "bytes": dst.stat().st_size, "crf": crf,
    }
