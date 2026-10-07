"""Tiny synthetic videos for pipeline tests. Requires ffmpeg on PATH."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def render_video(
    path: Path,
    *,
    size: str = "640x480",
    duration: float = 2.0,
    rate: int = 25,
    color: str | None = None,
    audio: str = "tone",
) -> None:
    if color:
        video = f"color=c={color}:s={size}:r={rate}:d={duration}"
    else:
        video = f"testsrc=size={size}:rate={rate}:duration={duration}"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", video]
    if audio == "tone":
        command += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=44100:duration={duration},volume=0.2",
        ]
    elif audio == "silence":
        command += ["-f", "lavfi", "-i", f"anullsrc=channel_layout=mono:sample_rate=44100:duration={duration}"]
    elif audio != "none":
        raise ValueError(audio)
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio != "none":
        command += ["-c:a", "aac", "-shortest"]
    command.append(str(path))
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(detail or f"ffmpeg exited {completed.returncode}")
