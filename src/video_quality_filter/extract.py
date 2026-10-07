"""Extract evenly spaced frames and a short audio window with ffmpeg."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from video_quality_filter.config import AudioConfig, FramesConfig
from video_quality_filter.util import run_command, truncate


@dataclass
class ExtractedFrame:
    timestamp_sec: float
    path: Path
    image: np.ndarray


def frame_timestamps(duration: float, count: int) -> list[float]:
    """Return `count` timestamps spread across the clip, inset from the ends."""
    if duration <= 0 or count < 1:
        return []
    margin = min(0.05, duration * 0.05)
    start = margin
    end = max(duration - margin, start)
    if count == 1 or end == start:
        return [duration / 2.0]
    step = (end - start) / (count - 1)
    return [start + index * step for index in range(count)]


def extract_frames(
    video: Path,
    duration: float,
    config: FramesConfig,
    directory: Path,
) -> tuple[list[ExtractedFrame], list[str]]:
    frames: list[ExtractedFrame] = []
    warnings: list[str] = []
    timestamps = frame_timestamps(duration, config.count)
    scale = f"scale={config.max_side}:{config.max_side}:force_original_aspect_ratio=decrease"
    for index, timestamp in enumerate(timestamps):
        destination = directory / f"frame_{index:02d}.png"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            scale,
            str(destination),
        ]
        try:
            completed = run_command(command, config.timeout_sec)
        except subprocess.TimeoutExpired:
            warnings.append(f"кадр на {timestamp:.3f} с: ffmpeg превысил таймаут")
            continue
        if completed.returncode != 0 or not destination.is_file():
            detail = truncate(completed.stderr or "ffmpeg не извлёк кадр")
            warnings.append(f"кадр на {timestamp:.3f} с: {detail}")
            continue
        try:
            with Image.open(destination) as image:
                array = np.asarray(image.convert("RGB"))
        except OSError as exc:
            warnings.append(f"кадр на {timestamp:.3f} с не читается: {exc}")
            continue
        frames.append(ExtractedFrame(timestamp_sec=timestamp, path=destination, image=array))
    if frames and len(frames) < len(timestamps):
        warnings.append(f"извлечено {len(frames)} из {len(timestamps)} кадров")
    return frames, warnings


def extract_audio(
    video: Path,
    destination: Path,
    duration: float,
    config: AudioConfig,
) -> None:
    length = min(config.max_analyze_sec, duration) if duration > 0 else config.max_analyze_sec
    start = 0.0
    if duration > config.max_analyze_sec:
        start = max(0.0, (duration - config.max_analyze_sec) / 2.0)
        length = min(config.max_analyze_sec, max(duration - start, 0.0))
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{length:.3f}",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(config.sample_rate),
        "-f",
        "wav",
        str(destination),
    ]
    try:
        completed = run_command(command, config.extract_timeout_sec)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg превысил таймаут при извлечении аудио") from exc
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        detail = truncate(completed.stderr or "ffmpeg не извлёк аудио")
        raise RuntimeError(detail or "ffmpeg не извлёк аудио")
