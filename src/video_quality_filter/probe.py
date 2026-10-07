"""Read technical metadata with ffprobe."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from video_quality_filter.models import VideoMetadata
from video_quality_filter.util import run_command, truncate


def probe_video(path: Path, timeout: float) -> VideoMetadata:
    file_size = path.stat().st_size if path.exists() else 0
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = run_command(command, timeout)
    except subprocess.TimeoutExpired:
        return _failed(path, file_size, "ffprobe превысил таймаут")
    except OSError as exc:
        return _failed(path, file_size, f"не удалось запустить ffprobe: {exc}")
    if completed.returncode != 0:
        detail = truncate(completed.stderr or completed.stdout or "ffprobe завершился с ошибкой")
        return _failed(path, file_size, detail or "ffprobe завершился с ошибкой")
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return _failed(path, file_size, "ffprobe вернул не-JSON")
    if not isinstance(payload, dict):
        return _failed(path, file_size, "ffprobe вернул неожиданный JSON")
    return parse_probe_payload(path, file_size, payload)


def parse_probe_payload(path: Path, file_size: int, payload: dict) -> VideoMetadata:
    streams = payload.get("streams") or []
    if not isinstance(streams, list):
        streams = []
    fmt = payload.get("format") or {}
    if not isinstance(fmt, dict):
        fmt = {}
    video = _primary_video_stream(streams)
    audio = _primary_audio_stream(streams)
    duration = _duration(fmt, video)
    audio_duration = _positive_float(audio.get("duration")) if audio else None
    width = _positive_int(video.get("width")) if video else None
    height = _positive_int(video.get("height")) if video else None
    fps = _fps(video, duration) if video else None
    return VideoMetadata(
        path=str(path.resolve()),
        file_size_bytes=file_size,
        duration_sec=duration,
        audio_duration_sec=audio_duration,
        width=width,
        height=height,
        fps=fps,
        video_codec=_text(video.get("codec_name")) if video else None,
        audio_codec=_text(audio.get("codec_name")) if audio else None,
        has_audio=audio is not None,
        format_name=_text(fmt.get("format_name")),
        bit_rate=_positive_int(fmt.get("bit_rate")),
        pixel_format=_text(video.get("pix_fmt")) if video else None,
        probe_error=None,
    )


def _primary_video_stream(streams: list) -> dict | None:
    videos = [stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"]
    real = [stream for stream in videos if not _is_attached_picture(stream)]
    chosen = real or videos
    return chosen[0] if chosen else None


def _primary_audio_stream(streams: list) -> dict | None:
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "audio":
            return stream
    return None


def _is_attached_picture(stream: dict) -> bool:
    disposition = stream.get("disposition") or {}
    if not isinstance(disposition, dict):
        return False
    return bool(disposition.get("attached_pic"))


def _duration(fmt: dict, video: dict | None) -> float | None:
    duration = _positive_float(fmt.get("duration"))
    if duration is not None:
        return duration
    if video is None:
        return None
    return _positive_float(video.get("duration"))


def _fps(video: dict, duration: float | None) -> float | None:
    for key in ("avg_frame_rate", "r_frame_rate"):
        parsed = _ratio(video.get(key))
        if parsed is not None and parsed > 0:
            return parsed
    frames = _positive_float(video.get("nb_frames"))
    if frames is not None and duration is not None and duration > 0:
        return frames / duration
    return None


def _ratio(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "0/0", "N/A"}:
        return None
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            den = float(denominator)
            if den == 0:
                return None
            return float(numerator) / den
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _positive_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _positive_int(value: object) -> int | None:
    parsed = _positive_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _failed(path: Path, file_size: int, message: str) -> VideoMetadata:
    resolved = str(path.resolve()) if path.exists() else str(path)
    return VideoMetadata(path=resolved, file_size_bytes=file_size, probe_error=message)
