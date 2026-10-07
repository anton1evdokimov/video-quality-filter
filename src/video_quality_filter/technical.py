"""Technical metadata and the rules that set technical_ok.

High frame rates are not rejected: fps_processed is the rate after downsampling
to target_fps. Low frame rates stay at the original rate and can be rejected.
"""

from __future__ import annotations

from video_quality_filter.config import FilteringConfig
from video_quality_filter.models import VideoMetadata

CODEC_ALIASES = {"h265": "hevc", "avc": "h264", "avc1": "h264"}


def processed_fps(original: float | None, target_fps: float) -> float | None:
    if original is None:
        return None
    if original > target_fps:
        return float(target_fps)
    return float(original)


def build_technical(
    metadata: VideoMetadata,
    policy: FilteringConfig,
    target_fps: float,
) -> tuple[dict, list[str]]:
    reasons = technical_reasons(metadata, policy)
    duration = metadata.duration_sec
    audio_duration = metadata.audio_duration_sec if metadata.has_audio else None
    diff = None
    if duration is not None and audio_duration is not None:
        diff = abs(duration - audio_duration)
    technical = {
        "duration": _round(duration, 3),
        "fps_original": _round(metadata.fps, 3),
        "fps_processed": _round(processed_fps(metadata.fps, target_fps), 3),
        "width": metadata.width,
        "height": metadata.height,
        "codec": _canonical_codec(metadata.video_codec),
        "audio_present": bool(metadata.has_audio),
        "audio_duration": _round(audio_duration, 3),
        "av_duration_diff": _round(diff, 3),
        "technical_ok": not reasons,
    }
    return technical, reasons


def technical_reasons(metadata: VideoMetadata, policy: FilteringConfig) -> list[str]:
    reasons: list[str] = []
    if metadata.file_size_bytes <= 0 or metadata.probe_error:
        reasons.append("unreadable")
        return reasons
    if metadata.video_codec is None or metadata.width is None or metadata.height is None:
        reasons.append("no_video_stream")
        return reasons

    if metadata.duration_sec is None:
        reasons.append("duration_unknown")
    elif metadata.duration_sec < policy.min_duration:
        reasons.append("duration_too_short")
    elif metadata.duration_sec > policy.max_duration:
        reasons.append("duration_too_long")

    if metadata.width < policy.min_width or metadata.height < policy.min_height:
        reasons.append("resolution_too_small")

    if metadata.fps is None:
        if policy.reject_unknown_fps:
            reasons.append("fps_unknown")
    elif metadata.fps < policy.min_fps:
        reasons.append("fps_too_low")

    codec = _canonical_codec(metadata.video_codec)
    allowed = {_canonical_codec(item) or item for item in policy.allowed_codecs}
    if codec is None:
        reasons.append("codec_invalid")
    elif allowed and codec not in allowed:
        reasons.append("codec_not_allowed")

    if policy.require_audio and not metadata.has_audio:
        reasons.append("audio_required_but_missing")

    if (
        policy.max_av_duration_diff is not None
        and metadata.has_audio
        and metadata.duration_sec is not None
        and metadata.audio_duration_sec is not None
        and abs(metadata.duration_sec - metadata.audio_duration_sec) > policy.max_av_duration_diff
    ):
        reasons.append("av_duration_mismatch")
    return reasons


def _canonical_codec(codec: str | None) -> str | None:
    if codec is None:
        return None
    lowered = codec.lower()
    return CODEC_ALIASES.get(lowered, lowered)


def _round(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)
