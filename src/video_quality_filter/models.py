"""Probe result and the public metadata record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VideoMetadata:
    path: str
    file_size_bytes: int
    duration_sec: float | None = None
    audio_duration_sec: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    has_audio: bool = False
    format_name: str | None = None
    bit_rate: int | None = None
    pixel_format: str | None = None
    probe_error: str | None = None


def empty_visual() -> dict[str, Any]:
    return {
        "aesthetic_score": None,
        "watermark_probability": None,
        "text_area_ratio": None,
    }


def empty_vlm() -> dict[str, Any]:
    return {
        "caption": None,
        "judge_model": None,
        "judge_caption": None,
        "semantic_consistency": None,
        "temporal_coverage": None,
        "completeness": None,
        "hallucination": None,
    }


def empty_video_text() -> dict[str, Any]:
    return {"model": None, "cosine_similarity": None}


def empty_audio() -> dict[str, Any]:
    return {
        "silence_ratio": None,
        "rms": None,
        "clipping_ratio": None,
        "audio_quality": None,
    }


def empty_dedup() -> dict[str, Any]:
    return {
        "cluster_id": None,
        "nearest_video_id": None,
        "similarity": None,
        "is_near_duplicate": False,
    }


def empty_filtering() -> dict[str, Any]:
    return {"status": "rejected", "reasons": []}
