"""YAML configuration. Filtering thresholds live apart from feature extraction."""

from __future__ import annotations

import types
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass
class FramesConfig:
    count: int = 4
    max_side: int = 512
    timeout_sec: float = 30.0


@dataclass
class VisualConfig:
    backend: str = "heuristic"


@dataclass
class QwenConfig:
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "auto"
    max_new_tokens: int = 256


@dataclass
class VlmConfig:
    backend: str = "off"
    judge_model_id: str = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"


@dataclass
class VideoTextConfig:
    backend: str = "off"
    model_id: str = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
    device: str = "auto"
    num_frames: int = 4
    replace_words: dict[str, str] = field(default_factory=dict)


@dataclass
class DedupConfig:
    embedding: str = "frame_histogram"
    cluster_similarity: float = 0.92
    near_duplicate_similarity: float = 0.98


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    max_analyze_sec: float = 180.0
    silence_threshold_db: float = -40.0
    extract_timeout_sec: float = 120.0
    weight_silence: float = 0.45
    weight_energy: float = 0.30
    weight_clipping: float = 0.25


@dataclass
class FilteringConfig:
    min_duration: float = 1.0
    max_duration: float = 7200.0
    min_fps: float = 8.0
    min_width: int = 256
    min_height: int = 256
    allowed_codecs: list[str] = field(
        default_factory=lambda: ["h264", "hevc", "vp9", "av1", "mpeg4"]
    )
    max_av_duration_diff: float | None = 1.0
    require_audio: bool = False
    reject_unknown_fps: bool = True
    min_aesthetic_score: float | None = 0.2
    max_watermark_probability: float | None = 0.95
    max_text_area_ratio: float | None = 0.85
    min_semantic_consistency: float | None = None
    min_temporal_coverage: float | None = None
    min_completeness: float | None = None
    max_hallucination: float | None = None
    min_video_text_cosine: float | None = None
    max_silence_ratio: float | None = 0.95
    min_rms: float | None = None
    max_clipping_ratio: float | None = 0.02
    reject_near_duplicates: bool = False
    require_visual: bool = False
    require_vlm: bool = False
    require_video_text: bool = False


@dataclass
class StorageConfig:
    enabled: bool = False
    bucket: str = ""
    prefix: str = "metadata"
    endpoint_url: str = ""
    region: str = ""


@dataclass
class AppConfig:
    extensions: list[str] = field(
        default_factory=lambda: ["mp4", "mov", "mkv", "webm", "avi", "m4v"]
    )
    recursive: bool = True
    output_dir: str = "reports"
    target_fps: float = 24.0
    probe_timeout_sec: float = 30.0
    run_heavy_on_technical_fail: bool = False
    frames: FramesConfig = field(default_factory=FramesConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    vlm: VlmConfig = field(default_factory=VlmConfig)
    video_text: VideoTextConfig = field(default_factory=VideoTextConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


def load_config(path: Path | None = None) -> AppConfig:
    data: dict[str, Any] = {}
    if path is not None:
        with path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Корень конфига {path} должен быть mapping")
        data = loaded
    config = _build(AppConfig, data, "config")
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    config.extensions = [item.lower().lstrip(".") for item in config.extensions if str(item).strip()]
    if not config.extensions:
        raise ValueError("extensions не должен быть пустым")
    if config.target_fps <= 0:
        raise ValueError("target_fps должен быть > 0")
    if config.probe_timeout_sec <= 0:
        raise ValueError("probe_timeout_sec должен быть > 0")
    if config.visual.backend not in {"heuristic", "qwen_vl"}:
        raise ValueError("visual.backend должен быть heuristic или qwen_vl")
    if config.vlm.backend not in {"off", "qwen_vl"}:
        raise ValueError("vlm.backend должен быть off или qwen_vl")
    if config.vlm.backend == "qwen_vl" and config.vlm.judge_model_id.strip() == config.qwen.model_id.strip():
        raise ValueError("vlm.judge_model_id должен отличаться от qwen.model_id")
    if not config.vlm.judge_model_id.strip():
        raise ValueError("vlm.judge_model_id не должен быть пустым")
    if config.video_text.backend not in {"off", "internvideo2"}:
        raise ValueError("video_text.backend должен быть off или internvideo2")
    if config.dedup.embedding not in {"frame_histogram", "internvideo2"}:
        raise ValueError("dedup.embedding должен быть frame_histogram или internvideo2")
    if config.qwen.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("qwen.device должен быть auto, cpu, cuda или mps")
    if config.video_text.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("video_text.device должен быть auto, cpu, cuda или mps")
    if config.qwen.max_new_tokens < 16:
        raise ValueError("qwen.max_new_tokens должен быть >= 16")
    if not 1 <= config.video_text.num_frames <= 16:
        raise ValueError("video_text.num_frames должен быть в диапазоне 1..16")
    for source in config.video_text.replace_words:
        if not source.strip():
            raise ValueError("video_text.replace_words: пустое слово заменять нельзя")
    if not -1.0 <= config.dedup.cluster_similarity <= 1.0:
        raise ValueError("dedup.cluster_similarity должен быть в диапазоне -1..1")
    if not -1.0 <= config.dedup.near_duplicate_similarity <= 1.0:
        raise ValueError("dedup.near_duplicate_similarity должен быть в диапазоне -1..1")

    frames = config.frames
    if not 1 <= frames.count <= 64:
        raise ValueError("frames.count должен быть в диапазоне 1..64")
    if not 32 <= frames.max_side <= 2048:
        raise ValueError("frames.max_side должен быть в диапазоне 32..2048")
    if frames.timeout_sec <= 0:
        raise ValueError("frames.timeout_sec должен быть > 0")

    audio = config.audio
    if audio.sample_rate < 8000:
        raise ValueError("audio.sample_rate должен быть >= 8000")
    if audio.max_analyze_sec <= 0 or audio.extract_timeout_sec <= 0:
        raise ValueError("audio.max_analyze_sec и extract_timeout_sec должны быть > 0")
    _require_weights(
        "audio",
        [audio.weight_silence, audio.weight_energy, audio.weight_clipping],
    )

    policy = config.filtering
    policy.allowed_codecs = [item.lower() for item in policy.allowed_codecs]
    if policy.max_duration <= policy.min_duration:
        raise ValueError("filtering.max_duration должен быть больше min_duration")
    if policy.min_duration < 0:
        raise ValueError("filtering.min_duration должен быть >= 0")
    if policy.min_fps <= 0:
        raise ValueError("filtering.min_fps должен быть > 0")
    if policy.min_width < 1 or policy.min_height < 1:
        raise ValueError("filtering.min_width и min_height должны быть >= 1")
    _optional_unit("filtering.max_av_duration_diff", policy.max_av_duration_diff, low=0.0, high=None)
    for name, value, low, high in (
        ("min_aesthetic_score", policy.min_aesthetic_score, 0.0, 1.0),
        ("max_watermark_probability", policy.max_watermark_probability, 0.0, 1.0),
        ("max_text_area_ratio", policy.max_text_area_ratio, 0.0, 1.0),
        ("min_semantic_consistency", policy.min_semantic_consistency, 0.0, 1.0),
        ("min_temporal_coverage", policy.min_temporal_coverage, 0.0, 1.0),
        ("min_completeness", policy.min_completeness, 0.0, 1.0),
        ("max_hallucination", policy.max_hallucination, 0.0, 1.0),
        ("min_video_text_cosine", policy.min_video_text_cosine, -1.0, 1.0),
        ("max_silence_ratio", policy.max_silence_ratio, 0.0, 1.0),
        ("min_rms", policy.min_rms, 0.0, None),
        ("max_clipping_ratio", policy.max_clipping_ratio, 0.0, 1.0),
    ):
        _optional_unit(f"filtering.{name}", value, low=low, high=high)
    if config.storage.enabled and not config.storage.bucket.strip():
        raise ValueError("storage.bucket обязателен, когда storage.enabled = true")


def _optional_unit(name: str, value: float | None, *, low: float, high: float | None) -> None:
    if value is None:
        return
    if value < low or (high is not None and value > high):
        upper = "" if high is None else f"..{high}"
        raise ValueError(f"{name} должен быть в диапазоне {low}{upper}")


def _require_weights(label: str, weights: list[float]) -> None:
    if any(weight < 0 for weight in weights):
        raise ValueError(f"веса {label} должны быть >= 0")
    if sum(weights) <= 0:
        raise ValueError(f"хотя бы один вес {label} должен быть > 0")


def _build(cls: type[Any], data: dict[str, Any], prefix: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{prefix} должен быть mapping")
    hints = get_type_hints(cls)
    unknown = sorted(set(data) - set(hints))
    if unknown:
        raise ValueError(f"Неизвестные ключи {prefix}: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, kind in hints.items():
        if name not in data:
            continue
        value = data[name]
        key = f"{prefix}.{name}"
        optional = _optional_inner(kind)
        if optional is not None:
            if value is None:
                kwargs[name] = None
            else:
                kwargs[name] = _coerce(optional, value, key)
            continue
        if is_dataclass(kind):
            if not isinstance(value, dict):
                raise ValueError(f"{key} должен быть mapping")
            kwargs[name] = _build(kind, value, key)
        else:
            kwargs[name] = _coerce(kind, value, key)
    return cls(**kwargs)


def _optional_inner(kind: Any) -> Any | None:
    origin = get_origin(kind)
    if origin not in (Union, types.UnionType):
        return None
    args = get_args(kind)
    non_none = [arg for arg in args if arg is not type(None)]
    if len(non_none) == 1 and type(None) in args:
        return non_none[0]
    return None


def _coerce(kind: Any, value: Any, key: str) -> Any:
    origin = get_origin(kind)
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{key} должен быть списком")
        return list(value)
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{key} должен быть mapping")
        key_type, value_type = get_args(kind)
        return {
            _coerce(key_type, item_key, key): _coerce(value_type, item_value, key)
            for item_key, item_value in value.items()
        }
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} должен быть числом")
        return float(value)
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} должен быть целым числом")
        return int(value)
    if kind is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{key} должен быть true или false")
        return value
    if kind is str:
        if not isinstance(value, str):
            raise ValueError(f"{key} должен быть строкой")
        return value
    raise ValueError(f"Неподдерживаемый тип поля {key}")
