"""Extract metadata for one video. Filtering is applied later, on the whole batch."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from video_quality_filter.alignment import MODEL_NAME, cosine_similarity
from video_quality_filter.audio_quality import analyze_audio_file
from video_quality_filter.config import AppConfig
from video_quality_filter.embeddings import frame_histogram
from video_quality_filter.extract import extract_audio, extract_frames
from video_quality_filter.models import (
    empty_audio,
    empty_dedup,
    empty_filtering,
    empty_video_text,
    empty_visual,
    empty_vlm,
)
from video_quality_filter.probe import probe_video
from video_quality_filter.technical import build_technical
from video_quality_filter.visual_metrics import estimate_visual

logger = logging.getLogger(__name__)


def extract_record(
    path: Path,
    root: Path,
    video_id: str,
    config: AppConfig,
    *,
    qwen=None,
    aligner=None,
) -> tuple[dict, object]:
    metadata = probe_video(path, config.probe_timeout_sec)
    technical, technical_reasons = build_technical(metadata, config.filtering, config.target_fps)
    record = {
        "video_id": video_id,
        "source": _source(path, root),
        "technical": technical,
        "visual": empty_visual(),
        "vlm": empty_vlm(),
        "video_text": empty_video_text(),
        "audio": empty_audio(),
        "deduplication": empty_dedup(),
        "filtering": empty_filtering(),
        "_technical_reasons": technical_reasons,
        "_extract_errors": [],
    }
    embedding = None
    if metadata.probe_error or metadata.file_size_bytes <= 0 or metadata.duration_sec is None:
        return record, embedding
    if metadata.video_codec is None or metadata.width is None:
        return record, embedding

    heavy_ok = technical["technical_ok"] or config.run_heavy_on_technical_fail
    with tempfile.TemporaryDirectory(prefix="vqf_") as temporary:
        work = Path(temporary)
        frames, warnings = extract_frames(path, metadata.duration_sec, config.frames, work)
        for warning in warnings:
            logger.warning("%s: %s", video_id, warning)
        if frames:
            record["visual"] = _visual(frames, config, qwen, record["_extract_errors"])
            if config.dedup.embedding == "frame_histogram":
                embedding = frame_histogram(frames)
            if heavy_ok and config.vlm.backend == "qwen_vl" and qwen is not None:
                _vlm(frames, qwen, record)
        elif metadata.duration_sec is not None:
            record["_extract_errors"].append("frame_extraction_failed")

        use_internvideo = config.video_text.backend == "internvideo2" or config.dedup.embedding == "internvideo2"
        if heavy_ok and aligner is not None and use_internvideo:
            video_vector = _video_vector(path, aligner, record)
            if video_vector is not None and config.dedup.embedding == "internvideo2":
                embedding = video_vector
            caption = record["vlm"]["caption"]
            if video_vector is not None and config.video_text.backend == "internvideo2" and caption:
                _cosine(aligner, video_vector, caption, record)

        if metadata.has_audio:
            _audio(path, work, metadata.duration_sec, config, record)
    return record, embedding


def _visual(frames, config: AppConfig, qwen, errors: list[str]) -> dict:
    if config.visual.backend == "qwen_vl" and qwen is not None:
        try:
            return qwen.visual_metrics(frames)
        except Exception as exc:
            logger.warning("Визуальная оценка Qwen не удалась: %s", exc)
            errors.append("visual_extraction_failed")
            return empty_visual()
    return estimate_visual(frames)


def _vlm(frames, qwen, record: dict) -> None:
    try:
        record["vlm"] = qwen.caption_and_judge(frames)
    except Exception as exc:
        logger.warning("VLM pass не удался для %s: %s", record["video_id"], exc)
        record["_extract_errors"].append("vlm_failed")


def _video_vector(path: Path, aligner, record: dict):
    try:
        return aligner.video_embedding(path)
    except Exception as exc:
        logger.warning("Эмбеддинг InternVideo2 не удался для %s: %s", record["video_id"], exc)
        record["_extract_errors"].append("video_text_failed")
        return None


def _cosine(aligner, video_vector, caption: str, record: dict) -> None:
    try:
        text_vector = aligner.text_embedding(caption)
        record["video_text"]["model"] = MODEL_NAME
        record["video_text"]["cosine_similarity"] = round(cosine_similarity(video_vector, text_vector), 4)
    except Exception as exc:
        logger.warning("Cosine InternVideo2 не удался для %s: %s", record["video_id"], exc)
        if "video_text_failed" not in record["_extract_errors"]:
            record["_extract_errors"].append("video_text_failed")


def _audio(path: Path, work: Path, duration: float, config: AppConfig, record: dict) -> None:
    wav_path = work / "audio.wav"
    try:
        extract_audio(path, wav_path, duration, config.audio)
        record["audio"] = analyze_audio_file(wav_path, config.audio)
    except Exception as exc:
        logger.warning("Анализ аудио не удался для %s: %s", record["video_id"], exc)
        record["_extract_errors"].append("audio_analysis_failed")


def _source(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name
