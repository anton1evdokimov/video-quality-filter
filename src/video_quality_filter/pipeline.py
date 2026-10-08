"""Feature extraction, dedup, then a separate filtering policy.

Source videos are only read. Sampled frames and audio excerpts live in a
temporary directory and are removed after each file.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from video_quality_filter.alignment import InternVideoAligner
from video_quality_filter.config import AppConfig, QwenConfig
from video_quality_filter.dedup import apply_dedup
from video_quality_filter.features import extract_record
from video_quality_filter.filtering import apply_policy
from video_quality_filter.report import write_reports
from video_quality_filter.storage import repo_root_from, upload_metadata
from video_quality_filter.visual_qwen import QwenClient

logger = logging.getLogger(__name__)

PUBLIC_KEYS = (
    "video_id",
    "source",
    "technical",
    "visual",
    "vlm",
    "video_text",
    "audio",
    "deduplication",
    "filtering",
)


@dataclass
class RunSummary:
    total: int
    accepted: int
    rejected: int
    errors: int
    jsonl_path: Path
    parquet_path: Path
    uploaded: list[str]


def run_pipeline(input_dir: Path, config: AppConfig, *, limit: int | None = None) -> RunSummary:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"каталог с видео не найден: {input_dir}")
    _require_tools()
    output_dir = Path(config.output_dir)
    _assert_output_separated(input_dir, output_dir)
    videos = discover_videos(input_dir, config.extensions, config.recursive, output_dir)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit должен быть >= 1")
        videos = videos[:limit]
    video_ids = _video_ids(videos, input_dir)

    qwen = None
    judge = None
    if config.visual.backend == "qwen_vl" or config.vlm.backend == "qwen_vl":
        qwen = QwenClient(config.qwen)
        qwen.warmup()
    if config.vlm.backend == "qwen_vl":
        judge = QwenClient(
            QwenConfig(
                model_id=config.vlm.judge_model_id,
                device=config.qwen.device,
                max_new_tokens=config.qwen.max_new_tokens,
            )
        )
        judge.warmup()
    aligner = None
    if config.video_text.backend == "internvideo2" or config.dedup.embedding == "internvideo2":
        aligner = InternVideoAligner(config.video_text)
        aligner.warmup()

    records = []
    embeddings = []
    for index, (path, video_id) in enumerate(zip(videos, video_ids), start=1):
        try:
            record, embedding = extract_record(
                path,
                input_dir,
                video_id,
                config,
                qwen=qwen,
                judge=judge,
                aligner=aligner,
            )
        except Exception as exc:
            logger.exception("Сбой на файле %s", path)
            record, embedding = _failed_record(path, input_dir, video_id, str(exc))
        records.append(record)
        embeddings.append(embedding)
        logger.info("[%d/%d] %s technical_ok=%s", index, len(videos), video_id, record["technical"]["technical_ok"])

    canonical = apply_dedup(records, embeddings, config.dedup)
    for record, canonical_id in zip(records, canonical):
        extract_errors = list(record.get("_extract_errors", []))
        apply_policy(
            record,
            config.filtering,
            canonical_video_id=canonical_id,
            extract_errors=extract_errors,
        )
        record.pop("_extract_errors", None)
        logger.info(
            "%s -> %s reasons=%s",
            record["video_id"],
            record["filtering"]["status"],
            ",".join(record["filtering"]["reasons"]) or "-",
        )

    public = [{key: record[key] for key in PUBLIC_KEYS} for record in records]
    jsonl_path, parquet_path = write_reports(output_dir, public)
    uploaded = upload_metadata(
        [jsonl_path, parquet_path],
        config.storage,
        repo_root=repo_root_from(),
    )
    errors = sum(1 for record in public if "processing_error" in record["filtering"]["reasons"])
    return RunSummary(
        total=len(public),
        accepted=sum(1 for record in public if record["filtering"]["status"] == "accepted"),
        rejected=sum(1 for record in public if record["filtering"]["status"] == "rejected"),
        errors=errors,
        jsonl_path=jsonl_path,
        parquet_path=parquet_path,
        uploaded=uploaded,
    )


def discover_videos(input_dir: Path, extensions: list[str], recursive: bool, output_dir: Path) -> list[Path]:
    allowed = {item.lower().lstrip(".") for item in extensions}
    output_resolved = output_dir.resolve()
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    found: list[Path] = []
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower().lstrip(".") not in allowed:
            continue
        if _is_inside(path, output_resolved):
            continue
        found.append(path)
    return sorted(found)


def _video_ids(paths: list[Path], root: Path) -> list[str]:
    stems = [path.stem for path in paths]
    used: set[str] = set()
    ids: list[str] = []
    for path, stem in zip(paths, stems):
        candidate = stem
        if stems.count(stem) > 1 or candidate in used:
            relative = path.resolve().relative_to(root.resolve()).with_suffix("")
            candidate = relative.as_posix().replace("/", "__")
        used.add(candidate)
        ids.append(candidate)
    return ids


def _failed_record(path: Path, root: Path, video_id: str, message: str) -> tuple[dict, None]:
    from video_quality_filter.models import (
        empty_audio,
        empty_dedup,
        empty_filtering,
        empty_video_text,
        empty_visual,
        empty_vlm,
    )

    try:
        source = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        source = path.name
    record = {
        "video_id": video_id,
        "source": source,
        "technical": {
            "duration": None,
            "fps_original": None,
            "fps_processed": None,
            "width": None,
            "height": None,
            "codec": None,
            "audio_present": False,
            "audio_duration": None,
            "av_duration_diff": None,
            "technical_ok": False,
        },
        "visual": empty_visual(),
        "vlm": empty_vlm(),
        "video_text": empty_video_text(),
        "audio": empty_audio(),
        "deduplication": empty_dedup(),
        "filtering": empty_filtering(),
        "_technical_reasons": ["unreadable"],
        "_extract_errors": ["processing_error"],
    }
    logger.error("%s: %s", video_id, message)
    return record, None


def _require_tools() -> None:
    missing = [tool for tool in ("ffprobe", "ffmpeg") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError("В PATH нет " + " и ".join(missing) + ". Установите FFmpeg.")


def _assert_output_separated(input_dir: Path, output_dir: Path) -> None:
    if _is_inside(input_dir, output_dir):
        raise ValueError(
            "Каталог отчётов не должен совпадать с каталогом видео и не должен быть его родителем"
        )


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
