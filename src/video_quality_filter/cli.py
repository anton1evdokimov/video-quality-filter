"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

from video_quality_filter.config import load_config, validate_config
from video_quality_filter.pipeline import run_pipeline
from video_quality_filter.storage import download_videos, repo_root_from, upload_videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-quality-filter",
        description=(
            "Снимает технические, визуальные, VLM, alignment и аудио-метрики видео "
            "и отдельно применяет политику фильтрации. Исходные файлы не удаляет."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Извлечь metadata и записать JSONL и Parquet")
    run.add_argument("--input", type=Path, help="Локальный каталог с видео. Не нужен вместе с --from-storage")
    run.add_argument(
        "--from-storage",
        action="store_true",
        help="Скачать видео из MinIO одним tar (или пофайлово) и обработать их",
    )
    run.add_argument("--config", type=Path, help="YAML-конфиг. Без него используются значения по умолчанию")
    run.add_argument("--output-dir", type=Path, help="Куда писать results.jsonl и results.parquet")
    run.add_argument("--limit", type=int, help="Обработать только первые N файлов после сортировки")
    run.add_argument("-v", "--verbose", action="store_true", help="Подробные логи")
    upload = sub.add_parser("upload", help="Загрузить видео из каталога в MinIO")
    upload.add_argument("--input", required=True, type=Path, help="Каталог с видео, обычно data/raw")
    upload.add_argument("--config", type=Path, help="YAML-конфиг. Без него используются значения по умолчанию")
    upload.add_argument("-v", "--verbose", action="store_true", help="Подробные логи")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(message)s",
        force=True,
    )
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "upload":
            return _cmd_upload(args)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"error: неизвестная команда {args.command}", file=sys.stderr)
    return 2


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    if args.from_storage:
        config.storage.read_videos = True
    validate_config(config)
    if config.storage.read_videos and args.input is not None:
        raise ValueError("с --from-storage каталог --input не используется")
    if not config.storage.read_videos and args.input is None:
        raise ValueError("нужен --input или --from-storage")

    temporary = None
    input_dir = args.input
    try:
        if config.storage.read_videos:
            temporary = tempfile.TemporaryDirectory(prefix="vqf-videos-")
            input_dir = Path(temporary.name)
            download_videos(
                input_dir,
                config.storage,
                config.extensions,
                repo_root=repo_root_from(),
            )
        summary = run_pipeline(input_dir, config, limit=args.limit)
    finally:
        if temporary is not None:
            temporary.cleanup()
    print(
        f"Обработано {summary.total}: accepted {summary.accepted}, "
        f"rejected {summary.rejected}, ошибок {summary.errors}"
    )
    print(f"JSONL: {summary.jsonl_path}")
    print(f"Parquet: {summary.parquet_path}")
    for uri in summary.uploaded:
        print(f"S3: {uri}")
    return 0


def _cmd_upload(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if not config.storage.enabled:
        raise ValueError("storage.enabled = false, загрузка видео в MinIO выключена")
    uris = upload_videos(
        args.input,
        config.storage,
        config.extensions,
        recursive=config.recursive,
        repo_root=repo_root_from(),
    )
    print(f"Загружено {len(uris)}")
    for uri in uris:
        print(f"S3: {uri}")
    return 0
