"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from video_quality_filter.config import load_config, validate_config
from video_quality_filter.pipeline import run_pipeline


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
    run.add_argument("--input", required=True, type=Path, help="Каталог с видеофайлами")
    run.add_argument("--config", type=Path, help="YAML-конфиг. Без него используются значения по умолчанию")
    run.add_argument("--output-dir", type=Path, help="Куда писать results.jsonl и results.parquet")
    run.add_argument("--limit", type=int, help="Обработать только первые N файлов после сортировки")
    run.add_argument("-v", "--verbose", action="store_true", help="Подробные логи")
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
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"error: неизвестная команда {args.command}", file=sys.stderr)
    return 2


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    validate_config(config)
    summary = run_pipeline(args.input, config, limit=args.limit)
    print(
        f"Обработано {summary.total}: accepted {summary.accepted}, "
        f"rejected {summary.rejected}, ошибок {summary.errors}"
    )
    print(f"JSONL: {summary.jsonl_path}")
    print(f"Parquet: {summary.parquet_path}")
    for uri in summary.uploaded:
        print(f"S3: {uri}")
    return 0
