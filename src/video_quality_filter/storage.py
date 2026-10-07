"""Upload metadata objects to S3 or MinIO. Raw videos are never uploaded."""

from __future__ import annotations

from pathlib import Path

from video_quality_filter.config import StorageConfig


def object_key(prefix: str, filename: str) -> str:
    cleaned = prefix.strip().strip("/")
    if not cleaned:
        return filename
    return f"{cleaned}/{filename}"


def upload_metadata(paths: list[Path], config: StorageConfig) -> list[str]:
    if not config.enabled:
        return []
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            'Загрузка в S3/MinIO требует boto3: pip install -e ".[storage]"'
        ) from exc
    client = boto3.client(
        "s3",
        endpoint_url=config.endpoint_url.strip() or None,
        region_name=config.region.strip() or None,
    )
    uris: list[str] = []
    for path in paths:
        key = object_key(config.prefix, path.name)
        client.upload_file(str(path), config.bucket, key)
        uris.append(f"s3://{config.bucket}/{key}")
    return uris
