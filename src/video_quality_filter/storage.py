"""S3/MinIO for metadata and raw videos.

DVC keeps its content-addressed cache under ``<remote>/files/md5``. Video
objects use ``storage.videos_prefix`` and must not land in that cache.
"""

from __future__ import annotations

import configparser
import logging
import os
from pathlib import Path

from video_quality_filter.config import StorageConfig

logger = logging.getLogger(__name__)


def object_key(prefix: str, filename: str) -> str:
    cleaned = prefix.strip().strip("/")
    if not cleaned:
        return filename
    return f"{cleaned}/{filename}"


def upload_metadata(paths: list[Path], config: StorageConfig, *, repo_root: Path | None = None) -> list[str]:
    if not config.enabled:
        return []
    client = s3_client(config, repo_root)
    uris: list[str] = []
    for path in paths:
        key = object_key(config.prefix, path.name)
        client.upload_file(str(path), config.bucket, key)
        uris.append(f"s3://{config.bucket}/{key}")
    return uris


def upload_videos(
    root: Path,
    config: StorageConfig,
    extensions: list[str],
    *,
    recursive: bool = True,
    repo_root: Path | None = None,
) -> list[str]:
    if not config.enabled:
        raise ValueError("storage.enabled = false, загрузка видео в MinIO выключена")
    ensure_videos_prefix(config, repo_root)
    if not root.is_dir():
        raise FileNotFoundError(f"каталог с видео не найден: {root}")
    client = s3_client(config, repo_root)
    uris: list[str] = []
    for path in _video_files(root, extensions, recursive):
        relative = path.relative_to(root).as_posix()
        key = object_key(config.videos_prefix, relative)
        client.upload_file(str(path), config.bucket, key)
        uris.append(f"s3://{config.bucket}/{key}")
    if not uris:
        raise FileNotFoundError(f"в {root} нет видео с расширениями {', '.join(extensions)}")
    return uris


def download_videos(
    destination: Path,
    config: StorageConfig,
    extensions: list[str],
    *,
    repo_root: Path | None = None,
) -> list[Path]:
    if not config.enabled:
        raise ValueError("storage.enabled = false, читать видео из MinIO нельзя")
    ensure_videos_prefix(config, repo_root)
    destination.mkdir(parents=True, exist_ok=True)
    client = s3_client(config, repo_root)
    allowed = {item.lower().lstrip(".") for item in extensions}
    prefix = config.videos_prefix.strip().strip("/")
    list_prefix = f"{prefix}/"
    saved: list[Path] = []
    for key in _list_keys(client, config.bucket, list_prefix):
        relative = key[len(list_prefix) :]
        relative_path = Path(relative)
        if not relative or relative.endswith("/") or relative_path.is_absolute() or ".." in relative_path.parts:
            continue
        suffix = relative_path.suffix.lower().lstrip(".")
        if suffix not in allowed:
            continue
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(config.bucket, key, str(target))
        saved.append(target)
    if not saved:
        raise FileNotFoundError(f"в s3://{config.bucket}/{list_prefix} нет видео")
    return sorted(saved)


def ensure_videos_prefix(config: StorageConfig, repo_root: Path | None) -> None:
    videos = config.videos_prefix.strip().strip("/")
    if not videos:
        raise ValueError("storage.videos_prefix не должен быть пустым")
    remote = dvc_s3_remote(repo_root) if repo_root is not None else None
    cache_roots = ["files"]
    if remote is not None:
        if remote.bucket and remote.bucket != config.bucket:
            logger.warning(
                "storage.bucket=%s, бакет DVC remote s3=%s",
                config.bucket,
                remote.bucket,
            )
        if remote.prefix:
            cache_roots.append(f"{remote.prefix}/files")
    for cache_root in cache_roots:
        if videos == cache_root or videos.startswith(cache_root + "/"):
            raise ValueError(
                f"storage.videos_prefix {videos!r} попадает в кэш DVC ({cache_root}). "
                "Оставьте префикс videos: dvc push пишет только в files/md5."
            )


def s3_client(config: StorageConfig, repo_root: Path | None = None):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            'Загрузка в S3/MinIO требует boto3: pip install -e ".[storage]"'
        ) from exc
    remote = dvc_s3_remote(repo_root) if repo_root is not None else None
    endpoint = config.endpoint_url.strip() or (remote.endpoint_url if remote else "")
    kwargs: dict[str, str | None] = {
        "endpoint_url": endpoint or None,
        "region_name": config.region.strip() or None,
    }
    if remote and remote.access_key_id and not os.environ.get("AWS_ACCESS_KEY_ID"):
        kwargs["aws_access_key_id"] = remote.access_key_id
        kwargs["aws_secret_access_key"] = remote.secret_access_key
    return boto3.client("s3", **kwargs)


class DvcS3Remote:
    def __init__(
        self,
        bucket: str,
        prefix: str,
        endpoint_url: str = "",
        access_key_id: str = "",
        secret_access_key: str = "",
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key


def dvc_s3_remote(repo_root: Path | None) -> DvcS3Remote | None:
    if repo_root is None:
        return None
    dvc_dir = repo_root / ".dvc"
    config_path = dvc_dir / "config"
    if not config_path.is_file():
        return None
    parser = configparser.ConfigParser(interpolation=None)
    parser.read([config_path, dvc_dir / "config.local"], encoding="utf-8")
    section = _remote_section(parser, "s3")
    if section is None:
        return None
    url = parser.get(section, "url", fallback="").strip()
    bucket, prefix = _split_s3_url(url)
    if not bucket:
        return None
    return DvcS3Remote(
        bucket=bucket,
        prefix=prefix,
        endpoint_url=parser.get(section, "endpointurl", fallback="").strip(),
        access_key_id=parser.get(section, "access_key_id", fallback="").strip(),
        secret_access_key=parser.get(section, "secret_access_key", fallback="").strip(),
    )


def repo_root_from(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".dvc" / "config").is_file():
            return candidate
    return None


def _remote_section(parser: configparser.ConfigParser, name: str) -> str | None:
    needle = f'remote "{name}"'
    for section in parser.sections():
        if needle in section:
            return section
    return None


def _split_s3_url(url: str) -> tuple[str, str]:
    if not url.startswith("s3://"):
        return "", ""
    rest = url[len("s3://") :]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.strip("/")


def _video_files(root: Path, extensions: list[str], recursive: bool) -> list[Path]:
    allowed = {item.lower().lstrip(".") for item in extensions}
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower().lstrip(".") in allowed
    )


def _list_keys(client, bucket: str, prefix: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents") or []:
            key = item.get("Key") or ""
            if key:
                keys.append(key)
    return keys
