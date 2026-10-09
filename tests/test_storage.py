import logging
from pathlib import Path

import pytest

from video_quality_filter.cli import main
from video_quality_filter.config import StorageConfig
from video_quality_filter.storage import (
    download_videos,
    dvc_s3_remote,
    ensure_videos_prefix,
    object_key,
    upload_metadata,
    upload_videos,
)

def _dvc_repo(tmp_path: Path, url: str = "s3://datasets/video-quality-filter") -> Path:
    dvc_dir = tmp_path / ".dvc"
    dvc_dir.mkdir()
    (dvc_dir / "config").write_text(f"['remote \"s3\"']\n    url = {url}\n", encoding="utf-8")
    return tmp_path


def test_object_key_joins_the_metadata_prefix():
    assert object_key("metadata", "results.jsonl") == "metadata/results.jsonl"
    assert object_key("/", "results.parquet") == "results.parquet"


def test_upload_is_skipped_when_storage_is_disabled(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    assert upload_metadata([path], StorageConfig(enabled=False)) == []


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        target = Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.objects[(bucket, key)])

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket: str, Prefix: str):
        contents = [
            {"Key": key}
            for bucket, key in self.objects
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return [{"Contents": contents}]


def test_videos_roundtrip_keeps_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "raw"
    nested = source / "clips"
    nested.mkdir(parents=True)
    (nested / "a.mp4").write_bytes(b"video-a")
    (source / "note.txt").write_text("skip", encoding="utf-8")
    store = _FakeS3()
    monkeypatch.setattr("video_quality_filter.storage.s3_client", lambda config, repo_root=None: store)
    config = StorageConfig(enabled=True, bucket="datasets", videos_prefix="videos")

    uris = upload_videos(source, config, ["mp4"], repo_root=tmp_path)

    assert uris == ["s3://datasets/videos/videos.tar"]
    assert list(store.objects) == [("datasets", "videos/videos.tar")]
    restored = tmp_path / "restored"
    paths = download_videos(restored, config, ["mp4"], repo_root=tmp_path)
    assert paths == [restored / "clips" / "a.mp4"]
    assert paths[0].read_bytes() == b"video-a"
    assert not (restored / "videos.tar").exists()


def test_file_layout_uploads_each_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "a.mp4").write_bytes(b"video-a")
    store = _FakeS3()
    monkeypatch.setattr("video_quality_filter.storage.s3_client", lambda config, repo_root=None: store)
    config = StorageConfig(enabled=True, bucket="datasets", videos_prefix="videos", archive="files")
    uris = upload_videos(source, config, ["mp4"], repo_root=tmp_path)
    assert uris == ["s3://datasets/videos/a.mp4"]


def test_tar_rejects_a_path_that_escapes_the_destination(tmp_path: Path):
    from video_quality_filter.storage import _extract_tar

    archive_path = tmp_path / "videos.tar"
    import tarfile

    with tarfile.open(archive_path, "w") as archive:
        payload = tmp_path / "payload.mp4"
        payload.write_bytes(b"nope")
        archive.add(payload, arcname="../outside.mp4")
    with pytest.raises(ValueError, match="недопустимый путь"):
        _extract_tar(archive_path, tmp_path / "out", {"mp4"})


def test_videos_prefix_must_stay_outside_the_dvc_cache(tmp_path: Path):
    repo = _dvc_repo(tmp_path)
    config = StorageConfig(enabled=True, bucket="datasets", videos_prefix="videos")
    ensure_videos_prefix(config, repo)
    blocked = StorageConfig(
        enabled=True,
        bucket="datasets",
        videos_prefix="video-quality-filter/files/md5",
    )
    with pytest.raises(ValueError, match="кэш DVC"):
        ensure_videos_prefix(blocked, repo)


def test_dvc_remote_endpoint_overrides_empty_app_endpoint(tmp_path: Path):
    dvc_dir = tmp_path / ".dvc"
    dvc_dir.mkdir()
    (dvc_dir / "config").write_text(
        "['remote \"s3\"']\n    url = s3://datasets/video-quality-filter\n",
        encoding="utf-8",
    )
    (dvc_dir / "config.local").write_text(
        "\n".join(
            [
                "['remote \"s3\"']",
                "    url = s3://datasets/video-quality-filter",
                "    endpointurl = http://127.0.0.1:9000",
                "    access_key_id = user",
                "    secret_access_key = secret",
                "",
            ]
        ),
        encoding="utf-8",
    )
    remote = dvc_s3_remote(tmp_path)
    assert remote is not None
    assert remote.bucket == "datasets"
    assert remote.prefix == "video-quality-filter"
    assert remote.endpoint_url == "http://127.0.0.1:9000"
    assert remote.access_key_id == "user"


def test_different_buckets_are_reported(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    repo = _dvc_repo(tmp_path)
    config = StorageConfig(enabled=True, bucket="other", videos_prefix="videos")
    with caplog.at_level(logging.WARNING):
        ensure_videos_prefix(config, repo)
    assert "other" in caplog.text
    assert "datasets" in caplog.text


def test_cli_requires_an_input_or_storage():
    assert main(["run"]) == 2


def test_cli_upload_is_rejected_when_storage_is_disabled(tmp_path: Path):
    assert main(["upload", "--input", str(tmp_path)]) == 2
