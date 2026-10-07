from pathlib import Path

from video_quality_filter.config import StorageConfig
from video_quality_filter.storage import object_key, upload_metadata


def test_object_key_joins_the_metadata_prefix():
    assert object_key("metadata", "results.jsonl") == "metadata/results.jsonl"
    assert object_key("/", "results.parquet") == "results.parquet"


def test_upload_is_skipped_when_storage_is_disabled(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    assert upload_metadata([path], StorageConfig(enabled=False)) == []
