from dataclasses import asdict
from pathlib import Path

import pytest

from video_quality_filter.config import AppConfig, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_code_defaults():
    loaded = load_config(ROOT / "config" / "default.yaml")
    assert asdict(loaded) == asdict(AppConfig())


def test_partial_qwen_config_keeps_other_defaults():
    loaded = load_config(ROOT / "config" / "qwen_vl.yaml")
    assert loaded.visual.backend == "qwen_vl"
    assert loaded.vlm.backend == "qwen_vl"
    assert loaded.video_text.backend == "internvideo2"
    assert loaded.dedup.embedding == "internvideo2"
    assert loaded.filtering.min_video_text_cosine == 0.2
    assert loaded.filtering.min_duration == AppConfig().filtering.min_duration
    assert loaded.target_fps == 24.0


def test_unknown_key_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("target_fps: 24\nnot_a_real_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_real_key"):
        load_config(path)


def test_disabled_threshold_must_be_null_not_a_string(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("filtering:\n  min_video_text_cosine: none\n", encoding="utf-8")
    with pytest.raises(ValueError, match="min_video_text_cosine"):
        load_config(path)


def test_storage_requires_a_bucket_when_enabled():
    config = AppConfig()
    config.storage.enabled = True
    with pytest.raises(ValueError, match="bucket"):
        validate_config(config)
