from dataclasses import asdict
from pathlib import Path

import pytest

from video_quality_filter.config import AppConfig, load_config, validate_config
from video_quality_filter.features import replace_caption_words

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_code_defaults():
    loaded = load_config(ROOT / "config" / "default.yaml")
    assert asdict(loaded) == asdict(AppConfig())


def test_minio_config_enables_storage_and_vlm():
    loaded = load_config(ROOT / "config" / "minio.yaml")
    assert loaded.storage.enabled is True
    assert loaded.storage.bucket == "datasets"
    assert loaded.storage.videos_prefix == "videos"
    assert loaded.visual.backend == "qwen_vl"
    assert loaded.vlm.backend == "qwen_vl"
    assert loaded.vlm.judge_model_id == "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
    assert loaded.filtering.require_vlm is True
    assert loaded.video_text.backend == "off"


def test_partial_qwen_config_keeps_other_defaults():
    loaded = load_config(ROOT / "config" / "qwen_vl.yaml")
    assert loaded.visual.backend == "qwen_vl"
    assert loaded.vlm.backend == "qwen_vl"
    assert loaded.vlm.judge_model_id == "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
    assert loaded.vlm.judge_model_id != loaded.qwen.model_id
    assert loaded.video_text.backend == "internvideo2"
    assert loaded.dedup.embedding == "internvideo2"
    assert loaded.content.backend == "qwen_vl"
    assert loaded.content.labels == AppConfig().content.labels
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


def test_replace_words_keeps_whole_words_only(tmp_path: Path):
    path = tmp_path / "swap.yaml"
    path.write_text("video_text:\n  replace_words:\n    man: hand\n", encoding="utf-8")
    loaded = load_config(path)
    assert loaded.video_text.replace_words == {"man": "hand"}
    assert replace_caption_words("a man of photo", {"man": "hand"}) == "a hand of photo"
    assert replace_caption_words("manual woman", {"man": "hand"}) == "manual woman"


def test_judge_must_differ_from_caption_model():
    config = AppConfig()
    config.vlm.backend = "qwen_vl"
    config.vlm.judge_model_id = config.qwen.model_id
    with pytest.raises(ValueError, match="judge_model_id"):
        validate_config(config)


def test_storage_requires_a_bucket_when_enabled():
    config = AppConfig()
    config.storage.enabled = True
    with pytest.raises(ValueError, match="bucket"):
        validate_config(config)


def test_reading_videos_requires_storage():
    config = AppConfig()
    config.storage.read_videos = True
    with pytest.raises(ValueError, match="read_videos"):
        validate_config(config)
