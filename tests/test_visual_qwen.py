import pytest

from video_quality_filter.config import QwenConfig
from video_quality_filter.visual_qwen import load_qwen, parse_json_object, unit_score


def test_parse_percent_scores_and_fenced_json():
    payload = parse_json_object(
        '```json\n{"aesthetic_score": 82, "watermark_probability": 0.03, "text_area_ratio": 4}\n```'
    )
    assert unit_score(payload["aesthetic_score"]) == pytest.approx(0.82)
    assert unit_score(payload["watermark_probability"]) == pytest.approx(0.03)
    assert unit_score(payload["text_area_ratio"]) == pytest.approx(0.04)


def test_parse_rejects_text_without_json():
    with pytest.raises(ValueError):
        parse_json_object("the frame looks fine")


def test_qwen_backend_explains_missing_extra():
    with pytest.raises(RuntimeError, match=r"\.\[vlm\]"):
        load_qwen(QwenConfig())
