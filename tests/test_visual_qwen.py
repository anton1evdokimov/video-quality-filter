import pytest

from video_quality_filter.config import QwenConfig
from video_quality_filter.visual_qwen import (
    _prepare_inputs_llava,
    load_qwen,
    parse_json_object,
    unit_score,
)


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


def test_llava_judge_passes_frames_as_separate_images():
    class Processor:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            images = [item for item in messages[0]["content"] if item["type"] == "image"]
            assert images == [{"type": "image"}, {"type": "image"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "prompt"

        def __call__(self, **kwargs):
            assert kwargs["text"] == ["prompt"]
            assert kwargs["images"] == ["frame-a", "frame-b"]
            assert kwargs["return_tensors"] == "pt"
            return {"input_ids": "ok"}

    result = _prepare_inputs_llava(Processor(), ["frame-a", "frame-b"], "score this")
    assert result["input_ids"] == "ok"


def test_qwen_backend_explains_missing_extra():
    with pytest.raises(RuntimeError, match=r"\.\[vlm\]"):
        load_qwen(QwenConfig())
