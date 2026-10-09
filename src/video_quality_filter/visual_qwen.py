"""Qwen-VL passes: per-frame visual metrics, then caption, then a separate judge."""

from __future__ import annotations

import json
import logging

from PIL import Image

from video_quality_filter.config import QwenConfig
from video_quality_filter.extract import ExtractedFrame
from video_quality_filter.models import empty_visual, empty_vlm

logger = logging.getLogger(__name__)

VISUAL_PROMPT = (
    "Rate this video frame for dataset filtering. Return JSON only, no markdown: "
    '{"aesthetic_score": <0-1>, "watermark_probability": <0-1>, "text_area_ratio": <0-1>}. '
    "aesthetic_score is visual appeal, not how interesting the subject is. "
    "watermark_probability is the chance of a logo, watermark, or channel bug. "
    "text_area_ratio is the fraction of the frame covered by text or subtitles."
)

CAPTION_PROMPT = (
    "These frames are in chronological order from one video. "
    "Describe what happens, in one or two sentences. No preamble."
)

JUDGE_CAPTION_PROMPT = (
    "These frames are in chronological order from one video. "
    "Write exactly two sentences. "
    "First sentence: the objects and how they are arranged at the start. "
    "Second sentence: what enters, moves, or changes later in the sequence. "
    "Name the visible things. Do not stop at a short phrase. No preamble."
)

CLASSIFY_PROMPT = (
    "These frames are in chronological order from one video. "
    "Choose exactly one content label from: {labels}. "
    "person: a person is visible, not only a hand. "
    "hands: a hand is the subject and a full person is not visible. "
    "object: a device or object is the subject, without a person or a hand as the subject. "
    "screen: a display, user interface, or recorded screen. "
    "scene: a place or landscape. "
    "text: the frames are mainly text or titles. "
    "animation: drawn or synthetic animation. "
    "other: none of the labels fit. "
    'Return JSON only, no markdown: {{"label": "<one label>"}}.'
)

JUDGE_PROMPT = (
    "The frames are in chronological order. The caption below was written for this video.\n"
    "Caption: {caption}\n"
    "Score the caption against the frames. Return JSON only, no markdown: "
    '{{"semantic_consistency": <0-1>, "temporal_coverage": <0-1>, '
    '"completeness": <0-1>, "hallucination": <0-1>}}. '
    "semantic_consistency: the caption matches what is visible. "
    "temporal_coverage: the caption follows the sequence, not a single frame. "
    "completeness: important visible events are included. "
    "hallucination: share of the caption that the frames do not support."
)


class QwenClient:
    def __init__(self, config: QwenConfig) -> None:
        self.config = config
        self._model = None
        self._processor = None

    def warmup(self) -> None:
        self._ensure()

    def visual_metrics(self, frames: list[ExtractedFrame]) -> dict:
        scores = []
        for frame in frames:
            text = self._generate([Image.fromarray(frame.image).convert("RGB")], VISUAL_PROMPT)
            payload = parse_json_object(text)
            scores.append(
                {
                    "aesthetic_score": unit_score(payload["aesthetic_score"]),
                    "watermark_probability": unit_score(payload["watermark_probability"]),
                    "text_area_ratio": unit_score(payload["text_area_ratio"]),
                }
            )
        if not scores:
            return empty_visual()
        return {
            key: round(sum(item[key] for item in scores) / len(scores), 4)
            for key in ("aesthetic_score", "watermark_probability", "text_area_ratio")
        }

    def classify(self, frames: list[ExtractedFrame], labels: list[str]) -> str:
        images = [Image.fromarray(frame.image).convert("RGB") for frame in frames]
        text = self._generate(images, CLASSIFY_PROMPT.format(labels=", ".join(labels)))
        payload = parse_json_object(text)
        if "label" not in payload:
            raise ValueError("в ответе классификации нет label")
        return str(payload["label"])

    def caption(self, frames: list[ExtractedFrame]) -> str:
        images = [Image.fromarray(frame.image).convert("RGB") for frame in frames]
        caption = " ".join(self._generate(images, CAPTION_PROMPT).split())
        if not caption:
            raise ValueError("VLM вернул пустой caption")
        return caption

    def judge(self, frames: list[ExtractedFrame], caption: str) -> dict:
        images = [Image.fromarray(frame.image).convert("RGB") for frame in frames]
        judge_caption = " ".join(self._generate(images, JUDGE_CAPTION_PROMPT).split())
        if not judge_caption:
            raise ValueError("судья вернул пустой caption")
        judged = self._generate(images, JUDGE_PROMPT.format(caption=caption))
        payload = parse_json_object(judged)
        result = empty_vlm()
        result["caption"] = caption
        result["judge_model"] = self.config.model_id
        result["judge_caption"] = judge_caption
        for key in ("semantic_consistency", "temporal_coverage", "completeness", "hallucination"):
            result[key] = round(unit_score(payload[key]), 4)
        return result

    def _generate(self, images: list[Image.Image], prompt: str) -> str:
        model, processor = self._ensure()
        inputs = _prepare_inputs(processor, images, prompt, self.config.model_id)
        device = next(model.parameters()).device
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        import torch

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = generated[:, input_len:] if generated.shape[-1] > input_len else generated
        return _decode(processor, new_tokens)

    def _ensure(self):
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        logger.info("Загружаю VLM %s", self.config.model_id)
        self._model, self._processor = load_qwen(self.config)
        return self._model, self._processor


def load_qwen(config: QwenConfig):
    try:
        import torch
        import transformers
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            'Бэкенд qwen_vl требует дополнительные пакеты: pip install -e ".[vlm]"'
        ) from exc
    if _is_llava_onevision(config.model_id):
        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
        if (major, minor) < (5, 7):
            raise RuntimeError(
                "LLaVA-OneVision-2 требует transformers>=5.7. "
                f"Сейчас установлен {transformers.__version__}."
            )
    dtype, device_map, move_to = _placement(torch, config.device)
    model = _from_pretrained(config.model_id, dtype=dtype, device_map=device_map)
    if move_to is not None:
        model = model.to(move_to)
    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    model.eval()
    return model, processor


def parse_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"в ответе VLM нет JSON: {text.strip()[:180]}")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON ответа VLM не разобран: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON ответа VLM должен быть объектом")
    return payload


def unit_score(value: object) -> float:
    score = float(value)
    if score > 1.0:
        score = score / 100.0
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"оценка вне диапазона 0..1: {value}")
    return score


def _placement(torch, device: str):
    if device == "cpu":
        return torch.float32, "cpu", None
    if device == "cuda":
        return torch.bfloat16, "cuda", None
    if device == "mps":
        return torch.float32, None, "mps"
    if torch.cuda.is_available():
        return torch.bfloat16, "auto", None
    return torch.float32, "cpu", None


def _from_pretrained(model_id: str, dtype, device_map: str | None):
    import transformers

    if hasattr(transformers, "AutoModelForImageTextToText"):
        model_cls = transformers.AutoModelForImageTextToText
    elif hasattr(transformers, "AutoModelForVision2Seq"):
        model_cls = transformers.AutoModelForVision2Seq
    else:
        raise RuntimeError("Нужен transformers>=4.57: в нём есть Qwen3-VL")
    kwargs = {"trust_remote_code": True}
    if device_map is not None:
        kwargs["device_map"] = device_map
    try:
        return model_cls.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        return model_cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs)


def _is_llava_onevision(model_id: str) -> bool:
    return "llava-onevision" in model_id.lower()


def _decode(processor, tokens) -> str:
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    return processor.tokenizer.decode(tokens[0], skip_special_tokens=True).strip()


def _prepare_inputs(processor, images: list[Image.Image], prompt: str, model_id: str):
    if _is_llava_onevision(model_id):
        return _prepare_inputs_llava(processor, images, prompt)
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    try:
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except (TypeError, ValueError):
        return _prepare_inputs_legacy(processor, messages)


def _prepare_inputs_llava(processor, images: list[Image.Image], prompt: str):
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=images, return_tensors="pt", padding=True)


def _prepare_inputs_legacy(processor, messages: list[dict]):
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            'Эта версия transformers не приняла изображения напрямую. Установите qwen-vl-utils: pip install -e ".[vlm]"'
        ) from exc
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
