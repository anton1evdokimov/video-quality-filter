"""InternVideo2 video/text embeddings and their cosine similarity.

The stored value is cosine similarity in [-1, 1]. Retrieval demos often compute
softmax(100 * cosine); that probability is intentionally not used here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from video_quality_filter.config import VideoTextConfig

logger = logging.getLogger(__name__)

MODEL_NAME = "InternVideo2"


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("эмбеддинги видео и текста разной размерности")
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        raise ValueError("нулевой эмбеддинг")
    return float(np.dot(a / a_norm, b / b_norm))


class InternVideoAligner:
    def __init__(self, config: VideoTextConfig) -> None:
        self.config = config
        self._model = None

    def warmup(self) -> None:
        self._ensure()

    def video_embedding(self, video_path: Path) -> np.ndarray:
        model = self._ensure()
        tensor = self._video_tensor(video_path)
        feature = model.get_vid_feat(tensor)
        return _as_unit_vector(feature)

    def text_embedding(self, text: str) -> np.ndarray:
        model = self._ensure()
        if hasattr(model, "get_txt_feat"):
            feature = model.get_txt_feat(text)
        elif hasattr(model, "get_text_feat"):
            feature = model.get_text_feat(text)
        else:
            raise RuntimeError("У модели InternVideo2 нет get_txt_feat / get_text_feat")
        return _as_unit_vector(feature)

    def _ensure(self):
        if self._model is not None:
            return self._model
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                'InternVideo2 требует дополнительные пакеты: pip install -e ".[internvideo]"'
            ) from exc
        logger.info("Загружаю %s %s", MODEL_NAME, self.config.model_id)
        model = AutoModel.from_pretrained(self.config.model_id, trust_remote_code=True)
        model.eval()
        device = _torch_device(torch, self.config.device)
        self._model = model.to(device)
        return self._model

    def _video_tensor(self, video_path: Path):
        module = __import__(self._model.__class__.__module__, fromlist=["vid2tensor"])
        vid2tensor = getattr(module, "vid2tensor", None)
        if vid2tensor is None:
            raise RuntimeError("В коде InternVideo2 нет vid2tensor; нужен trust_remote_code модели")
        return vid2tensor(str(video_path), fnum=self.config.num_frames)


def _torch_device(torch, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    if requested == "mps":
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _as_unit_vector(feature) -> np.ndarray:
    if hasattr(feature, "detach"):
        feature = feature.detach().float().cpu().numpy()
    array = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise RuntimeError("InternVideo2 вернул нулевой эмбеддинг")
    return array / norm
