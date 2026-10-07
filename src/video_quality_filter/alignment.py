"""InternVideo2 video/text embeddings and their cosine similarity.

The stored value is cosine similarity in [-1, 1]. Retrieval demos often compute
softmax(100 * cosine); that probability is intentionally not used here.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import numpy as np

from video_quality_filter.config import VideoTextConfig

logger = logging.getLogger(__name__)

MODEL_NAME = "InternVideo2"
# The 1B repo is a gated .pt with no config.json. Stage2 code lives in the 6B repo.
_STAGE2_CODE_REPO = "OpenGVLab/InternVideo2-Stage2_6B"
_RAW_CHECKPOINTS = {
    "OpenGVLab/InternVideo2-Stage2_1B-224p-f4": "InternVideo2-stage2_1b-224p-f4.pt",
}
_FLASH_IMPORT = """try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
except ImportError:
    flash_attn_varlen_qkvpacked_func = None
    unpad_input = None
    pad_input = None
"""
_FLASH_IMPORT_PATTERN = re.compile(
    r"^from flash_attn\.flash_attn_interface import flash_attn_varlen_qkvpacked_func\n"
    r"^from flash_attn\.bert_padding import unpad_input, pad_input\n",
    re.M,
)


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
        device = _torch_device(torch, self.config.device)
        filename = raw_checkpoint_filename(self.config.model_id)
        logger.info("Загружаю %s %s", MODEL_NAME, self.config.model_id)
        if filename is None:
            model = AutoModel.from_pretrained(self.config.model_id, trust_remote_code=True)
        else:
            logger.info(
                "У %s нет config.json, читаю чекпоинт %s",
                self.config.model_id,
                filename,
            )
            model = _load_raw_stage2(self.config, filename, device)
        model.eval()
        self._model = model.to(device)
        return self._model

    def _video_tensor(self, video_path: Path):
        module = __import__(self._model.__class__.__module__, fromlist=["vid2tensor"])
        vid2tensor = getattr(module, "vid2tensor", None)
        if vid2tensor is None:
            raise RuntimeError("В коде InternVideo2 нет vid2tensor; нужен trust_remote_code модели")
        device = next(self._model.parameters()).device
        return vid2tensor(str(video_path), fnum=self.config.num_frames, device=device)


def raw_checkpoint_filename(model_id: str) -> str | None:
    """Return the .pt name when the Hub repo has weights but no transformers config."""
    return _RAW_CHECKPOINTS.get(model_id.strip().strip("/"))


def make_flash_attn_optional(path: Path) -> None:
    """The Stage2 remote module imports flash_attn even when fused kernels are off."""
    text = path.read_text(encoding="utf-8")
    if "flash_attn_varlen_qkvpacked_func = None" in text:
        return
    updated, count = _FLASH_IMPORT_PATTERN.subn(_FLASH_IMPORT, text, count=1)
    if count != 1:
        raise RuntimeError("Не удалось подготовить код InternVideo2: неожиданный импорт flash_attn")
    path.write_text(updated, encoding="utf-8")


def _load_raw_stage2(config: VideoTextConfig, filename: str, device: str):
    from transformers import AutoConfig, AutoModel, BertTokenizer

    code_dir = _stage2_code_dir()
    BertTokenizer.from_pretrained("bert-large-uncased")
    cfg = AutoConfig.from_pretrained(code_dir, trust_remote_code=True)
    vision = cfg.model["vision_encoder"]
    vision["name"] = "pretrain_internvideo2_1b_patch14_224"
    vision["use_flash_attn"] = False
    vision["use_fused_mlp"] = False
    vision["use_fused_rmsnorm"] = False
    vision["num_frames"] = config.num_frames
    text_config = cfg.model["text_encoder"]["config"]
    if not Path(text_config).is_absolute():
        cfg.model["text_encoder"]["config"] = str(code_dir / text_config)
    cfg.num_frames = config.num_frames
    cfg.device = device
    model = AutoModel.from_config(cfg, trust_remote_code=True)
    checkpoint = _download_checkpoint(config.model_id, filename)
    _apply_checkpoint(model, checkpoint)
    return model


def _stage2_code_dir() -> Path:
    dest = Path.home() / ".cache" / "video-quality-filter" / "internvideo2-stage2-code"
    modeling = dest / "modeling_internvideo2.py"
    if modeling.is_file() and "flash_attn_varlen_qkvpacked_func = None" in modeling.read_text(encoding="utf-8"):
        return dest
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            'InternVideo2 требует дополнительные пакеты: pip install -e ".[internvideo]"'
        ) from exc
    source = Path(
        snapshot_download(
            _STAGE2_CODE_REPO,
            allow_patterns=["modeling_internvideo2.py", "config.json", "configs/*.json"],
        )
    )
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copy2(source / "modeling_internvideo2.py", modeling)
    shutil.copy2(source / "config.json", dest / "config.json")
    shutil.copytree(source / "configs", dest / "configs")
    make_flash_attn_optional(modeling)
    return dest


def _download_checkpoint(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            'InternVideo2 требует дополнительные пакеты: pip install -e ".[internvideo]"'
        ) from exc
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except Exception as exc:
        text = str(exc).lower()
        if type(exc).__name__ in {"GatedRepoError", "RepositoryNotFoundError"} or "gated" in text or "403" in text:
            raise RuntimeError(
                f"{repo_id} отдаёт только {filename} и закрыт условиями Hugging Face. "
                "Откройте страницу модели, примите условия и выполните huggingface-cli login."
            ) from exc
        raise


def _apply_checkpoint(model, path: Path) -> None:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = _checkpoint_state(checkpoint)
    loaded = _load_matching(model, state)
    if loaded == 0 and hasattr(model, "vision_encoder"):
        loaded = _load_matching(model.vision_encoder, state)
    if loaded == 0:
        raise RuntimeError(f"Чекпоинт {path.name} не совпал с архитектурой InternVideo2-1B")
    logger.info("Загружено весов InternVideo2: %d", loaded)


def _checkpoint_state(checkpoint) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("model", "module", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise RuntimeError("Чекпоинт InternVideo2 не похож на state dict")


def _load_matching(module, state: dict) -> int:
    own = module.state_dict()
    filtered = {
        key: value
        for key, value in state.items()
        if key in own and hasattr(value, "shape") and tuple(own[key].shape) == tuple(value.shape)
    }
    if not filtered:
        return 0
    module.load_state_dict(filtered, strict=False)
    return len(filtered)


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
