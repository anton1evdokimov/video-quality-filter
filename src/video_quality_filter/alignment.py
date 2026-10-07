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
_MODELING_UTILS_IMPORT = re.compile(
    r"from transformers\.modeling_utils import \(\s*PreTrainedModel\s*,\s*"
    r"apply_chunking_to_forward\s*,\s*"
    r"find_pruneable_heads_and_indices\s*,\s*"
    r"prune_linear_layer\s*\)",
    re.S,
)
_PREVIOUS_HELPER_IMPORT = re.compile(
    r"from transformers\.modeling_utils import PreTrainedModel\n"
    r"try:\n"
    r"    from transformers\.pytorch_utils import \(\n"
    r"        apply_chunking_to_forward,\n"
    r"        find_pruneable_heads_and_indices,\n"
    r"        prune_linear_layer,\n"
    r"    \)\n"
    r"except ImportError:\n"
    r"    from transformers\.modeling_utils import \(\n"
    r"        apply_chunking_to_forward,\n"
    r"        find_pruneable_heads_and_indices,\n"
    r"        prune_linear_layer,\n"
    r"    \)",
    re.M,
)
_LOCAL_BERT_HELPERS = '''from transformers.modeling_utils import PreTrainedModel

def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    if chunk_size <= 0:
        return forward_fn(*input_tensors)
    num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
    chunks = tuple(tensor.chunk(num_chunks, dim=chunk_dim) for tensor in input_tensors)
    outputs = tuple(forward_fn(*parts) for parts in zip(*chunks))
    return torch.cat(outputs, dim=chunk_dim)

def prune_linear_layer(layer, index, dim=0):
    index = index.to(layer.weight.device)
    weight = layer.weight.index_select(dim, index).detach().clone()
    bias = None
    if layer.bias is not None:
        bias = layer.bias.detach().clone() if dim == 1 else layer.bias[index].detach().clone()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad_(False)
    new_layer.weight.copy_(weight.contiguous())
    new_layer.weight.requires_grad_(True)
    if bias is not None:
        new_layer.bias.requires_grad_(False)
        new_layer.bias.copy_(bias.contiguous())
        new_layer.bias.requires_grad_(True)
    return new_layer

def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - set(already_pruned_heads)
    for head in heads:
        shifted = head - sum(1 for pruned in already_pruned_heads if pruned < head)
        mask[shifted] = 0
    keep = mask.view(-1).contiguous().eq(1)
    index = torch.arange(keep.numel())[keep].long()
    return heads, index
'''
_TOKEN_IMPORT = re.compile(
    r"from transformers\.tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace"
)
_TOKEN_HELPERS = '''import unicodedata
try:
    from transformers.tokenization_utils import PreTrainedTokenizer
except ImportError:
    from transformers.tokenization_python import PreTrainedTokenizer

def _is_whitespace(char):
    if char in (" ", chr(9), chr(10), chr(13)):
        return True
    return unicodedata.category(char) == "Zs"

def _is_control(char):
    if char in (chr(9), chr(10), chr(13)):
        return False
    return unicodedata.category(char).startswith("C")

def _is_punctuation(char):
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(char).startswith("P")
'''


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


def make_transformers_imports_compatible(path: Path) -> None:
    """Define BERT helpers locally. Current transformers split or removed the old imports."""
    text = path.read_text(encoding="utf-8")
    if "def find_pruneable_heads_and_indices(" not in text:
        updated, count = _PREVIOUS_HELPER_IMPORT.subn(_LOCAL_BERT_HELPERS, text, count=1)
        if count == 0:
            updated, count = _MODELING_UTILS_IMPORT.subn(_LOCAL_BERT_HELPERS, text, count=1)
        if count != 1:
            raise RuntimeError("Не удалось подготовить код InternVideo2: неожиданный импорт transformers")
        text = updated
    if "def _is_punctuation(" not in text:
        updated, count = _TOKEN_IMPORT.subn(_TOKEN_HELPERS, text, count=1)
        if count != 1:
            raise RuntimeError("Не удалось подготовить код InternVideo2: неожиданный импорт токенизатора")
        text = updated
    path.write_text(text, encoding="utf-8")


_INIT_WEIGHTS = "        self.init_weights()\n"
_INIT_WEIGHTS_WITH_TIED_KEYS = """        self.init_weights()
        if not hasattr(self, "all_tied_weights_keys"):
            tied = getattr(type(self), "_tied_weights_keys", None)
            self.all_tied_weights_keys = dict(tied) if isinstance(tied, dict) else {}
"""


_IGNORE_KEY_LISTS = (
    ('    _keys_to_ignore_on_load_missing = [r"position_ids"]\n', '    _keys_to_ignore_on_load_missing = {r"position_ids"}\n'),
    (
        '    _keys_to_ignore_on_load_unexpected = [r"pooler"]\n',
        '    _keys_to_ignore_on_load_unexpected = {r"pooler"}\n',
    ),
    (
        '    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]\n',
        '    _keys_to_ignore_on_load_missing = {r"position_ids", r"predictions.decoder.bias"}\n',
    ),
)


def make_ignore_keys_compatible(path: Path) -> None:
    """transformers 5 unions these patterns with a set. InternVideo2 stores them as lists."""
    text = path.read_text(encoding="utf-8")
    if '_keys_to_ignore_on_load_unexpected = {r"pooler"}' in text:
        return
    for old, new in _IGNORE_KEY_LISTS:
        if old not in text:
            raise RuntimeError("Не удалось подготовить код InternVideo2: нет списка ignore-ключей")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


_HEAD_MASK_CALL = "        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)\n"
_HEAD_MASK_CALL_FIXED = """        if head_mask is None or not hasattr(self, "get_head_mask"):
            head_mask = [None] * self.config.num_hidden_layers
        else:
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
"""


def make_head_mask_compatible(path: Path) -> None:
    """transformers 5 removed PreTrainedModel.get_head_mask. Inference always passes None."""
    text = path.read_text(encoding="utf-8")
    if 'not hasattr(self, "get_head_mask")' in text:
        return
    if text.count(_HEAD_MASK_CALL) != 1:
        raise RuntimeError("Не удалось подготовить код InternVideo2: неожиданный вызов get_head_mask")
    path.write_text(text.replace(_HEAD_MASK_CALL, _HEAD_MASK_CALL_FIXED, 1), encoding="utf-8")


def make_tied_weights_compatible(path: Path) -> None:
    """transformers 5 reads all_tied_weights_keys, which only post_init() creates."""
    text = path.read_text(encoding="utf-8")
    if "self.all_tied_weights_keys" in text:
        return
    if _INIT_WEIGHTS not in text:
        raise RuntimeError("Не удалось подготовить код InternVideo2: нет init_weights()")
    path.write_text(text.replace(_INIT_WEIGHTS, _INIT_WEIGHTS_WITH_TIED_KEYS), encoding="utf-8")


def _modeling_is_patched(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return (
        "flash_attn_varlen_qkvpacked_func = None" in text
        and "def find_pruneable_heads_and_indices(" in text
        and "def _is_punctuation(" in text
        and "self.all_tied_weights_keys" in text
        and '_keys_to_ignore_on_load_unexpected = {r"pooler"}' in text
        and 'not hasattr(self, "get_head_mask")' in text
    )


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
    if _modeling_is_patched(modeling) and (dest / "config.json").is_file():
        return dest
    if modeling.is_file() and (dest / "config.json").is_file():
        make_flash_attn_optional(modeling)
        make_transformers_imports_compatible(modeling)
        make_tied_weights_compatible(modeling)
        make_ignore_keys_compatible(modeling)
        make_head_mask_compatible(modeling)
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
    make_transformers_imports_compatible(modeling)
    make_tied_weights_compatible(modeling)
    make_ignore_keys_compatible(modeling)
    make_head_mask_compatible(modeling)
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
