import numpy as np
import pytest

from video_quality_filter.alignment import (
    InternVideoAligner,
    cosine_similarity,
    make_flash_attn_optional,
    make_tied_weights_compatible,
    make_transformers_imports_compatible,
    raw_checkpoint_filename,
)
from video_quality_filter.config import VideoTextConfig


def test_cosine_stays_a_similarity_and_can_be_negative():
    left = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(left, np.array([0.0, 1.0], dtype=np.float32)) == pytest.approx(0.0)
    assert cosine_similarity(left, np.array([-1.0, 0.0], dtype=np.float32)) == pytest.approx(-1.0)
    assert cosine_similarity(left, np.array([2.0, 0.0], dtype=np.float32)) == pytest.approx(1.0)


def test_1b_repo_is_a_raw_checkpoint_and_6b_is_not():
    assert raw_checkpoint_filename("OpenGVLab/InternVideo2-Stage2_1B-224p-f4") == (
        "InternVideo2-stage2_1b-224p-f4.pt"
    )
    assert raw_checkpoint_filename("OpenGVLab/InternVideo2-Stage2_6B") is None


def test_flash_attn_import_becomes_optional(tmp_path):
    source = tmp_path / "modeling_internvideo2.py"
    source.write_text(
        "import torch\n"
        "from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func\n"
        "from flash_attn.bert_padding import unpad_input, pad_input\n"
        "x = 1\n",
        encoding="utf-8",
    )
    make_flash_attn_optional(source)
    text = source.read_text(encoding="utf-8")
    assert "flash_attn_varlen_qkvpacked_func = None" in text
    assert "x = 1" in text


def test_bert_helpers_are_local_instead_of_transformers_imports(tmp_path):
    source = tmp_path / "modeling_internvideo2.py"
    source.write_text(
        "from transformers.modeling_utils import (PreTrainedModel,\n"
        "                                         apply_chunking_to_forward,\n"
        "                                         find_pruneable_heads_and_indices,\n"
        "                                         prune_linear_layer)\n"
        "from transformers.tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace\n"
        "keep = 1\n",
        encoding="utf-8",
    )
    make_transformers_imports_compatible(source)
    text = source.read_text(encoding="utf-8")
    assert "def find_pruneable_heads_and_indices(" in text
    assert "def _is_punctuation(" in text
    assert "from transformers.pytorch_utils import (" not in text
    assert "keep = 1" in text


def test_previous_transformers_import_patch_is_replaced(tmp_path):
    source = tmp_path / "modeling_internvideo2.py"
    source.write_text(
        "from transformers.modeling_utils import PreTrainedModel\n"
        "try:\n"
        "    from transformers.pytorch_utils import (\n"
        "        apply_chunking_to_forward,\n"
        "        find_pruneable_heads_and_indices,\n"
        "        prune_linear_layer,\n"
        "    )\n"
        "except ImportError:\n"
        "    from transformers.modeling_utils import (\n"
        "        apply_chunking_to_forward,\n"
        "        find_pruneable_heads_and_indices,\n"
        "        prune_linear_layer,\n"
        "    )\n"
        "from transformers.tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace\n",
        encoding="utf-8",
    )
    make_transformers_imports_compatible(source)
    text = source.read_text(encoding="utf-8")
    assert "def find_pruneable_heads_and_indices(" in text
    assert "from transformers.pytorch_utils import (" not in text


def test_init_weights_also_sets_all_tied_weights_keys(tmp_path):
    source = tmp_path / "modeling_internvideo2.py"
    source.write_text(
        "class BertModel:\n"
        "    def __init__(self):\n"
        "        self.init_weights()\n"
        "\n"
        "class BertForMaskedLM:\n"
        "    def __init__(self):\n"
        "        self.init_weights()\n",
        encoding="utf-8",
    )
    make_tied_weights_compatible(source)
    text = source.read_text(encoding="utf-8")
    assert text.count("self.all_tied_weights_keys") == 2
    assert text.count("self.init_weights()") == 2


def test_internvideo_backend_explains_missing_extra():
    with pytest.raises(RuntimeError, match="internvideo"):
        InternVideoAligner(VideoTextConfig(backend="internvideo2")).warmup()
