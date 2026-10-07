import numpy as np
import pytest

from video_quality_filter.alignment import (
    InternVideoAligner,
    cosine_similarity,
    make_flash_attn_optional,
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


def test_internvideo_backend_explains_missing_extra():
    with pytest.raises(RuntimeError, match="internvideo"):
        InternVideoAligner(VideoTextConfig(backend="internvideo2")).warmup()
