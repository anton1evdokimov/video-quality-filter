import numpy as np
import pytest

from video_quality_filter.alignment import InternVideoAligner, cosine_similarity
from video_quality_filter.config import VideoTextConfig


def test_cosine_stays_a_similarity_and_can_be_negative():
    left = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(left, np.array([0.0, 1.0], dtype=np.float32)) == pytest.approx(0.0)
    assert cosine_similarity(left, np.array([-1.0, 0.0], dtype=np.float32)) == pytest.approx(-1.0)
    assert cosine_similarity(left, np.array([2.0, 0.0], dtype=np.float32)) == pytest.approx(1.0)


def test_internvideo_backend_explains_missing_extra():
    with pytest.raises(RuntimeError, match="internvideo"):
        InternVideoAligner(VideoTextConfig(backend="internvideo2")).warmup()
