import numpy as np

from video_quality_filter.config import ContentConfig
from video_quality_filter.content import apply_content_clusters, normalize_content_label
from video_quality_filter.models import empty_content


def _record(video_id: str) -> dict:
    return {"video_id": video_id, "content": empty_content()}


def test_unknown_label_falls_back_to_other():
    labels = ["person", "hands", "other"]
    assert normalize_content_label(" Hands ", labels) == "hands"
    assert normalize_content_label("landscape", labels) == "other"
    assert normalize_content_label("landscape", ["person", "hands"]) is None


def test_content_clusters_are_coarser_than_copies():
    records = [_record("a"), _record("b"), _record("c")]
    left = np.array([1.0, 0.0], dtype=np.float32)
    near = np.array([0.95, 0.05], dtype=np.float32)
    near = near / np.linalg.norm(near)
    far = np.array([0.0, 1.0], dtype=np.float32)
    apply_content_clusters(records, [left, near, far], ContentConfig(cluster_similarity=0.45))
    by_id = {record["video_id"]: record["content"] for record in records}
    assert by_id["a"]["cluster_id"] == by_id["b"]["cluster_id"]
    assert by_id["a"]["cluster_id"] != by_id["c"]["cluster_id"]
    assert by_id["a"]["cluster_size"] == 2
    assert by_id["c"]["cluster_size"] == 1
    assert float(np.dot(left, near)) >= 0.45
