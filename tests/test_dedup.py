import numpy as np

from video_quality_filter.config import DedupConfig
from video_quality_filter.dedup import apply_dedup


def _record(video_id: str) -> dict:
    return {"video_id": video_id, "deduplication": {}}


def test_near_duplicates_share_a_cluster_and_keep_a_canonical_id():
    records = [_record("b"), _record("a"), _record("c")]
    same = np.array([1.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0], dtype=np.float32)
    canonical = apply_dedup(records, [same, same.copy(), other], DedupConfig())
    by_id = {record["video_id"]: record["deduplication"] for record in records}
    assert by_id["a"]["cluster_id"] == by_id["b"]["cluster_id"]
    assert by_id["a"]["cluster_id"] != by_id["c"]["cluster_id"]
    assert by_id["a"]["is_near_duplicate"] is True
    assert by_id["b"]["is_near_duplicate"] is True
    assert by_id["c"]["is_near_duplicate"] is False
    assert by_id["a"]["nearest_video_id"] == "b"
    assert by_id["a"]["similarity"] == 1
    assert canonical[records.index(next(record for record in records if record["video_id"] == "b"))] == "a"
    assert "a" in canonical
    assert canonical[0] == "a" or canonical[1] == "a"


def test_single_video_has_no_neighbor():
    records = [_record("only")]
    canonical = apply_dedup(records, [np.array([1.0, 0.0], dtype=np.float32)], DedupConfig())
    dedup = records[0]["deduplication"]
    assert dedup["nearest_video_id"] is None
    assert dedup["similarity"] is None
    assert dedup["is_near_duplicate"] is False
    assert dedup["cluster_id"] == 0
    assert canonical == ["only"]
