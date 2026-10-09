from video_quality_filter.config import FilteringConfig
from video_quality_filter.filtering import apply_policy
from video_quality_filter.models import empty_audio, empty_video_text, empty_vlm


def _record(**updates) -> dict:
    record = {
        "video_id": "000123",
        "source": "raw/videos/000123.mp4",
        "technical": {
            "duration": 12.4,
            "fps_original": 60.0,
            "fps_processed": 24.0,
            "width": 1920,
            "height": 1080,
            "codec": "h264",
            "audio_present": True,
            "audio_duration": 12.2,
            "av_duration_diff": 0.2,
            "technical_ok": True,
        },
        "visual": {"aesthetic_score": 0.82, "watermark_probability": 0.03, "text_area_ratio": 0.04},
        "vlm": {
            "caption": "A man opens a car door and gets into the vehicle.",
            "semantic_consistency": 0.93,
            "temporal_coverage": 0.84,
            "completeness": 0.81,
            "hallucination": 0.05,
        },
        "video_text": {"model": "InternVideo2", "cosine_similarity": 0.78},
        "audio": {"silence_ratio": 0.04, "rms": 0.18, "clipping_ratio": 0.002, "audio_quality": 0.9},
        "deduplication": {
            "cluster_id": 1,
            "nearest_video_id": "000981",
            "similarity": 0.5,
            "is_near_duplicate": False,
        },
        "_technical_reasons": [],
    }
    record.update(updates)
    return record


def test_policy_accepts_without_collapsing_metrics_into_one_score():
    record = _record()
    apply_policy(record, FilteringConfig(), canonical_video_id="000123", extract_errors=[])
    assert record["filtering"] == {"status": "accepted", "reasons": []}
    assert "quality_score" not in record


def test_cosine_threshold_is_a_similarity_gate():
    record = _record()
    record["video_text"] = {"model": "InternVideo2", "cosine_similarity": -0.2}
    policy = FilteringConfig(min_video_text_cosine=0.3)
    apply_policy(record, policy, canonical_video_id="000123", extract_errors=[])
    assert record["filtering"]["status"] == "rejected"
    assert record["filtering"]["reasons"] == ["video_text_cosine_below_threshold"]
    assert record["video_text"]["cosine_similarity"] == -0.2


def test_missing_vlm_scores_are_not_rejected_until_required():
    record = _record(vlm=empty_vlm(), video_text=empty_video_text())
    apply_policy(
        record,
        FilteringConfig(min_semantic_consistency=0.8, min_video_text_cosine=0.5),
        canonical_video_id="000123",
        extract_errors=[],
    )
    assert record["filtering"]["status"] == "accepted"
    apply_policy(
        record,
        FilteringConfig(require_vlm=True),
        canonical_video_id="000123",
        extract_errors=[],
    )
    assert "vlm_required_but_missing" in record["filtering"]["reasons"]


def test_near_duplicate_keeps_the_canonical_video():
    record = _record(
        deduplication={
            "cluster_id": 4,
            "nearest_video_id": "000100",
            "similarity": 0.99,
            "is_near_duplicate": True,
        }
    )
    policy = FilteringConfig(reject_near_duplicates=True)
    apply_policy(record, policy, canonical_video_id="000100", extract_errors=[])
    assert record["filtering"]["reasons"] == ["near_duplicate"]
    apply_policy(record, policy, canonical_video_id="000123", extract_errors=[])
    assert record["filtering"]["status"] == "accepted"


def test_rejected_content_label_is_a_reason():
    record = _record(content={"label": "screen", "cluster_id": 1, "cluster_size": 2})
    apply_policy(
        record,
        FilteringConfig(reject_labels=["screen"]),
        canonical_video_id="000123",
        extract_errors=[],
    )
    assert record["filtering"]["reasons"] == ["content_label_rejected"]


def test_audio_thresholds_use_raw_features():
    record = _record(audio={**empty_audio(), "silence_ratio": 0.99, "rms": 0.0, "clipping_ratio": 0.2})
    apply_policy(record, FilteringConfig(), canonical_video_id="000123", extract_errors=[])
    assert "silence_ratio_above_threshold" in record["filtering"]["reasons"]
    assert "clipping_ratio_above_threshold" in record["filtering"]["reasons"]
    assert "audio_quality" not in record["filtering"]["reasons"]
