"""Accept or reject a record from configurable thresholds.

This stage does not blend metrics into one score. A missing measurement is
ignored unless the matching require_* flag is set. video_text cosine is
compared as a similarity, not as a probability.
"""

from __future__ import annotations

from video_quality_filter.config import FilteringConfig


def apply_policy(
    record: dict,
    policy: FilteringConfig,
    *,
    canonical_video_id: str,
    extract_errors: list[str],
) -> None:
    reasons = list(record.get("_technical_reasons", []))
    reasons.extend(extract_errors)
    visual = record["visual"]
    vlm = record["vlm"]
    video_text = record["video_text"]
    audio = record["audio"]
    dedup = record["deduplication"]
    content = record.get("content") or {}
    technical = record["technical"]

    _below(reasons, "aesthetic_score_below_threshold", visual.get("aesthetic_score"), policy.min_aesthetic_score)
    _above(
        reasons,
        "watermark_probability_above_threshold",
        visual.get("watermark_probability"),
        policy.max_watermark_probability,
    )
    _above(reasons, "text_area_ratio_above_threshold", visual.get("text_area_ratio"), policy.max_text_area_ratio)
    if policy.require_visual and _visual_missing(visual):
        reasons.append("visual_required_but_missing")

    _below(
        reasons,
        "semantic_consistency_below_threshold",
        vlm.get("semantic_consistency"),
        policy.min_semantic_consistency,
    )
    _below(reasons, "temporal_coverage_below_threshold", vlm.get("temporal_coverage"), policy.min_temporal_coverage)
    _below(reasons, "completeness_below_threshold", vlm.get("completeness"), policy.min_completeness)
    _above(reasons, "hallucination_above_threshold", vlm.get("hallucination"), policy.max_hallucination)
    if policy.require_vlm and vlm.get("caption") is None:
        reasons.append("vlm_required_but_missing")

    _below(
        reasons,
        "video_text_cosine_below_threshold",
        video_text.get("cosine_similarity"),
        policy.min_video_text_cosine,
    )
    if policy.require_video_text and video_text.get("cosine_similarity") is None:
        reasons.append("video_text_required_but_missing")

    if technical.get("audio_present"):
        _above(reasons, "silence_ratio_above_threshold", audio.get("silence_ratio"), policy.max_silence_ratio)
        _below_raw(reasons, "rms_below_threshold", audio.get("rms"), policy.min_rms)
        _above(reasons, "clipping_ratio_above_threshold", audio.get("clipping_ratio"), policy.max_clipping_ratio)

    if (
        policy.reject_near_duplicates
        and dedup.get("is_near_duplicate")
        and record["video_id"] != canonical_video_id
    ):
        reasons.append("near_duplicate")

    label = content.get("label")
    if label is not None and label in policy.reject_labels:
        reasons.append("content_label_rejected")
    if policy.require_content and label is None:
        reasons.append("content_required_but_missing")

    unique: list[str] = []
    for reason in reasons:
        if reason not in unique:
            unique.append(reason)
    record["filtering"] = {
        "status": "rejected" if unique else "accepted",
        "reasons": unique,
    }
    record.pop("_technical_reasons", None)


def _visual_missing(visual: dict) -> bool:
    return any(visual.get(key) is None for key in ("aesthetic_score", "watermark_probability", "text_area_ratio"))


def _below(reasons: list[str], code: str, value, threshold: float | None) -> None:
    if threshold is None or value is None:
        return
    if float(value) < float(threshold):
        reasons.append(code)


def _below_raw(reasons: list[str], code: str, value, threshold: float | None) -> None:
    _below(reasons, code, value, threshold)


def _above(reasons: list[str], code: str, value, threshold: float | None) -> None:
    if threshold is None or value is None:
        return
    if float(value) > float(threshold):
        reasons.append(code)
