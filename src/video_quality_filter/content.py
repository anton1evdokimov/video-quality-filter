"""Content label and a coarser cluster than near-duplicate dedup.

Clustering uses the same video embedding as deduplication. The threshold is
lower, so a cluster is similar content, not a copy. Raw files are not removed.
"""

from __future__ import annotations

from video_quality_filter.config import ContentConfig
from video_quality_filter.dedup import component_ids


def normalize_content_label(raw: str, labels: list[str]) -> str | None:
    label = raw.strip().lower()
    if label in labels:
        return label
    if "other" in labels:
        return "other"
    return None


def apply_content_clusters(records: list[dict], embeddings: list, config: ContentConfig) -> None:
    cluster_ids = component_ids(embeddings, config.cluster_similarity)
    sizes: dict[int, int] = {}
    for cluster_id in cluster_ids:
        sizes[cluster_id] = sizes.get(cluster_id, 0) + 1
    for record, cluster_id in zip(records, cluster_ids):
        content = record["content"]
        content["cluster_id"] = cluster_id
        content["cluster_size"] = sizes[cluster_id]
