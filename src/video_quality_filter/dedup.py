"""Cluster video embeddings and mark near-duplicates. Raw files are not removed."""

from __future__ import annotations

from video_quality_filter.config import DedupConfig


def apply_dedup(records: list[dict], embeddings: list, config: DedupConfig) -> list[str]:
    """Write deduplication metadata. Returns the canonical video id for each row.

    The canonical id is the lexicographically smallest id in a near-duplicate
    component. A filtering policy can reject the other copies and still keep
    this one. Raw files are never deleted.
    """
    info = annotate_dedup([record["video_id"] for record in records], embeddings, config)
    canonical = info["canonical_id"]
    for index, record in enumerate(records):
        similarity = info["similarity"][index]
        record["deduplication"] = {
            "cluster_id": info["cluster_id"][index],
            "nearest_video_id": info["nearest_video_id"][index],
            "similarity": None if similarity is None else round(float(similarity), 4),
            "is_near_duplicate": bool(info["is_near_duplicate"][index]),
        }
    return canonical


def annotate_dedup(
    video_ids: list[str],
    embeddings: list,
    config: DedupConfig,
) -> dict[str, list]:
    """Return parallel lists: nearest id, similarity, cluster id, canonical id, flag."""
    count = len(video_ids)
    nearest_id: list[str | None] = [None] * count
    nearest_sim: list[float | None] = [None] * count
    for index, _video_id in enumerate(video_ids):
        best_sim = None
        best_other = None
        for other in range(count):
            if other == index:
                continue
            similarity = _similarity(embeddings[index], embeddings[other])
            if similarity is None:
                continue
            if best_sim is None or similarity > best_sim:
                best_sim = similarity
                best_other = other
        if best_other is not None and best_sim is not None:
            nearest_id[index] = video_ids[best_other]
            nearest_sim[index] = best_sim

    cluster_ids = _component_ids(video_ids, embeddings, config.cluster_similarity)
    canonical = _canonical_ids(video_ids, embeddings, config.near_duplicate_similarity)
    return {
        "nearest_video_id": nearest_id,
        "similarity": nearest_sim,
        "cluster_id": cluster_ids,
        "canonical_id": canonical,
        "is_near_duplicate": [
            nearest_sim[index] is not None and nearest_sim[index] >= config.near_duplicate_similarity
            for index in range(count)
        ],
    }


def _similarity(left, right) -> float | None:
    if left is None or right is None:
        return None
    import numpy as np

    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        return None
    return float(np.dot(a, b))


def _component_ids(video_ids: list[str], embeddings: list, threshold: float) -> list[int]:
    parent = list(range(len(video_ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(video_ids)):
        for right in range(left + 1, len(video_ids)):
            similarity = _similarity(embeddings[left], embeddings[right])
            if similarity is not None and similarity >= threshold:
                union(left, right)
    roots = [find(index) for index in range(len(video_ids))]
    order = sorted(set(roots), key=lambda root: min(index for index, item in enumerate(roots) if item == root))
    mapping = {root: cluster_id for cluster_id, root in enumerate(order)}
    return [mapping[root] for root in roots]


def _canonical_ids(video_ids: list[str], embeddings: list, threshold: float) -> list[str]:
    parent = list(range(len(video_ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(video_ids)):
        for right in range(left + 1, len(video_ids)):
            similarity = _similarity(embeddings[left], embeddings[right])
            if similarity is not None and similarity >= threshold:
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(len(video_ids)):
        groups.setdefault(find(index), []).append(index)
    canonical = [""] * len(video_ids)
    for indexes in groups.values():
        keeper = min(indexes, key=lambda index: video_ids[index])
        keeper_id = video_ids[keeper]
        for index in indexes:
            canonical[index] = keeper_id
    return canonical
