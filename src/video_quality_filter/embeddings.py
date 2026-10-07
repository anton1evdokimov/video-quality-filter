"""Compact video embedding used for dedup when InternVideo2 is not enabled."""

from __future__ import annotations

import numpy as np

from video_quality_filter.extract import ExtractedFrame


def frame_histogram(frames: list[ExtractedFrame], bins: int = 16) -> np.ndarray:
    parts: list[np.ndarray] = []
    for frame in frames:
        image = np.asarray(frame.image)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        small = image[::8, ::8, :3]
        channel_hists = []
        for channel in range(3):
            hist, _edges = np.histogram(small[:, :, channel], bins=bins, range=(0, 255), density=True)
            channel_hists.append(hist.astype(np.float32))
        parts.append(np.concatenate(channel_hists))
    if not parts:
        return np.zeros(bins * 3, dtype=np.float32)
    vector = np.mean(np.stack(parts), axis=0)
    return _normalize(vector)


def _normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return array / norm
