"""Frame-level visual metrics aggregated to one video.

aesthetic_score is a lightweight proxy (color, contrast, sharpness, exposure).
watermark_probability rises when a corner stays stable while the center changes.
text_area_ratio is the share of the frame covered by text-like horizontal strokes.
"""

from __future__ import annotations

import math

import numpy as np

from video_quality_filter.extract import ExtractedFrame
from video_quality_filter.util import clamp


def estimate_visual(frames: list[ExtractedFrame]) -> dict[str, float]:
    images = [np.asarray(frame.image) for frame in frames]
    aesthetic = float(np.mean([aesthetic_score(image) for image in images]))
    text_ratio = float(np.mean([text_area_ratio(image) for image in images]))
    return {
        "aesthetic_score": round(aesthetic, 4),
        "watermark_probability": round(watermark_probability(images), 4),
        "text_area_ratio": round(text_ratio, 4),
    }


def aesthetic_score(image: np.ndarray) -> float:
    rgb = _rgb(image)
    gray = _luminance(rgb)
    color = _colorfulness(rgb)
    contrast = clamp((float(gray.std()) - 8.0) / 40.0, 0.0, 1.0)
    sharpness = 1.0 - math.exp(-_laplacian_variance(gray) / 400.0)
    exposure = _band_score(float(gray.mean()) / 255.0, 0.18, 0.82)
    score = 0.35 * color + 0.25 * contrast + 0.20 * sharpness + 0.20 * exposure
    return float(clamp(score, 0.0, 1.0))


def text_area_ratio(image: np.ndarray) -> float:
    gray = _luminance(_rgb(image))
    if gray.shape[1] < 3 or gray.shape[0] < 3:
        return 0.0
    gradient = np.abs(np.diff(gray, axis=1))
    strong = gradient >= 25.0
    row_density = strong.mean(axis=1)
    text_rows = (row_density > 0.05) & (row_density < 0.35)
    if not np.any(text_rows):
        return 0.0
    kernel = np.ones(21, dtype=np.float32)
    covered = 0
    width = strong.shape[1]
    for row in np.flatnonzero(text_rows):
        signal = strong[row].astype(np.float32)
        if float(np.convolve(signal, kernel, mode="same").max()) <= 0:
            continue
        covered += int(np.count_nonzero(np.convolve(signal, kernel, mode="same") > 0))
    return float(clamp(covered / float(gray.shape[0] * width), 0.0, 1.0))


def watermark_probability(images: list[np.ndarray]) -> float:
    grays = [_luminance(_rgb(image)) for image in images]
    energies = [_corner_structure(gray) for gray in grays]
    energy = float(np.mean(energies)) if energies else 0.0
    if len(grays) < 2:
        return float(clamp(energy, 0.0, 1.0))
    persistence = _corner_persistence(grays)
    score = energy * (0.35 + 0.65 * persistence)
    return float(clamp(score, 0.0, 1.0))


def _corner_structure(gray: np.ndarray) -> float:
    height, width = gray.shape
    corner_h = max(1, height // 5)
    corner_w = max(1, width // 5)
    corners = (
        gray[:corner_h, :corner_w],
        gray[:corner_h, -corner_w:],
        gray[-corner_h:, :corner_w],
        gray[-corner_h:, -corner_w:],
    )
    y0, y1 = height // 3, max(height // 3 + 1, (2 * height) // 3)
    x0, x1 = width // 3, max(width // 3 + 1, (2 * width) // 3)
    center_density = _edge_density(gray[y0:y1, x0:x1])
    corner_density = max(_edge_density(corner) for corner in corners)
    if corner_density < 0.02:
        return 0.0
    return float(clamp((corner_density - center_density) / corner_density, 0.0, 1.0))


def _corner_persistence(grays: list[np.ndarray]) -> float:
    deltas: list[float] = []
    for left, right in zip(grays, grays[1:]):
        for crop_a, crop_b in zip(_corners(left), _corners(right)):
            height = min(crop_a.shape[0], crop_b.shape[0])
            width = min(crop_a.shape[1], crop_b.shape[1])
            if height == 0 or width == 0:
                continue
            delta = np.abs(crop_a[:height, :width] - crop_b[:height, :width])
            deltas.append(float(np.mean(delta) / 255.0))
    if not deltas:
        return 0.0
    return float(clamp(1.0 - (float(np.mean(deltas)) / 0.12), 0.0, 1.0))


def _corners(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = gray.shape
    corner_h = max(1, height // 5)
    corner_w = max(1, width // 5)
    return (
        gray[:corner_h, :corner_w],
        gray[:corner_h, -corner_w:],
        gray[-corner_h:, :corner_w],
        gray[-corner_h:, -corner_w:],
    )


def _edge_density(region: np.ndarray) -> float:
    if region.size == 0 or region.shape[1] < 2:
        return 0.0
    gradient = np.abs(np.diff(region.astype(np.float32), axis=1))
    return float(np.mean(gradient >= 20.0))


def _colorfulness(rgb: np.ndarray) -> float:
    red = rgb[:, :, 0].astype(np.float32)
    green = rgb[:, :, 1].astype(np.float32)
    blue = rgb[:, :, 2].astype(np.float32)
    red_green = red - green
    yellow_blue = 0.5 * (red + green) - blue
    raw = math.sqrt(float(red_green.var()) + float(yellow_blue.var()))
    raw += 0.3 * math.sqrt(float(red_green.mean()) ** 2 + float(yellow_blue.mean()) ** 2)
    return float(clamp(1.0 - math.exp(-raw / 40.0), 0.0, 1.0))


def _rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        scaled = array.astype(np.float32)
        if scaled.size and float(np.nanmax(scaled)) <= 1.0:
            scaled *= 255.0
        array = np.clip(scaled, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array[:, :, :3])


def _luminance(rgb: np.ndarray) -> np.ndarray:
    channels = rgb.astype(np.float32)
    return 0.299 * channels[:, :, 0] + 0.587 * channels[:, :, 1] + 0.114 * channels[:, :, 2]


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * center
    )
    return float(lap.var())


def _band_score(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        if low <= 0:
            return 0.0
        return float(clamp(value / low, 0.0, 1.0))
    span = 1.0 - high
    if span <= 0:
        return 0.0
    return float(clamp((1.0 - value) / span, 0.0, 1.0))
