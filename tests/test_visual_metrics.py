import numpy as np

from video_quality_filter.extract import ExtractedFrame
from video_quality_filter.visual_metrics import aesthetic_score, estimate_visual, text_area_ratio, watermark_probability


def _frame(image: np.ndarray, timestamp: float = 0.5) -> ExtractedFrame:
    from pathlib import Path

    return ExtractedFrame(timestamp_sec=timestamp, path=Path("frame.jpg"), image=image)


def test_black_frame_is_less_aesthetic_than_color_noise():
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    assert aesthetic_score(black) < 0.15
    assert aesthetic_score(noise) > aesthetic_score(black)


def test_text_like_strokes_cover_more_area_than_a_blank_frame():
    blank = np.full((80, 160, 3), 20, dtype=np.uint8)
    text = blank.copy()
    for row in (30, 32, 34):
        text[row, 20:140:6] = 240
        text[row + 1, 20:140:6] = 240
    assert text_area_ratio(text) > text_area_ratio(blank)


def test_stable_corner_logo_raises_watermark_probability():
    rng = np.random.default_rng(1)
    frames = []
    for timestamp in (0.2, 0.8):
        image = rng.integers(0, 40, size=(80, 80, 3), dtype=np.uint8)
        image[:16, :16] = 255
        image[2:14, 2:14] = 0
        frames.append(image)
    blank = [np.zeros((80, 80, 3), dtype=np.uint8), np.zeros((80, 80, 3), dtype=np.uint8)]
    assert watermark_probability(frames) > watermark_probability(blank)
    aggregated = estimate_visual([_frame(frames[0], 0.2), _frame(frames[1], 0.8)])
    assert set(aggregated) == {"aesthetic_score", "watermark_probability", "text_area_ratio"}
