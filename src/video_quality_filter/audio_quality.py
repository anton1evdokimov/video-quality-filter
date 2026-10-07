"""Audio features from librosa. Raw measurements are always kept.

audio_quality is an optional aggregate of those measurements. Filtering uses
the raw features and its own thresholds, not this aggregate.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from video_quality_filter.config import AudioConfig
from video_quality_filter.util import clamp, weighted_mean


def analyze_audio_file(path: Path, config: AudioConfig) -> dict[str, float | None]:
    samples, sample_rate = librosa.load(path, sr=config.sample_rate, mono=True)
    return analyze_waveform(np.asarray(samples, dtype=np.float32), int(sample_rate), config)


def analyze_waveform(samples: np.ndarray, sample_rate: int, config: AudioConfig) -> dict[str, float | None]:
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    if sample_rate <= 0 or waveform.size == 0:
        raise ValueError("аудиодорожка пустая или частота дискретизации некорректна")
    rms_frames = np.asarray(librosa.feature.rms(y=waveform))
    if rms_frames.ndim == 2:
        rms_frames = rms_frames[0]
    if rms_frames.size == 0:
        raise ValueError("не удалось посчитать RMS")
    threshold = float(librosa.db_to_amplitude(config.silence_threshold_db))
    silence_ratio = float(np.mean(rms_frames < threshold))
    rms = float(np.mean(rms_frames))
    clipping_ratio = float(np.mean(np.abs(waveform) >= 0.999))
    quality = weighted_mean(
        [
            (_silence_score(silence_ratio), config.weight_silence),
            (_energy_score(rms), config.weight_energy),
            (1.0 - clamp(clipping_ratio / 0.01, 0.0, 1.0), config.weight_clipping),
        ]
    )
    return {
        "silence_ratio": round(silence_ratio, 4),
        "rms": round(rms, 6),
        "clipping_ratio": round(clipping_ratio, 6),
        "audio_quality": round(float(quality), 4),
    }


def _silence_score(ratio: float) -> float:
    if ratio <= 0.25:
        return 1.0
    return float(clamp((0.95 - ratio) / 0.70, 0.0, 1.0))


def _energy_score(rms: float) -> float:
    if rms <= 0:
        return 0.0
    if rms < 0.005:
        return float(clamp(rms / 0.005, 0.0, 1.0) * 0.5)
    if rms < 0.02:
        return float(0.5 + 0.5 * (rms - 0.005) / 0.015)
    if rms <= 0.30:
        return 1.0
    return float(clamp(1.0 - (rms - 0.30) / 0.70, 0.0, 1.0))
