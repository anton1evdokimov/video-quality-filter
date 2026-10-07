import numpy as np

from video_quality_filter.audio_quality import analyze_waveform
from video_quality_filter.config import AudioConfig


def _tone(amplitude: float, seconds: float = 1.0, rate: int = 16000) -> np.ndarray:
    time = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * 440 * time)).astype(np.float32)


def test_raw_audio_features_are_kept_and_silence_is_worse_than_a_tone():
    config = AudioConfig()
    silence = analyze_waveform(np.zeros(16000, dtype=np.float32), 16000, config)
    tone = analyze_waveform(_tone(0.2), 16000, config)
    clipped = analyze_waveform(np.clip(_tone(3.0), -1.0, 1.0), 16000, config)
    for features in (silence, tone, clipped):
        assert set(features) >= {"silence_ratio", "rms", "clipping_ratio", "audio_quality"}
    assert silence["silence_ratio"] > 0.9
    assert silence["rms"] < tone["rms"]
    assert silence["audio_quality"] < tone["audio_quality"]
    assert clipped["clipping_ratio"] > 0.02
    assert clipped["audio_quality"] < tone["audio_quality"]
