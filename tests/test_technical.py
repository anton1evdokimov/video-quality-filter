import pytest

from video_quality_filter.config import FilteringConfig
from video_quality_filter.models import VideoMetadata
from video_quality_filter.probe import _fps, _ratio
from video_quality_filter.technical import build_technical, processed_fps


def _meta(**overrides) -> VideoMetadata:
    payload = dict(
        path="/tmp/clip.mp4",
        file_size_bytes=1000,
        duration_sec=12.4,
        audio_duration_sec=12.2,
        width=1920,
        height=1080,
        fps=60.0,
        video_codec="h264",
        audio_codec="aac",
        has_audio=True,
    )
    payload.update(overrides)
    return VideoMetadata(**payload)


def test_high_fps_is_normalized_and_not_rejected():
    technical, reasons = build_technical(_meta(), FilteringConfig(), target_fps=24)
    assert reasons == []
    assert technical["technical_ok"] is True
    assert technical["fps_original"] == 60
    assert technical["fps_processed"] == 24
    assert technical["audio_present"] is True
    assert technical["audio_duration"] == 12.2
    assert technical["av_duration_diff"] == pytest.approx(0.2)
    assert technical["codec"] == "h264"


def test_low_fps_short_and_small_video_are_rejected():
    technical, reasons = build_technical(
        _meta(duration_sec=0.2, fps=4, width=64, height=64, has_audio=False, audio_duration_sec=None),
        FilteringConfig(require_audio=True),
        target_fps=24,
    )
    assert technical["technical_ok"] is False
    assert technical["fps_processed"] == 4
    assert reasons == [
        "duration_too_short",
        "resolution_too_small",
        "fps_too_low",
        "audio_required_but_missing",
    ]


def test_bad_codec_and_unreadable_file():
    _technical, codec_reasons = build_technical(_meta(video_codec="mjpeg"), FilteringConfig(), target_fps=24)
    assert "codec_not_allowed" in codec_reasons
    technical, reasons = build_technical(
        _meta(file_size_bytes=0, probe_error="boom", video_codec=None, width=None, height=None),
        FilteringConfig(),
        target_fps=24,
    )
    assert reasons == ["unreadable"]
    assert technical["technical_ok"] is False


def test_h265_alias_is_hevc():
    technical, reasons = build_technical(_meta(video_codec="h265"), FilteringConfig(), target_fps=24)
    assert reasons == []
    assert technical["codec"] == "hevc"


def test_processed_fps_does_not_upsample():
    assert processed_fps(12.0, 24) == 12
    assert processed_fps(None, 24) is None


def test_fps_ratio_parser():
    assert _ratio("30000/1001") == pytest.approx(30000 / 1001)
    assert _ratio("0/0") is None
    assert _fps({"avg_frame_rate": "0/0", "r_frame_rate": "25/1"}, 2.0) == 25.0
