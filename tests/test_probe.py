from pathlib import Path

from video_quality_filter.probe import parse_probe_payload


def test_parse_ignores_attached_picture_and_reads_audio_duration():
    payload = {
        "format": {"duration": "12.4", "format_name": "mov,mp4,m4a,3gp,3g2,mj2", "bit_rate": "1000"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "mjpeg",
                "width": 120,
                "height": 120,
                "avg_frame_rate": "0/0",
                "disposition": {"attached_pic": 1},
            },
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "60/1",
                "pix_fmt": "yuv420p",
                "duration": "12.4",
            },
            {"codec_type": "audio", "codec_name": "aac", "duration": "12.2"},
        ],
    }
    meta = parse_probe_payload(Path("/tmp/clip.mp4"), 42, payload)
    assert meta.video_codec == "h264"
    assert meta.width == 1920
    assert meta.height == 1080
    assert meta.fps == 60
    assert meta.has_audio is True
    assert meta.audio_duration_sec == 12.2
    assert meta.duration_sec == 12.4
    assert meta.probe_error is None
