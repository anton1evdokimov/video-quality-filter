import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tests.media_factory import ffmpeg_available, render_video
from video_quality_filter.cli import main
from video_quality_filter.config import AppConfig
from video_quality_filter.pipeline import PUBLIC_KEYS, discover_videos, run_pipeline

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg and ffprobe are required")


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir()) if path.is_file()}


def test_discover_skips_other_extensions_and_the_output_directory(tmp_path: Path):
    (tmp_path / "keep.mp4").write_bytes(b"1")
    (tmp_path / "note.txt").write_text("no", encoding="utf-8")
    nested = tmp_path / "reports"
    nested.mkdir()
    (nested / "old.mp4").write_bytes(b"2")
    found = discover_videos(tmp_path, ["mp4"], True, nested)
    assert [path.name for path in found] == ["keep.mp4"]


def test_pipeline_writes_metadata_and_preserves_sources(tmp_path: Path):
    videos = tmp_path / "videos"
    videos.mkdir()
    render_video(videos / "good.mp4", audio="tone")
    shutil.copyfile(videos / "good.mp4", videos / "twin.mp4")
    render_video(videos / "no_audio.mp4", audio="none")
    render_video(videos / "black.mp4", color="black", audio="tone")
    render_video(videos / "tiny.mp4", size="64x64", color="blue", audio="none")
    render_video(videos / "short.mp4", duration=0.3, audio="tone")
    (videos / "broken.mp4").write_bytes(b"this is not a video")
    before = _snapshot(videos)

    output = tmp_path / "out"
    summary = run_pipeline(videos, AppConfig(output_dir=str(output)))

    assert _snapshot(videos) == before
    assert summary.total == 7
    assert summary.parquet_path.is_file()
    rows = [json.loads(line) for line in summary.jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert [set(row) for row in rows] == [set(PUBLIC_KEYS)] * len(rows)
    by_name = {row["video_id"]: row for row in rows}

    good = by_name["good"]
    assert good["filtering"]["status"] == "accepted", good
    assert good["source"] == "good.mp4"
    assert good["technical"]["technical_ok"] is True
    assert good["technical"]["fps_original"] > 24
    assert good["technical"]["fps_processed"] == 24
    assert good["technical"]["audio_present"] is True
    assert good["technical"]["audio_duration"] is not None
    assert {"silence_ratio", "rms", "clipping_ratio", "audio_quality"} <= set(good["audio"])
    assert good["vlm"]["caption"] is None
    assert good["video_text"]["cosine_similarity"] is None
    assert "quality_score" not in good

    twin = by_name["twin"]
    assert twin["filtering"]["status"] == "accepted", twin
    assert good["deduplication"]["is_near_duplicate"] is True
    assert twin["deduplication"]["is_near_duplicate"] is True
    assert good["deduplication"]["cluster_id"] == twin["deduplication"]["cluster_id"]
    assert good["deduplication"]["similarity"] >= 0.98
    assert twin["deduplication"]["similarity"] >= 0.98

    no_audio = by_name["no_audio"]
    assert no_audio["filtering"]["status"] == "accepted", no_audio
    assert no_audio["technical"]["audio_present"] is False
    assert no_audio["audio"]["rms"] is None

    assert by_name["black"]["technical"]["technical_ok"] is True
    assert by_name["black"]["filtering"]["status"] == "rejected", by_name["black"]
    assert "aesthetic_score_below_threshold" in by_name["black"]["filtering"]["reasons"]
    assert "resolution_too_small" in by_name["tiny"]["filtering"]["reasons"]
    assert by_name["tiny"]["technical"]["technical_ok"] is False
    assert "duration_too_short" in by_name["short"]["filtering"]["reasons"]
    assert "unreadable" in by_name["broken"]["filtering"]["reasons"]
    assert by_name["broken"]["visual"]["aesthetic_score"] is None

    table = pq.read_table(summary.parquet_path).to_pylist()
    assert {row["video_id"] for row in table} == set(by_name)
    assert table[0]["filtering"]["status"] in {"accepted", "rejected"}


def test_cli_writes_jsonl_and_parquet(tmp_path: Path):
    videos = tmp_path / "videos"
    videos.mkdir()
    render_video(videos / "clip.MP4", audio="tone")
    output = tmp_path / "reports"
    code = main(["run", "--input", str(videos), "--output-dir", str(output)])
    assert code == 0
    assert (output / "results.jsonl").is_file()
    assert (output / "results.parquet").is_file()
    assert (videos / "clip.MP4").is_file()


def test_output_directory_cannot_contain_the_input(tmp_path: Path):
    videos = tmp_path / "videos"
    videos.mkdir()
    with pytest.raises(ValueError):
        run_pipeline(videos, AppConfig(output_dir=str(tmp_path)))
