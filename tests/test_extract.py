from video_quality_filter.extract import frame_timestamps


def test_timestamps_are_evenly_spaced_inside_the_clip():
    stamps = frame_timestamps(10.0, 4)
    assert len(stamps) == 4
    assert stamps[0] > 0
    assert stamps[-1] < 10
    gaps = [right - left for left, right in zip(stamps, stamps[1:])]
    assert max(gaps) - min(gaps) < 1e-9


def test_single_timestamp_is_the_midpoint():
    assert frame_timestamps(3.0, 1) == [1.5]


def test_empty_duration_has_no_timestamps():
    assert frame_timestamps(0.0, 4) == []
