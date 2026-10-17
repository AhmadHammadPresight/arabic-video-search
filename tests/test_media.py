from __future__ import annotations

import numpy as np
import pytest

from avs.media import MediaError, encode_h264, iter_frames, probe


def test_probe_reads_size_rate_and_duration(moving_box_video):
    info = probe(moving_box_video)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(10)
    assert info.duration == pytest.approx(3, abs=0.2)


def test_iter_frames_samples_at_the_requested_rate(moving_box_video):
    frames = list(iter_frames(moving_box_video, sample_fps=4))
    times = [t for t, _ in frames]
    assert times[:3] == [0.0, 0.25, 0.5]
    assert len(frames) == pytest.approx(12, abs=1)
    _, frame = frames[0]
    assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8
    # The red box starts near x=20..60 and moves right.
    red = (frame[..., 0] > 150) & (frame[..., 1] < 80)
    assert red[:, 20:60].any() and not red[:, 200:].any()


def test_encode_h264_trims_and_drops_audio(moving_box_video, tmp_path):
    out = tmp_path / "clip.mp4"
    encode_h264(moving_box_video, out, start=1.0, duration=1.0)
    info = probe(out)
    assert info.codec == "h264"
    assert info.duration == pytest.approx(1.0, abs=0.2)
    assert not out.with_name("clip.partial.mp4").exists()


def test_probe_rejects_non_video(tmp_path):
    bogus = tmp_path / "notes.mp4"
    bogus.write_text("not a video")
    with pytest.raises(MediaError):
        probe(bogus)
