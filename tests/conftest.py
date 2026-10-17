from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from avs.api.vectors import VectorStore
from avs.settings import Settings, get_settings

DIM = 4


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data = tmp_path / "data"
    data.mkdir()
    return dataclasses.replace(get_settings(), data_dir=data, db_path=data / "avs.db")


@pytest.fixture
def store() -> VectorStore:
    vs = VectorStore(QdrantClient(":memory:"), "crops", DIM)
    vs.ensure_collection()
    return vs


@pytest.fixture(scope="session")
def moving_box_video(tmp_path_factory) -> Path:
    """3 s, 10 fps, 320x240 clip: a red box moving left to right on grey."""
    path = tmp_path_factory.mktemp("video") / "moving_box.mp4"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=10:d=3",
            "-f", "lavfi", "-i", "color=c=red:s=40x120:r=10:d=3",
            "-filter_complex", "[0][1]overlay=x='20+t*60':y=60:eval=frame",
            "-c:v", "h264_videotoolbox", "-b:v", "1M", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    return path
