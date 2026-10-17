"""Runtime settings, read once from environment variables.

Every service reads the same variables, so one `scripts/start.sh` run configures them all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    qdrant_url: str
    qdrant_collection: str
    inference_url: str
    # Indexing
    sample_fps: float
    detector: str
    detection_threshold: float
    crops_per_track: int
    min_track_detections: int
    device: str
    embed_model: str
    embed_revision: str

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def playback_dir(self) -> Path:
        return self.data_dir / "playback"

    @property
    def crops_dir(self) -> Path:
        return self.data_dir / "crops"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def resolve(self, relative: str) -> Path:
        """Turn a path stored relative to the data directory into a safe absolute path."""
        path = (self.data_dir / relative).resolve()
        if not path.is_relative_to(self.data_dir.resolve()):
            raise ValueError(f"path escapes the data directory: {relative}")
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(_env("AVS_DATA_DIR", str(REPO_ROOT / "data"))).resolve()
    return Settings(
        data_dir=data_dir,
        db_path=Path(_env("AVS_DB_PATH", str(data_dir / "avs.db"))),
        qdrant_url=_env("AVS_QDRANT_URL", "http://127.0.0.1:6433"),
        qdrant_collection=_env("AVS_QDRANT_COLLECTION", "crops"),
        inference_url=_env("AVS_INFERENCE_URL", "http://127.0.0.1:8471"),
        sample_fps=float(_env("AVS_SAMPLE_FPS", "4")),
        detector=_env("AVS_DETECTOR", "rfdetr-medium"),
        detection_threshold=float(_env("AVS_DETECTION_THRESHOLD", "0.35")),
        crops_per_track=int(_env("AVS_CROPS_PER_TRACK", "5")),
        min_track_detections=int(_env("AVS_MIN_TRACK_DETECTIONS", "3")),
        device=_env("AVS_DEVICE", "mps"),
        embed_model=_env("AVS_EMBED_MODEL", "google/siglip2-so400m-patch16-384"),
        # Pinned Hugging Face commit, so every Mac embeds with exactly the same weights.
        embed_revision=_env("AVS_EMBED_REVISION", "dd658faac399427308559e2c3ac1e99cbe43845d"),
    )
