"""Fetch the model weights once, during setup; the services then run with the hub offline.

- SigLIP 2 So400m (Apache-2.0) at the pinned revision, into the Hugging Face cache.
- RF-DETR COCO weights (Apache-2.0) into data/models; rfdetr checks their MD5 on download.
"""

from __future__ import annotations

from huggingface_hub import snapshot_download

from avs.inference.detector import RFDETRDetector
from avs.settings import get_settings


def main() -> None:
    settings = get_settings()
    path = snapshot_download(
        settings.embed_model,
        revision=settings.embed_revision,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )
    print(f"SigLIP 2 ready: {settings.embed_model}@{settings.embed_revision[:10]} in {path}")
    RFDETRDetector(settings.detector, settings.models_dir, "cpu", settings.detection_threshold)
    print(f"RF-DETR ready: {settings.detector} in {settings.models_dir}")


if __name__ == "__main__":
    main()
