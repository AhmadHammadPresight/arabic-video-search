"""SigLIP 2 image and text embeddings (google/siglip2-so400m-patch16-384, Apache-2.0).

Image crops and Arabic or English text land in the same vector space, so a text query
ranks crops by cosine similarity. Vectors are L2-normalised before they leave this module.
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np
from PIL import Image

# SigLIP 2 was trained with text padded to 64 tokens; shorter padding hurts retrieval.
TEXT_MAX_LENGTH = 64


class Embedder(Protocol):
    dim: int

    def embed_images(self, images: list[Image.Image]) -> np.ndarray: ...

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class SigLIP2Embedder:
    def __init__(self, model_name: str, revision: str, device: str, batch_size: int = 16) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.name = model_name
        self.device = device
        self.batch_size = batch_size
        self.dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        self._model = AutoModel.from_pretrained(model_name, revision=revision, dtype=self.dtype).to(device).eval()
        self._processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.dim = int(self._model.config.text_config.projection_size)
        # The learned temperature of SigLIP's image-text logits, used for zero-shot attribute prompts.
        self.logit_scale = float(self._model.logit_scale.exp().item())
        self._lock = threading.Lock()

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        for start in range(0, len(images), self.batch_size):
            batch = [im.convert("RGB") for im in images[start : start + self.batch_size]]
            inputs = self._processor(images=batch, return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.device, self.dtype)
            with self._lock, self._torch.inference_mode():
                features = self._model.get_image_features(pixel_values=pixels)
            out.append(_features(features).float().cpu().numpy())
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.concatenate(out))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # The model card asks for lower-cased text; Arabic has no case, so this only touches Latin script.
        inputs = self._processor(
            text=[t.lower() for t in texts], padding="max_length", max_length=TEXT_MAX_LENGTH,
            truncation=True, return_tensors="pt",
        ).to(self.device)
        with self._lock, self._torch.inference_mode():
            features = self._model.get_text_features(**inputs)
        return _normalise(_features(features).float().cpu().numpy())


def _features(output):
    # transformers 5 returns a model output object; the projected embedding is its pooler_output.
    return getattr(output, "pooler_output", output)
