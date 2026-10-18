"""HTTP client for the native inference service."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx


class InferenceError(RuntimeError):
    pass


class InferenceClient:
    def __init__(self, base_url: str) -> None:
        # Indexing streams for minutes; only the gap between two events is bounded.
        self._client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(30.0, read=600.0))

    def health(self) -> dict:
        response = self._client.get("/health", timeout=5.0)
        response.raise_for_status()
        return response.json()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client.post("/v1/embed/text", json={"texts": texts})
        if response.status_code != 200:
            raise InferenceError(f"embedding failed: {response.status_code} {response.text}")
        return response.json()["embeddings"]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def check(self, crops: list[str], subject: str, english: str, details: list[str]) -> dict:
        body = {"crops": crops, "subject": subject, "english": english, "details": details}
        response = self._client.post("/v1/check", json=body, timeout=120.0)
        if response.status_code != 200:
            raise InferenceError(f"double-check failed: {response.status_code} {response.text}")
        return response.json()

    def describe(self, crops: list[str], subject: str, label: str, method: str) -> dict:
        body = {"crops": crops, "subject": subject, "label": label, "method": method}
        response = self._client.post("/v1/describe", json=body, timeout=120.0)
        if response.status_code != 200:
            raise InferenceError(f"describe failed: {response.status_code} {response.text}")
        return response.json()

    def trace_frames(self, recording_id: str, video_path: str, times: list[float]) -> dict:
        body = {"recording_id": recording_id, "video_path": video_path, "times": times}
        response = self._client.post("/v1/trace/frames", json=body, timeout=120.0)
        if response.status_code != 200:
            raise InferenceError(f"frame trace failed: {response.status_code} {response.text}")
        return response.json()

    def index(self, recording_id: str, video_path: str, sample_fps: float | None = None) -> Iterator[dict]:
        body = {"recording_id": recording_id, "video_path": video_path, "sample_fps": sample_fps}
        with self._client.stream("POST", "/v1/index", json=body) as response:
            if response.status_code != 200:
                response.read()
                raise InferenceError(f"indexing refused: {response.status_code} {response.text}")
            for line in response.iter_lines():
                if line.strip():
                    event = json.loads(line)
                    if event.get("type") == "error":
                        raise InferenceError(event.get("message", "indexing failed"))
                    yield event
