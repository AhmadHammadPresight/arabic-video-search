"""Qdrant collection of crop embeddings: one point per crop, 3-5 points per track.

Each point carries the camera, place, absolute times and a pointer back to its recording,
so one query can filter by camera and time and then rank by cosine similarity.
Results are grouped by track, so a person is listed once with their best-matching crop.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

POINT_NAMESPACE = uuid.UUID("6b2f0f0e-6c7e-4d4b-9d63-3f1c2a1b5e10")


@dataclass(frozen=True)
class Hit:
    score: float
    payload: dict


def point_id(track_key: str, crop_index: int) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, f"{track_key}#{crop_index}"))


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
        )
        for field, schema in (
            ("camera_id", models.PayloadSchemaType.KEYWORD),
            ("recording_id", models.PayloadSchemaType.KEYWORD),
            ("track_key", models.PayloadSchemaType.KEYWORD),
            ("label", models.PayloadSchemaType.KEYWORD),
            ("abs_start", models.PayloadSchemaType.FLOAT),
            ("abs_end", models.PayloadSchemaType.FLOAT),
        ):
            self.client.create_payload_index(self.collection, field, field_schema=schema)

    def replace_recording(self, recording_id: str, points: list[models.PointStruct]) -> None:
        self.delete_recording(recording_id)
        for start in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[start : start + 256], wait=True)

    def delete_recording(self, recording_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=_match("recording_id", recording_id)),
            wait=True,
        )

    def search(
        self,
        vector: list[float],
        limit: int,
        cameras: list[str] | None = None,
        abs_from: float | None = None,
        abs_to: float | None = None,
    ) -> list[Hit]:
        result = self.client.query_points_groups(
            self.collection,
            query=vector,
            group_by="track_key",
            group_size=1,
            limit=limit,
            query_filter=build_filter(cameras, abs_from, abs_to),
            with_payload=True,
        )
        return [Hit(score=g.hits[0].score, payload=g.hits[0].payload or {}) for g in result.groups if g.hits]


def build_filter(cameras: list[str] | None, abs_from: float | None, abs_to: float | None) -> models.Filter | None:
    """Camera filter plus time-window overlap: a track counts if any part of it falls in the window."""
    must: list[models.Condition] = []
    if cameras:
        must.append(models.FieldCondition(key="camera_id", match=models.MatchAny(any=list(cameras))))
    if abs_from is not None:
        must.append(models.FieldCondition(key="abs_end", range=models.Range(gte=abs_from)))
    if abs_to is not None:
        must.append(models.FieldCondition(key="abs_start", range=models.Range(lte=abs_to)))
    return models.Filter(must=must) if must else None


def _match(key: str, value: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))])
