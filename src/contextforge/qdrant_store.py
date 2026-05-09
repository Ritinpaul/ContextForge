from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


@dataclass
class QdrantStore:
    url: str
    api_key: str | None
    collection: str

    def __post_init__(self) -> None:
        self.client = QdrantClient(url=self.url, api_key=self.api_key)

    def ensure_collection(self, *, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        if any(c.name == self.collection for c in collections):
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert_chunks(
        self,
        *,
        ids: list[str],
        vectors: np.ndarray,
        payloads: list[dict[str, Any]],
    ) -> None:
        if len(ids) != len(payloads):
            raise ValueError("ids and payloads length mismatch")
        if vectors.shape[0] != len(ids):
            raise ValueError("vectors and ids length mismatch")

        points: list[PointStruct] = []
        for idx, point_id in enumerate(ids):
            points.append(
                PointStruct(id=point_id, vector=vectors[idx].tolist(), payload=payloads[idx])
            )

        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        *,
        query_vector: np.ndarray,
        limit: int = 10,
        with_payload: bool = True,
    ) -> list[dict[str, Any]]:
        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector.tolist(),
                limit=limit,
                with_payload=with_payload,
                with_vectors=False,
            )
            hits = response.points
        except AttributeError:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector.tolist(),
                limit=limit,
                with_payload=with_payload,
            )
        out: list[dict[str, Any]] = []
        for hit in hits:
            out.append(
                {
                    "id": str(hit.id),
                    "score": float(hit.score),
                    "payload": dict(hit.payload or {}),
                }
            )
        return out
