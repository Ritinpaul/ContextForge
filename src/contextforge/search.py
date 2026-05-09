from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .embeddings import Embedder
from .qdrant_store import QdrantStore


@dataclass
class SearchHit:
    id: str
    score: float
    source_path: str | None
    chunk_index: int | None
    text: str | None


def search(
    query: str,
    *,
    settings: Settings,
    limit: int = 10,
) -> list[SearchHit]:
    embedder = Embedder(settings.embedding_model, device=settings.embedding_device)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection)

    qv = embedder.embed_texts([query])[0]
    results = store.search(query_vector=qv, limit=limit, with_payload=True)

    hits: list[SearchHit] = []
    for r in results:
        payload = r.get("payload") or {}
        hits.append(
            SearchHit(
                id=r["id"],
                score=r["score"],
                source_path=payload.get("source_path"),
                chunk_index=payload.get("chunk_index"),
                text=payload.get("text"),
            )
        )
    return hits
