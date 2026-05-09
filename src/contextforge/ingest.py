from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunking import chunk_text
from .config import Settings
from .embeddings import Embedder
from .loaders import load_document
from .qdrant_store import QdrantStore


@dataclass
class IngestResult:
    files: int
    chunks: int


def _iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]

    exts = {".md", ".markdown", ".html", ".htm", ".pdf", ".docx"}
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def _stable_chunk_id(source_path: str, chunk_index: int) -> str:
    # Qdrant point IDs must be either an unsigned int or a UUID.
    # Use a deterministic UUID so re-ingesting produces stable IDs.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_path}::{chunk_index}"))


def ingest_path(root: Path, *, settings: Settings) -> IngestResult:
    files = _iter_files(root)
    if not files:
        return IngestResult(files=0, chunks=0)

    embedder = Embedder(settings.embedding_model, device=settings.embedding_device)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection)
    store.ensure_collection(vector_size=embedder.dim)

    total_chunks = 0

    for f in files:
        loaded = load_document(f, max_chars=settings.ingest_max_chars)
        chunks = chunk_text(
            loaded.text,
            chunk_tokens=settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        if not chunks:
            continue

        texts = [c.text for c in chunks]
        vectors = embedder.embed_texts(texts)

        ids: list[str] = []
        payloads: list[dict[str, Any]] = []

        for c in chunks:
            ids.append(_stable_chunk_id(loaded.source_path, c.index))
            payloads.append(
                {
                    "source_path": loaded.source_path,
                    "source_type": loaded.source_type,
                    "title": loaded.title,
                    "chunk_index": c.index,
                    "chunk_start_token": c.start_token,
                    "chunk_end_token": c.end_token,
                    "text": c.text,
                }
            )

        store.upsert_chunks(ids=ids, vectors=vectors, payloads=payloads)
        total_chunks += len(chunks)

    return IngestResult(files=len(files), chunks=total_chunks)
