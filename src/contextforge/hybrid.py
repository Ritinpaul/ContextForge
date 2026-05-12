from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RRFConfig:
    k: int = 60
    weight_dense: float = 0.7
    weight_sparse: float = 0.3


def reciprocal_rank_fusion(
    *,
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    config: RRFConfig,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Fuse two ranked lists using weighted RRF.

    Each hit dict is expected to contain: id, payload, rank (1-based).
    """

    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for hit in dense_hits:
        doc_id = str(hit["id"])
        rank = int(hit.get("rank") or 10**9)
        scores[doc_id] = scores.get(doc_id, 0.0) + config.weight_dense * (1.0 / (config.k + rank))
        payloads.setdefault(doc_id, dict(hit.get("payload") or {}))

    for hit in sparse_hits:
        doc_id = str(hit["id"])
        rank = int(hit.get("rank") or 10**9)
        scores[doc_id] = scores.get(doc_id, 0.0) + config.weight_sparse * (1.0 / (config.k + rank))
        payloads.setdefault(doc_id, dict(hit.get("payload") or {}))

    ranked_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)[:top_k]

    out: list[dict[str, Any]] = []
    for idx, doc_id in enumerate(ranked_ids, start=1):
        out.append(
            {
                "id": doc_id,
                "score": float(scores[doc_id]),
                "rank": idx,
                "payload": payloads.get(doc_id, {}),
            }
        )
    return out
