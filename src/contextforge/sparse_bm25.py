from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class BM25Index:
    """In-memory BM25 over chunk texts.

    This is intentionally simple (Phase 2). It rebuilds from Qdrant payloads
    on demand and keeps everything in memory.
    """

    ids: list[str]
    texts: list[str]
    payloads: list[dict[str, Any]]

    def __post_init__(self) -> None:
        if not (len(self.ids) == len(self.texts) == len(self.payloads)):
            raise ValueError("BM25Index length mismatch")

        from rank_bm25 import BM25Okapi

        self._tokenized = [_tokenize(t) for t in self.texts]
        self._bm25 = BM25Okapi(self._tokenized)

    @classmethod
    def from_points(cls, points: Iterable[dict[str, Any]]) -> "BM25Index":
        ids: list[str] = []
        texts: list[str] = []
        payloads: list[dict[str, Any]] = []

        for p in points:
            payload = dict(p.get("payload") or {})
            text = str(payload.get("text") or "")
            if not text.strip():
                continue
            ids.append(str(p.get("id")))
            texts.append(text)
            payloads.append(payload)

        return cls(ids=ids, texts=texts, payloads=payloads)

    def search(self, query: str, *, top_k: int = 50) -> list[dict[str, Any]]:
        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)

        # scores is numpy array
        ranked = sorted(range(len(self.ids)), key=lambda i: float(scores[i]), reverse=True)
        ranked = ranked[:top_k]

        out: list[dict[str, Any]] = []
        for rank, i in enumerate(ranked, start=1):
            out.append(
                {
                    "id": self.ids[i],
                    "score": float(scores[i]),
                    "rank": rank,
                    "payload": self.payloads[i],
                }
            )
        return out
