from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossEncoderReranker:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __post_init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        *,
        query: str,
        texts: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Return (index, score) pairs for the top_k texts."""

        if not texts:
            return []

        pairs = [(query, t) for t in texts]
        scores = self._model.predict(pairs)

        ranked = sorted(range(len(texts)), key=lambda i: float(scores[i]), reverse=True)
        ranked = ranked[:top_k]
        return [(i, float(scores[i])) for i in ranked]
