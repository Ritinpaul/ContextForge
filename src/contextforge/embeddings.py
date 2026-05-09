from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Embedder:
    model_name: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def dim(self) -> int:
        # sentence-transformers renamed this method; keep backward compatibility.
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if callable(get_dim):
            return int(get_dim())
        return int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)
