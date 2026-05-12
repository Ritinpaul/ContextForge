from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .embeddings import Embedder
from .llm import LLM, LLMConfig
from .phase2_graph import Phase2Deps, run_phase2
from .qdrant_store import QdrantStore
from .rerank import CrossEncoderReranker
from .sparse_bm25 import BM25Index
from .hybrid import RRFConfig


@dataclass(frozen=True)
class AskResult:
    answer: str
    sources: list[str]
    faithfulness: float | None
    retries: int


def ask(question: str, *, settings: Settings, debug: bool = False) -> AskResult:
    embedder = Embedder(settings.embedding_model, device=settings.embedding_device)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection)

    # Build sparse BM25 corpus from all stored chunks.
    points = list(store.iter_points(limit=256, with_payload=True))
    bm25 = BM25Index.from_points(points)

    llm = LLM(
        LLMConfig(
            model=settings.llm_model,
            task=settings.llm_task,
            max_new_tokens=settings.llm_max_new_tokens,
            temperature=settings.llm_temperature,
        )
    )

    reranker = CrossEncoderReranker(settings.reranker_model)

    rrf = RRFConfig(
        k=settings.rrf_k,
        weight_dense=settings.rrf_weight_dense,
        weight_sparse=settings.rrf_weight_sparse,
    )

    def dense_search_fn(query_vec, limit: int):
        return store.search(query_vector=query_vec, limit=limit, with_payload=True)

    def embed_text_fn(text: str):
        return embedder.embed_texts([text])[0]

    deps = Phase2Deps(
        llm=llm,
        bm25=bm25,
        rrf=rrf,
        reranker=reranker,
        dense_search_fn=dense_search_fn,
        embed_text_fn=embed_text_fn,
        top_k_dense=settings.hybrid_dense_top_k,
        top_k_sparse=settings.hybrid_sparse_top_k,
        top_k_fused=settings.hybrid_fused_top_k,
        top_k_final=settings.final_top_k,
        faithfulness_threshold=settings.faithfulness_threshold,
        max_retries=settings.faithfulness_max_retries,
    )

    state = run_phase2(question, deps=deps, debug=debug)

    answer = (state.get("answer") or "").strip()
    sources = list(state.get("sources") or [])

    # If we couldn't compute faithfulness (missing deps), return answer anyway.
    faith = state.get("faithfulness_score")

    return AskResult(
        answer=answer,
        sources=sources,
        faithfulness=faith,
        retries=int(state.get("retries") or 0),
    )
