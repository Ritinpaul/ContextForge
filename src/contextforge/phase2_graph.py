from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from .faithfulness import compute_faithfulness_ragas
from .hybrid import RRFConfig, reciprocal_rank_fusion
from .llm import LLM, LLMConfig
from .rerank import CrossEncoderReranker
from .sparse_bm25 import BM25Index


QueryType = Literal["factual", "multi-hop", "comparative", "summarization", "ambiguous"]


class RAGState(TypedDict, total=False):
    question: str
    query_type: QueryType
    sub_questions: list[str]
    hyde_queries: list[str]
    dense_hits: list[dict[str, Any]]
    sparse_hits: list[dict[str, Any]]
    fused_hits: list[dict[str, Any]]
    reranked_hits: list[dict[str, Any]]
    contexts: list[str]
    answer: str
    sources: list[str]
    faithfulness_score: float | None
    retries: int
    debug: dict[str, Any]


_CLASSIFY_EXAMPLES = """
You are a query classifier for a RAG system.

Classify the user question into exactly one label from:
- factual
- multi-hop
- comparative
- summarization
- ambiguous

Definitions:
- factual: asks for a specific fact, definition, or direct lookup
- multi-hop: requires combining multiple pieces of information or steps
- comparative: asks to compare two or more items
- summarization: asks to summarize a document or content
- ambiguous: question is underspecified or needs clarification

Examples (3 each):

factual:
Q: What is the default port for Qdrant?
A: factual
Q: Define reciprocal rank fusion.
A: factual
Q: What does BM25 stand for?
A: factual

multi-hop:
Q: How do I ingest PDFs into Qdrant and then evaluate recall@10?
A: multi-hop
Q: Given a query, how do I expand it with HyDE and then fuse dense+sparse results?
A: multi-hop
Q: What steps are needed to build a RAG pipeline with reranking and a faithfulness check?
A: multi-hop

comparative:
Q: Compare BM25 and dense embeddings for retrieval.
A: comparative
Q: Compare RRF and simple score averaging for fusion.
A: comparative
Q: Compare cross-encoder reranking vs bi-encoder retrieval.
A: comparative

summarization:
Q: Summarize the key points of this document.
A: summarization
Q: Give me a short summary of the provided meeting notes.
A: summarization
Q: Summarize the main arguments in the text.
A: summarization

ambiguous:
Q: Tell me about it.
A: ambiguous
Q: What should I do next?
A: ambiguous
Q: Is this good?
A: ambiguous

Now classify this question. Output ONLY the label.
Question: """


def _parse_label(text: str) -> QueryType:
    t = text.strip().lower()
    for label in ("factual", "multi-hop", "comparative", "summarization", "ambiguous"):
        if label in t:
            return label  # type: ignore[return-value]
    return "ambiguous"  # type: ignore[return-value]


_BULLET_RE = re.compile(r"^\s*[-*\d.)]+\s*")


def _parse_lines(text: str) -> list[str]:
    lines = []
    for raw in (text or "").splitlines():
        s = _BULLET_RE.sub("", raw).strip()
        if s:
            lines.append(s)
    return lines


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = it.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


@dataclass
class Phase2Deps:
    llm: LLM
    bm25: BM25Index
    rrf: RRFConfig
    reranker: CrossEncoderReranker
    dense_search_fn: Any  # callable(query_vec, limit)->list[dict]
    embed_text_fn: Any  # callable(text)->np.ndarray
    top_k_dense: int = 50
    top_k_sparse: int = 50
    top_k_fused: int = 20
    top_k_final: int = 5
    faithfulness_threshold: float = 0.75
    max_retries: int = 2


def node_classify_query(state: RAGState, deps: Phase2Deps) -> RAGState:
    q = state["question"]
    out = deps.llm.generate(_CLASSIFY_EXAMPLES + q)
    label = _parse_label(out)

    debug = dict(state.get("debug") or {})
    debug["classify_raw"] = out

    return {"query_type": label, "debug": debug}


def node_hyde_expand(state: RAGState, deps: Phase2Deps) -> RAGState:
    q = state["question"]
    qt = state.get("query_type") or "ambiguous"
    retries = int(state.get("retries") or 0)

    if qt == "multi-hop":
        prompt = (
            "You create sub-questions for multi-hop retrieval.\n"
            "Return 3-5 short sub-questions, one per line.\n"
            f"Question: {q}\n"
        )
        sub_raw = deps.llm.generate(prompt)
        subs = _parse_lines(sub_raw)[:5]
        subs = _dedupe_keep_order(subs)
        if not subs:
            subs = [q]

        # For each sub-question, create a HyDE hypothetical answer.
        hyde_queries: list[str] = []
        for sq in subs:
            hyde_prompt = (
                "Write a plausible, concise hypothetical answer to the question. "
                "Do not cite sources. 2-5 sentences.\n"
                f"Question: {sq}\n"
            )
            hyde = deps.llm.generate(hyde_prompt, temperature=0.7)
            hyde_queries.append(hyde)

        debug = dict(state.get("debug") or {})
        debug["sub_questions_raw"] = sub_raw
        debug["hyde_queries"] = hyde_queries

        return {"sub_questions": subs, "hyde_queries": hyde_queries, "debug": debug}

    # Non-multi-hop: single HyDE expansion
    style_hint = ""
    if qt == "summarization":
        style_hint = " Focus on capturing key points and structure."
    if qt == "comparative":
        style_hint = " Focus on criteria and differences."
    if qt == "ambiguous":
        style_hint = " Make reasonable assumptions and note ambiguity briefly."

    # On retries, nudge HyDE to be more specific.
    retry_hint = ""
    if retries > 0:
        retry_hint = " Be more specific and include key terms and constraints."

    hyde_prompt = (
        "Write a plausible, concise hypothetical answer to the question." + style_hint + retry_hint + "\n"
        "Do not cite sources. 2-6 sentences.\n"
        f"Question: {q}\n"
    )
    hyde = deps.llm.generate(hyde_prompt, temperature=0.7)

    debug = dict(state.get("debug") or {})
    debug["hyde_query"] = hyde

    return {"sub_questions": [], "hyde_queries": [hyde], "debug": debug}


def node_hybrid_retrieve(state: RAGState, deps: Phase2Deps) -> RAGState:
    q = state["question"]
    subs = state.get("sub_questions") or []
    hyde_queries = state.get("hyde_queries") or [q]

    dense_hits_all: list[dict[str, Any]] = []
    sparse_hits_all: list[dict[str, Any]] = []

    # Dense retrieval: embed HyDE text(s)
    for hq in hyde_queries:
        qv = deps.embed_text_fn(hq)
        hits = deps.dense_search_fn(qv, deps.top_k_dense)
        dense_hits_all.extend(hits)

    # Sparse retrieval: use original question and sub-questions
    sparse_queries = [q] + subs
    for sq in sparse_queries:
        hits = deps.bm25.search(sq, top_k=deps.top_k_sparse)
        sparse_hits_all.extend(hits)

    # Rank annotate dense hits (Qdrant gives its own score; we treat its order as rank)
    dense_ranked: list[dict[str, Any]] = []
    for rank, hit in enumerate(dense_hits_all, start=1):
        dense_ranked.append(
            {
                "id": str(hit["id"]),
                "score": float(hit.get("score") or 0.0),
                "rank": rank,
                "payload": dict(hit.get("payload") or {}),
            }
        )

    # Sparse already has rank
    sparse_ranked: list[dict[str, Any]] = []
    for rank, hit in enumerate(sparse_hits_all, start=1):
        sparse_ranked.append(
            {
                "id": str(hit["id"]),
                "score": float(hit.get("score") or 0.0),
                "rank": rank,
                "payload": dict(hit.get("payload") or {}),
            }
        )

    fused = reciprocal_rank_fusion(
        dense_hits=dense_ranked,
        sparse_hits=sparse_ranked,
        config=deps.rrf,
        top_k=deps.top_k_fused,
    )

    debug = dict(state.get("debug") or {})
    debug["dense_hits_n"] = len(dense_ranked)
    debug["sparse_hits_n"] = len(sparse_ranked)

    return {
        "dense_hits": dense_ranked,
        "sparse_hits": sparse_ranked,
        "fused_hits": fused,
        "debug": debug,
    }


def _format_source(payload: dict[str, Any]) -> str:
    sp = payload.get("source_path")
    ci = payload.get("chunk_index")
    if sp is None:
        return "unknown"
    if ci is None:
        return str(sp)
    return f"{sp}#{ci}"


def _compact_contexts(texts: list[str], *, max_chars: int = 6000) -> list[str]:
    # Dedupe exacts; remove near-duplicates by substring.
    uniq: list[str] = []
    for t in texts:
        s = (t or "").strip()
        if not s:
            continue
        if any(s == u for u in uniq):
            continue
        if any(s in u or u in s for u in uniq):
            continue
        uniq.append(s)

    out: list[str] = []
    total = 0
    for t in uniq:
        if total + len(t) > max_chars:
            break
        out.append(t)
        total += len(t)
    return out


def node_rerank_and_answer(state: RAGState, deps: Phase2Deps) -> RAGState:
    q = state["question"]
    fused = state.get("fused_hits") or []

    texts: list[str] = []
    sources: list[str] = []
    for h in fused:
        payload = dict(h.get("payload") or {})
        texts.append(str(payload.get("text") or ""))
        sources.append(_format_source(payload))

    ranked = deps.reranker.rerank(query=q, texts=texts, top_k=min(deps.top_k_final, len(texts)))

    reranked_hits: list[dict[str, Any]] = []
    chosen_texts: list[str] = []
    chosen_sources: list[str] = []
    for idx, score in ranked:
        reranked_hits.append({"id": fused[idx]["id"], "score": score, "payload": fused[idx]["payload"]})
        chosen_texts.append(texts[idx])
        chosen_sources.append(sources[idx])

    contexts = _compact_contexts(chosen_texts)

    # Answer prompt with explicit grounding requirement.
    ctx_block = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)])
    answer_prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided contexts. "
        "If the contexts do not contain enough information, say you don't have enough information.\n\n"
        f"Question: {q}\n\n"
        f"Contexts:\n{ctx_block}\n\n"
        "Answer (be concise, and do not hallucinate):"
    )
    answer = deps.llm.generate(answer_prompt, temperature=0.2)

    return {
        "reranked_hits": reranked_hits,
        "contexts": contexts,
        "answer": answer.strip(),
        "sources": _dedupe_keep_order(chosen_sources),
    }


def node_faithfulness_check(state: RAGState, deps: Phase2Deps) -> RAGState:
    q = state["question"]
    answer = state.get("answer") or ""
    contexts = state.get("contexts") or []

    res = compute_faithfulness_ragas(question=q, answer=answer, contexts=contexts)

    debug = dict(state.get("debug") or {})
    debug["faithfulness_error"] = res.error

    return {"faithfulness_score": res.score, "debug": debug}


def should_retry(state: RAGState, deps: Phase2Deps) -> bool:
    score = state.get("faithfulness_score")
    retries = int(state.get("retries") or 0)

    if score is None:
        return False
    if score >= deps.faithfulness_threshold:
        return False
    return retries < deps.max_retries


def build_langgraph(deps: Phase2Deps):
    """Build the LangGraph pipeline. Imported lazily to keep base install light."""

    from langgraph.graph import END, StateGraph

    g = StateGraph(RAGState)

    g.add_node("classify_query", lambda s: node_classify_query(s, deps))
    g.add_node("hyde_expand", lambda s: node_hyde_expand(s, deps))
    g.add_node("hybrid_retrieve", lambda s: node_hybrid_retrieve(s, deps))
    g.add_node("rerank", lambda s: node_rerank_and_answer(s, deps))
    g.add_node("faithfulness", lambda s: node_faithfulness_check(s, deps))

    g.set_entry_point("classify_query")
    g.add_edge("classify_query", "hyde_expand")
    g.add_edge("hyde_expand", "hybrid_retrieve")
    g.add_edge("hybrid_retrieve", "rerank")
    g.add_edge("rerank", "faithfulness")

    def _route(state: RAGState):
        if should_retry(state, deps):
            return "hyde_expand"
        return END

    g.add_conditional_edges("faithfulness", _route)

    return g.compile()


def run_phase2(question: str, *, deps: Phase2Deps, debug: bool = False) -> RAGState:
    app = build_langgraph(deps)

    state: RAGState = {
        "question": question,
        "retries": 0,
        "debug": {},
    }

    while True:
        out: RAGState = app.invoke(state)

        # LangGraph returns merged state; ensure retries increment on retry.
        state.update(out)

        if should_retry(state, deps):
            state["retries"] = int(state.get("retries") or 0) + 1
            continue

        if not debug:
            state.pop("debug", None)
        return state
