from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .search import search


@dataclass(frozen=True)
class GoldenExample:
    query: str
    relevant_source_paths: set[str]


def load_golden_jsonl(path: Path) -> list[GoldenExample]:
    examples: list[GoldenExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        query = str(obj["query"]).strip()
        rel = set(str(x) for x in (obj.get("relevant_source_paths") or []))
        examples.append(GoldenExample(query=query, relevant_source_paths=rel))
    return examples


@dataclass(frozen=True)
class EvalMetrics:
    n: int
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float


def _recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = retrieved[:k]
    return 1.0 if any(r in relevant for r in topk) else 0.0


def _mrr_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    topk = retrieved[:k]
    for i, r in enumerate(topk):
        if r in relevant:
            return 1.0 / float(i + 1)
    return 0.0


def evaluate(*, golden_path: Path, settings: Settings) -> EvalMetrics:
    examples = load_golden_jsonl(golden_path)
    if not examples:
        return EvalMetrics(n=0, recall_at_5=0.0, recall_at_10=0.0, mrr_at_10=0.0)

    recall5 = 0.0
    recall10 = 0.0
    mrr10 = 0.0

    for ex in examples:
        hits = search(ex.query, settings=settings, limit=10)
        retrieved_paths = [h.source_path for h in hits if h.source_path]
        recall5 += _recall_at_k(ex.relevant_source_paths, retrieved_paths, 5)
        recall10 += _recall_at_k(ex.relevant_source_paths, retrieved_paths, 10)
        mrr10 += _mrr_at_k(ex.relevant_source_paths, retrieved_paths, 10)

    n = len(examples)
    return EvalMetrics(
        n=n,
        recall_at_5=recall5 / n,
        recall_at_10=recall10 / n,
        mrr_at_10=mrr10 / n,
    )
