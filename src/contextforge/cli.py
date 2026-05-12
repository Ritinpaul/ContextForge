from __future__ import annotations

import argparse
from pathlib import Path

from .ask import ask as ask_question
from .config import Settings
from .eval import evaluate
from .ingest import ingest_path
from .search import search


def _cmd_ingest(args: argparse.Namespace) -> int:
    settings = Settings()
    res = ingest_path(Path(args.path), settings=settings)
    print(f"Ingested files={res.files} chunks={res.chunks} into {settings.qdrant_collection}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    settings = Settings()
    hits = search(args.query, settings=settings, limit=args.limit)
    for h in hits:
        snippet = (h.text or "").replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        print(f"{h.score:.4f}\t{h.source_path}\t#{h.chunk_index}\t{snippet}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    settings = Settings()
    metrics = evaluate(golden_path=Path(args.golden), settings=settings)
    print(
        "\n".join(
            [
                f"n={metrics.n}",
                f"recall@5={metrics.recall_at_5:.3f}",
                f"recall@10={metrics.recall_at_10:.3f}",
                f"mrr@10={metrics.mrr_at_10:.3f}",
            ]
        )
    )
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    settings = Settings()
    res = ask_question(args.question, settings=settings, debug=bool(args.debug))

    print(res.answer)

    if res.sources:
        print("\nSources:")
        for s in res.sources:
            print(f"- {s}")

    if args.debug:
        print("\nDebug:")
        print(f"- faithfulness={res.faithfulness}")
        print(f"- retries={res.retries}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="contextforge")
    sub = p.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest a folder/file into Qdrant")
    ingest_p.add_argument("path", help="Path to folder or file")
    ingest_p.set_defaults(func=_cmd_ingest)

    search_p = sub.add_parser("search", help="Search Qdrant")
    search_p.add_argument("query", help="Query text")
    search_p.add_argument(
        "--limit",
        "--top-k",
        dest="limit",
        type=int,
        default=10,
        metavar="N",
        help="Number of results to return (alias: --top-k)",
    )
    search_p.set_defaults(func=_cmd_search)

    eval_p = sub.add_parser("eval", help="Evaluate retrieval on golden dataset")
    eval_p.add_argument("--golden", required=True, help="Path to golden JSONL")
    eval_p.set_defaults(func=_cmd_eval)

    ask_p = sub.add_parser(
        "ask",
        help="Answer a question using hybrid retrieval + reranking + faithfulness retry (Phase 2)",
    )
    ask_p.add_argument("question", help="Question to answer")
    ask_p.add_argument(
        "--debug",
        action="store_true",
        help="Print extra debug info (faithfulness score, retries)",
    )
    ask_p.set_defaults(func=_cmd_ask)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
