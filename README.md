# ContextForge

Phase 1 delivers a runnable local pipeline:

- ingest documents (PDF/HTML/Markdown/DOCX)
- chunk (token-based)
- embed (SentenceTransformers)
- index into Qdrant
- evaluate retrieval baseline (Recall@k, MRR)

## Quickstart (Phase 1)

1) Start Qdrant

```bash
docker compose up -d
```

2) Install (editable)

```bash
python -m pip install -e .
```

3) Ingest a folder of docs

```bash
contextforge ingest ./docs
```

4) Run the sample evaluation

```bash
contextforge eval --golden ./data/golden_sample.jsonl
```

## CLI

- `contextforge ingest <path>`: ingest a file or directory
- `contextforge search "<query>"`: dense search against Qdrant
- `contextforge eval --golden <jsonl>`: compute Recall@k and MRR

## Notes

- Configure via `.env` (copy from `.env.example`).
- Default embedding model is `sentence-transformers/all-MiniLM-L6-v2` for speed.
