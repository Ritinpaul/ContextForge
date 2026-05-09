from __future__ import annotations

from dataclasses import dataclass

import tiktoken


@dataclass(frozen=True)
class Chunk:
    text: str
    start_token: int
    end_token: int
    index: int


def _get_encoder() -> tiktoken.Encoding:
    # cl100k_base is a good general-purpose tokenizer.
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be < chunk_tokens")

    encoder = _get_encoder()
    tokens = encoder.encode(text)
    if not tokens:
        return []

    chunks: list[Chunk] = []
    start = 0
    i = 0
    step = chunk_tokens - overlap_tokens
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk_text_str = encoder.decode(tokens[start:end]).strip()
        if chunk_text_str:
            chunks.append(Chunk(text=chunk_text_str, start_token=start, end_token=end, index=i))
            i += 1
        if end == len(tokens):
            break
        start = start + step

    return chunks
