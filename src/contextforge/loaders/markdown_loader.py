from __future__ import annotations

from pathlib import Path

from .base import BaseLoader, LoadedDocument


class MarkdownLoader(BaseLoader):
    source_type = "markdown"

    def load(self, path: Path, *, max_chars: int) -> LoadedDocument:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if len(text) > max_chars:
            text = text[:max_chars]
        title = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip() or None
                break
        return LoadedDocument(
            source_path=str(path), source_type=self.source_type, title=title, text=text
        )
