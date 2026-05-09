from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from .base import BaseLoader, LoadedDocument, LoaderError


class PdfLoader(BaseLoader):
    source_type = "pdf"

    def load(self, path: Path, *, max_chars: int) -> LoadedDocument:
        try:
            doc = fitz.open(str(path))
        except Exception as exc:  # pragma: no cover
            raise LoaderError(f"Failed to open PDF: {path}") from exc

        parts: list[str] = []
        try:
            for page in doc:
                parts.append(page.get_text("text"))
                if sum(len(p) for p in parts) > max_chars:
                    break
        finally:
            doc.close()

        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars]

        return LoadedDocument(
            source_path=str(path),
            source_type=self.source_type,
            title=None,
            text=text,
        )
