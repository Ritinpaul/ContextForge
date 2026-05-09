from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from .base import BaseLoader, LoadedDocument


class HtmlLoader(BaseLoader):
    source_type = "html"

    def load(self, path: Path, *, max_chars: int) -> LoadedDocument:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "lxml")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else None
        text = soup.get_text("\n", strip=True)
        if len(text) > max_chars:
            text = text[:max_chars]

        return LoadedDocument(
            source_path=str(path), source_type=self.source_type, title=title, text=text
        )
