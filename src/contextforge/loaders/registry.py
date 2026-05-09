from __future__ import annotations

from pathlib import Path

from .base import BaseLoader, LoadedDocument, LoaderError
from .docx_loader import DocxLoader
from .html_loader import HtmlLoader
from .markdown_loader import MarkdownLoader
from .pdf_loader import PdfLoader


def _suffix(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


LOADERS_BY_EXT: dict[str, BaseLoader] = {
    "md": MarkdownLoader(),
    "markdown": MarkdownLoader(),
    "html": HtmlLoader(),
    "htm": HtmlLoader(),
    "pdf": PdfLoader(),
    "docx": DocxLoader(),
}


def load_document(path: Path, *, max_chars: int) -> LoadedDocument:
    ext = _suffix(path)
    loader = LOADERS_BY_EXT.get(ext)
    if loader is None:
        raise LoaderError(f"Unsupported file type: .{ext} ({path})")
    return loader.load(path, max_chars=max_chars)
