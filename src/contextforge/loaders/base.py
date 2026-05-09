from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedDocument:
    source_path: str
    source_type: str
    title: str | None
    text: str


class LoaderError(RuntimeError):
    pass


class BaseLoader:
    source_type: str

    def load(self, path: Path, *, max_chars: int) -> LoadedDocument:
        raise NotImplementedError
