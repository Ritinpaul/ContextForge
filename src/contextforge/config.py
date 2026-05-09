from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="contextforge_chunks", alias="QDRANT_COLLECTION")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    chunk_tokens: int = Field(default=512, alias="CHUNK_TOKENS")
    chunk_overlap_tokens: int = Field(default=200, alias="CHUNK_OVERLAP_TOKENS")

    ingest_max_chars: int = Field(default=2_000_000, alias="INGEST_MAX_CHARS")


def get_settings() -> Settings:
    return Settings()
