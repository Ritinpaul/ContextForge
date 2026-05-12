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

    # Phase 2: local generation + hybrid retrieval
    llm_model: str = Field(default="google/flan-t5-base", alias="LLM_MODEL")
    llm_task: str = Field(default="text2text-generation", alias="LLM_TASK")
    llm_max_new_tokens: int = Field(default=256, alias="LLM_MAX_NEW_TOKENS")
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")

    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )

    hybrid_dense_top_k: int = Field(default=50, alias="HYBRID_DENSE_TOP_K")
    hybrid_sparse_top_k: int = Field(default=50, alias="HYBRID_SPARSE_TOP_K")
    hybrid_fused_top_k: int = Field(default=20, alias="HYBRID_FUSED_TOP_K")
    final_top_k: int = Field(default=5, alias="FINAL_TOP_K")

    rrf_k: int = Field(default=60, alias="RRF_K")
    rrf_weight_dense: float = Field(default=0.7, alias="RRF_WEIGHT_DENSE")
    rrf_weight_sparse: float = Field(default=0.3, alias="RRF_WEIGHT_SPARSE")

    faithfulness_threshold: float = Field(default=0.75, alias="FAITHFULNESS_THRESHOLD")
    faithfulness_max_retries: int = Field(default=2, alias="FAITHFULNESS_MAX_RETRIES")


def get_settings() -> Settings:
    return Settings()
