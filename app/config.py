"""Configuration loader and validation for DocuMind."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


class AppConfig(BaseModel):
    name: str = "DocuMind"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=4, gt=0)
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_multiplier: int = Field(default=3, ge=1)


class EmbeddingsConfig(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    normalize_embeddings: bool = True
    device: str = "cpu"


class QdrantConfig(BaseModel):
    mode: Literal["memory", "docker", "cloud"] = "memory"
    host: str = "localhost"
    port: int = Field(default=6333, ge=1, le=65535)
    collection_name: str = "documind_docs"
    url: str | None = None
    api_key: str | None = None

    @model_validator(mode="after")
    def validate_cloud_credentials(self) -> QdrantConfig:
        if self.mode == "cloud" and (not self.url or not self.api_key):
            raise ValueError("QDRANT_URL and QDRANT_API_KEY are required when QDRANT_MODE=cloud")
        return self


class LLMConfig(BaseModel):
    provider: str = "gemini"
    model_name: str = "gemini-2.0-flash"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, gt=0)
    max_retries: int = Field(default=4, ge=0)
    base_delay_seconds: float = Field(default=1.0, gt=0.0)
    max_delay_seconds: float = Field(default=16.0, gt=0.0)
    api_key: str | None = None


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {
        section: dict(values) if isinstance(values, dict) else {}
        for section, values in raw.items()
    }

    app_cfg = data.setdefault("app", {})
    if host := os.getenv("API_HOST"):
        app_cfg["host"] = host
    if port := os.getenv("API_PORT"):
        app_cfg["port"] = int(port)

    chunk_cfg = data.setdefault("chunking", {})
    if chunk_size := os.getenv("CHUNK_SIZE"):
        chunk_cfg["chunk_size"] = int(chunk_size)
    if chunk_overlap := os.getenv("CHUNK_OVERLAP"):
        chunk_cfg["chunk_overlap"] = int(chunk_overlap)

    ret_cfg = data.setdefault("retrieval", {})
    if top_k := os.getenv("TOP_K"):
        ret_cfg["top_k"] = int(top_k)
    if use_reranker := os.getenv("USE_RERANKER"):
        ret_cfg["use_reranker"] = _parse_bool(use_reranker)
    if reranker_model := os.getenv("RERANKER_MODEL"):
        ret_cfg["reranker_model"] = reranker_model

    emb_cfg = data.setdefault("embeddings", {})
    if emb_model := os.getenv("EMBEDDING_MODEL"):
        emb_cfg["model_name"] = emb_model

    qdrant_cfg = data.setdefault("qdrant", {})
    if mode := os.getenv("QDRANT_MODE"):
        qdrant_cfg["mode"] = mode.strip().lower()
    if q_host := os.getenv("QDRANT_HOST"):
        qdrant_cfg["host"] = q_host
    if q_port := os.getenv("QDRANT_PORT"):
        qdrant_cfg["port"] = int(q_port)
    if q_col := os.getenv("QDRANT_COLLECTION"):
        qdrant_cfg["collection_name"] = q_col
    if q_url := os.getenv("QDRANT_URL"):
        qdrant_cfg["url"] = q_url
    if q_key := os.getenv("QDRANT_API_KEY"):
        qdrant_cfg["api_key"] = q_key

    llm_cfg = data.setdefault("llm", {})
    if gemini_model := os.getenv("GEMINI_MODEL"):
        llm_cfg["model_name"] = gemini_model
    if gemini_key := os.getenv("GEMINI_API_KEY"):
        llm_cfg["api_key"] = gemini_key

    return data


def load_settings(config_path: Path | None = None, load_env_file: bool = True) -> Settings:
    """Load YAML configuration and apply environment variable overrides."""
    if load_env_file:
        load_dotenv(override=False)

    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
            if isinstance(loaded, dict):
                raw = loaded

    merged = _apply_env_overrides(raw)
    return Settings.model_validate(merged)

