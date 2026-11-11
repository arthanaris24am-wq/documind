"""Tests for YAML configuration loading and environment overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import ChunkingConfig, QdrantConfig, load_settings


def test_load_default_config_yaml() -> None:
    settings = load_settings(load_env_file=False)
    assert settings.app.name == "DocuMind"
    assert settings.chunking.chunk_size > 0
    assert settings.chunking.chunk_overlap < settings.chunking.chunk_size
    assert settings.embeddings.model_name == "BAAI/bge-small-en-v1.5"
    assert settings.qdrant.mode in {"memory", "docker", "cloud"}


def test_env_overrides_yaml_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "chunking:\n  chunk_size: 400\n  chunk_overlap: 40\nqdrant:\n  mode: memory\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHUNK_OVERLAP", "80")
    monkeypatch.setenv("TOP_K", "6")
    monkeypatch.setenv("USE_RERANKER", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")

    settings = load_settings(config_path=cfg_file, load_env_file=False)
    assert settings.chunking.chunk_size == 800
    assert settings.chunking.chunk_overlap == 80
    assert settings.retrieval.top_k == 6
    assert settings.retrieval.use_reranker is True
    assert settings.llm.api_key == "test-key-123"


def test_qdrant_cloud_requires_url_and_key() -> None:
    with pytest.raises(ValueError, match="QDRANT_URL and QDRANT_API_KEY"):
        QdrantConfig(mode="cloud", url=None, api_key=None)

    cfg = QdrantConfig(
        mode="cloud",
        url="https://example.qdrant.io",
        api_key="secret-token",
    )
    assert cfg.mode == "cloud"


def test_chunk_overlap_must_be_less_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)

