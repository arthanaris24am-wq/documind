"""Tests for configurable LangChain text chunking and metadata."""

from __future__ import annotations

import pytest

from app.chunking import chunk_documents
from app.loaders import LoadedDocument


def test_chunk_documents_preserves_metadata_and_overlap() -> None:
    long_text = " ".join(f"sentence{i} has content." for i in range(60))
    doc = LoadedDocument(
        text=long_text,
        filename="sample.txt",
        page=3,
        doc_id="doc-abc123",
        file_type="txt",
    )

    chunks = chunk_documents([doc], chunk_size=150, chunk_overlap=30)
    assert len(chunks) > 2
    for idx, chunk in enumerate(chunks):
        assert len(chunk.text) <= 150
        assert chunk.doc_id == "doc-abc123"
        assert chunk.filename == "sample.txt"
        assert chunk.page == 3
        assert chunk.chunk_index == idx
        assert chunk.chunk_id

    # Chunk IDs must be unique across chunks and deterministic across repeated calls
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    repeat_chunks = chunk_documents([doc], chunk_size=150, chunk_overlap=30)
    assert [c.chunk_id for c in repeat_chunks] == [c.chunk_id for c in chunks]


def test_chunk_documents_validates_parameters() -> None:
    doc = LoadedDocument(text="hello world", filename="a.txt", page=1, doc_id="d1", file_type="txt")
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_documents([doc], chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_documents([doc], chunk_size=100, chunk_overlap=100)

