"""Tests for PDF, TXT, and Markdown document loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.loaders import (
    UnsupportedFileTypeError,
    load_document_from_bytes,
    load_document_from_path,
)
from tests.conftest import build_minimal_pdf_bytes


def test_load_txt_and_md_from_path(tmp_path: Path) -> None:
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("DocuMind uses Qdrant for vector search.", encoding="utf-8")
    md_file = tmp_path / "guide.md"
    md_file.write_text("# Guide\nFastAPI serves the REST endpoints.", encoding="utf-8")

    txt_docs = load_document_from_path(txt_file)
    md_docs = load_document_from_path(md_file)

    assert len(txt_docs) == 1
    assert txt_docs[0].filename == "notes.txt"
    assert txt_docs[0].page == 1
    assert txt_docs[0].file_type == "txt"
    assert "Qdrant" in txt_docs[0].text

    assert len(md_docs) == 1
    assert md_docs[0].filename == "guide.md"
    assert md_docs[0].file_type == "md"
    assert "FastAPI" in md_docs[0].text


def test_load_multipage_pdf_bytes() -> None:
    pdf_bytes = build_minimal_pdf_bytes(
        [
            "Page one discusses BAAI bge-small-en-v1.5 embeddings.",
            "Page two covers Hugging Face Spaces free deployment.",
        ]
    )
    docs = load_document_from_bytes("arch.pdf", pdf_bytes)
    assert len(docs) == 2
    assert docs[0].filename == "arch.pdf"
    assert docs[0].page == 1
    assert docs[0].file_type == "pdf"
    assert "embeddings" in docs[0].text
    assert docs[1].page == 2
    assert "Hugging Face Spaces" in docs[1].text
    assert docs[0].doc_id == docs[1].doc_id


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        load_document_from_path(bad_file)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document_from_path(tmp_path / "does_not_exist.txt")

