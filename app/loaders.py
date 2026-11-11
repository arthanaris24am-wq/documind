"""Document loaders for PDF, TXT, and Markdown files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UnsupportedFileTypeError(ValueError):
    """Raised when an unsupported document extension is supplied."""


@dataclass(frozen=True)
class LoadedDocument:
    """Extracted text unit from a source document (one per page for PDFs)."""

    text: str
    filename: str
    page: int
    doc_id: str
    file_type: str


def compute_doc_id(filename: str, content: bytes) -> str:
    """Compute a deterministic document identifier from filename and bytes."""
    digest = hashlib.sha256()
    digest.update(filename.encode("utf-8"))
    digest.update(b":")
    digest.update(content)
    return digest.hexdigest()[:16]


def _extract_pdf_pages(content: bytes, filename: str, doc_id: str) -> list[LoadedDocument]:
    reader = PdfReader(BytesIO(content))
    documents: list[LoadedDocument] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(
                LoadedDocument(
                    text=text,
                    filename=filename,
                    page=index,
                    doc_id=doc_id,
                    file_type="pdf",
                )
            )
    return documents


def _extract_text_document(
    content: bytes,
    filename: str,
    doc_id: str,
    file_type: str,
) -> list[LoadedDocument]:
    text = content.decode("utf-8").strip()
    if not text:
        return []
    return [
        LoadedDocument(
            text=text,
            filename=filename,
            page=1,
            doc_id=doc_id,
            file_type=file_type,
        )
    ]


def load_document_from_bytes(
    filename: str,
    content: bytes,
    doc_id: str | None = None,
) -> list[LoadedDocument]:
    """Load a PDF, TXT, or Markdown document from raw bytes."""
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}' for '{safe_name}'. Supported: {supported}"
        )

    resolved_doc_id = doc_id or compute_doc_id(safe_name, content)
    if suffix == ".pdf":
        return _extract_pdf_pages(content, safe_name, resolved_doc_id)
    return _extract_text_document(content, safe_name, resolved_doc_id, suffix.lstrip("."))


def load_document_from_path(path: Path, doc_id: str | None = None) -> list[LoadedDocument]:
    """Load a PDF, TXT, or Markdown document from a filesystem path."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}' for '{file_path.name}'. Supported: {supported}"
        )

    if suffix == ".pdf":
        content = file_path.read_bytes()
    else:
        content = file_path.read_text(encoding="utf-8").encode("utf-8")

    return load_document_from_bytes(file_path.name, content, doc_id=doc_id)

