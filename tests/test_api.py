"""Integration tests for FastAPI endpoints with Qdrant memory mode and mocked LLM."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import QdrantConfig, Settings
from app.main import create_app
from app.prompts import INSUFFICIENT_CONTEXT_RESPONSE
from app.vectorstore import QdrantVectorStore
from tests.conftest import DeterministicEmbeddingClient, FakeLLMClient, build_minimal_pdf_bytes


def test_health_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["qdrant_mode"] == "memory"
    assert payload["llm_configured"] is True
    assert payload["total_chunks"] == 0


def test_query_returns_insufficient_context_when_index_empty(api_client: TestClient) -> None:
    response = api_client.post("/query", json={"question": "What is DocuMind?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == INSUFFICIENT_CONTEXT_RESPONSE
    assert payload["sources"] == []
    assert payload["latency_ms"] >= 0.0


def test_ingest_query_list_and_delete_lifecycle(
    api_client: TestClient,
    fake_llm: FakeLLMClient,
) -> None:
    pdf_bytes = build_minimal_pdf_bytes(["DocuMind deploys to Hugging Face Spaces on port 7860."])
    ingest_res = api_client.post(
        "/ingest",
        files=[
            ("files", ("guide.pdf", pdf_bytes, "application/pdf")),
            ("files", ("notes.md", b"# Stack\nFastAPI and Streamlit.", "text/markdown")),
        ],
    )
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert len(ingest_data["files"]) == 2
    assert ingest_data["total_chunks_indexed"] >= 2

    docs_res = api_client.get("/documents")
    assert docs_res.status_code == 200
    docs_data = docs_res.json()
    assert docs_data["total_documents"] == 2

    query_res = api_client.post(
        "/query",
        json={"question": "What port does Hugging Face Spaces use?", "top_k": 2},
    )
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert query_data["answer"].startswith("Mocked grounded answer")
    assert len(query_data["sources"]) == 2
    assert query_data["sources"][0]["filename"] in {"guide.pdf", "notes.md"}
    assert "score" in query_data["sources"][0]
    assert query_data["latency_ms"] >= 0.0
    assert len(fake_llm.prompts) == 1

    target_doc_id = docs_data["documents"][0]["doc_id"]
    del_res = api_client.delete(f"/documents/{target_doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted_chunks"] >= 1

    missing_del = api_client.delete(f"/documents/{target_doc_id}")
    assert missing_del.status_code == 404


def test_ingest_unsupported_file_returns_400(api_client: TestClient) -> None:
    response = api_client.post(
        "/ingest",
        files=[("files", ("bad.exe", b"binary", "application/octet-stream"))],
    )
    assert response.status_code == 400


def test_query_returns_503_when_llm_not_configured() -> None:
    settings = Settings(qdrant=QdrantConfig(mode="memory", collection_name="no_llm_test"))
    embedder = DeterministicEmbeddingClient(dim=16)
    store = QdrantVectorStore(config=settings.qdrant, vector_size=embedder.dimension)
    app = create_app(
        settings=settings,
        embedding_client=embedder,
        vector_store=store,
        llm_client=None,
    )
    with TestClient(app) as client:
        client.post(
            "/ingest",
            files=[("files", ("doc.txt", b"Some relevant document content.", "text/plain"))],
        )
        res = client.post("/query", json={"question": "What is in the document?"})
        assert res.status_code == 503

