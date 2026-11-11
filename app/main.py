"""FastAPI application entrypoint and endpoint definitions for DocuMind."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status

from app.config import Settings, load_settings
from app.embeddings import EmbeddingClient, SentenceTransformerEmbedding
from app.llm import GeminiLLMClient, LLMClient, LLMConfigurationError, LLMError
from app.loaders import UnsupportedFileTypeError
from app.rag import RAGService
from app.reranker import CrossEncoderReranker, RerankerClient
from app.schemas import (
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IngestFileResult,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from app.vectorstore import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _build_runtime_service(settings: Settings) -> RAGService:
    """Initialize models and stores once at application startup."""
    logger.info("Loading embedding model: %s", settings.embeddings.model_name)
    embedding_client = SentenceTransformerEmbedding(
        model_name=settings.embeddings.model_name,
        device=settings.embeddings.device,
        normalize_embeddings=settings.embeddings.normalize_embeddings,
    )

    reranker_client: RerankerClient | None = None
    if settings.retrieval.use_reranker:
        logger.info("Loading cross-encoder reranker: %s", settings.retrieval.reranker_model)
        reranker_client = CrossEncoderReranker(
            model_name=settings.retrieval.reranker_model,
            device=settings.embeddings.device,
        )

    logger.info(
        "Connecting Qdrant store in mode=%s collection=%s",
        settings.qdrant.mode,
        settings.qdrant.collection_name,
    )
    vector_store = QdrantVectorStore(
        config=settings.qdrant,
        vector_size=embedding_client.dimension,
    )

    llm_client: LLMClient | None = None
    if settings.llm.api_key:
        llm_client = GeminiLLMClient(config=settings.llm)
    else:
        logger.warning("GEMINI_API_KEY not set; /query will fail when context requires LLM.")

    return RAGService(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
        llm_client=llm_client,
        reranker_client=reranker_client,
    )


def create_app(
    settings: Settings | None = None,
    embedding_client: EmbeddingClient | None = None,
    vector_store: QdrantVectorStore | None = None,
    llm_client: LLMClient | None = None,
    reranker_client: RerankerClient | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if embedding_client is not None and vector_store is not None:
            resolved_settings = settings or load_settings()
            service = RAGService(
                settings=resolved_settings,
                embedding_client=embedding_client,
                vector_store=vector_store,
                llm_client=llm_client,
                reranker_client=reranker_client,
            )
        else:
            resolved_settings = settings or load_settings()
            service = _build_runtime_service(resolved_settings)

        application.state.settings = resolved_settings
        application.state.rag_service = service
        yield

    application = FastAPI(
        title="DocuMind RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        service: RAGService = request.app.state.rag_service
        cfg: Settings = request.app.state.settings
        return HealthResponse(
            status="ok",
            app_name=cfg.app.name,
            qdrant_mode=cfg.qdrant.mode,
            collection_name=cfg.qdrant.collection_name,
            embedding_model=cfg.embeddings.model_name,
            reranker_enabled=cfg.retrieval.use_reranker and service.reranker_client is not None,
            llm_configured=service.llm_client is not None,
            total_chunks=service.vector_store.count_points(),
        )

    @application.post("/ingest", response_model=IngestResponse)
    async def ingest(
        request: Request,
        files: list[UploadFile] = File(...),
    ) -> IngestResponse:
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one file must be uploaded",
            )

        service: RAGService = request.app.state.rag_service
        results: list[IngestFileResult] = []
        total_chunks = 0

        for upload in files:
            filename = upload.filename or "unnamed"
            content = await upload.read()
            try:
                outcome = service.ingest_bytes(filename=filename, content=content)
            except UnsupportedFileTypeError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc

            results.append(
                IngestFileResult(
                    doc_id=outcome.doc_id,
                    filename=outcome.filename,
                    pages_loaded=outcome.pages_loaded,
                    chunks_indexed=outcome.chunks_indexed,
                )
            )
            total_chunks += outcome.chunks_indexed

        return IngestResponse(files=results, total_chunks_indexed=total_chunks)

    @application.post("/query", response_model=QueryResponse)
    def query(request: Request, body: QueryRequest) -> QueryResponse:
        service: RAGService = request.app.state.rag_service
        try:
            outcome = service.query(question=body.question, top_k=body.top_k)
        except LLMConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except LLMError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        return QueryResponse(
            answer=outcome.answer,
            sources=[
                SourceChunk(
                    chunk_id=src.chunk_id,
                    doc_id=src.doc_id,
                    filename=src.filename,
                    page=src.page,
                    chunk_index=src.chunk_index,
                    text=src.text,
                    score=src.score,
                )
                for src in outcome.sources
            ],
            latency_ms=outcome.latency_ms,
            model=outcome.model,
            reranked=outcome.reranked,
        )

    @application.get("/documents", response_model=DocumentListResponse)
    def list_documents(request: Request) -> DocumentListResponse:
        service: RAGService = request.app.state.rag_service
        docs = service.list_documents()
        return DocumentListResponse(
            documents=[
                DocumentInfo(
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    chunk_count=doc.chunk_count,
                )
                for doc in docs
            ],
            total_documents=len(docs),
        )

    @application.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
    def delete_document(request: Request, doc_id: str) -> DeleteDocumentResponse:
        service: RAGService = request.app.state.rag_service
        deleted = service.delete_document(doc_id)
        if deleted == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{doc_id}' not found",
            )
        return DeleteDocumentResponse(doc_id=doc_id, deleted_chunks=deleted)

    return application


app = create_app()

