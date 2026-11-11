"""RAGAS + MLflow evaluation pipeline with Gemini judge throttling and retry."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import ChunkingConfig, QdrantConfig, RetrievalConfig, Settings, load_settings
from app.embeddings import EmbeddingClient, SentenceTransformerEmbedding
from app.llm import GeminiLLMClient, LLMClient, LLMConfigurationError
from app.rag import RAGService
from app.reranker import CrossEncoderReranker, RerankerClient
from app.vectorstore import QdrantVectorStore
from pipelines.dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_SAMPLES_DIR,
    QAPair,
    list_corpus_files,
    load_qa_dataset,
)

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = Path("data/eval/eval_results.json")


@dataclass(frozen=True)
class EvalSampleRecord:
    """Single evaluated question output from the RAG pipeline."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    latency_ms: float
    sources: list[dict[str, Any]]


def is_rate_limit_error(exc: Exception) -> bool:
    """Detect HTTP 429 or resource exhaustion errors from Gemini or RAGAS."""
    msg = str(exc).lower()
    markers = ("429", "resource_exhausted", "rate limit", "quota", "too many requests")
    return any(marker in msg for marker in markers)


class ThrottledLLMClient(LLMClient):
    """Wraps an LLMClient to enforce a minimum interval between calls."""

    def __init__(
        self,
        inner: LLMClient,
        min_interval_seconds: float = 1.5,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._min_interval = max(0.0, min_interval_seconds)
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._last_call_ts: float | None = None

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def generate(self, prompt: str) -> str:
        if self._last_call_ts is not None and self._min_interval > 0.0:
            elapsed = self._clock() - self._last_call_ts
            remaining = self._min_interval - elapsed
            if remaining > 0.0:
                self._sleep(remaining)
        result = self._inner.generate(prompt)
        self._last_call_ts = self._clock()
        return result


def build_rag_service_for_eval(
    settings: Settings,
    embedding_client: EmbeddingClient,
    llm_client: LLMClient | None,
    reranker_client: RerankerClient | None = None,
    collection_name: str = "documind_eval",
) -> RAGService:
    """Construct an isolated in-memory RAGService for evaluation."""
    eval_qdrant = QdrantConfig(mode="memory", collection_name=collection_name)
    vector_store = QdrantVectorStore(config=eval_qdrant, vector_size=embedding_client.dimension)
    return RAGService(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
        llm_client=llm_client,
        reranker_client=reranker_client,
    )


def index_corpus(rag_service: RAGService, samples_dir: Path = DEFAULT_SAMPLES_DIR) -> int:
    """Ingest all supported documents from samples_dir into rag_service."""
    total_chunks = 0
    for path in list_corpus_files(samples_dir):
        outcome = rag_service.ingest_bytes(filename=path.name, content=path.read_bytes())
        total_chunks += outcome.chunks_indexed
    return total_chunks


def run_rag_on_dataset(
    rag_service: RAGService,
    qa_pairs: Sequence[QAPair],
    top_k: int | None = None,
    retrieval_only: bool = False,
    throttle_seconds: float = 0.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[list[EvalSampleRecord], float]:
    """Run all Q&A pairs through retrieval (and generation unless retrieval_only)."""
    records: list[EvalSampleRecord] = []
    latencies: list[float] = []

    for idx, pair in enumerate(qa_pairs):
        if idx > 0 and throttle_seconds > 0.0:
            sleep_fn(throttle_seconds)

        if retrieval_only:
            start = time.perf_counter()
            chunks, _ = rag_service.retrieve(pair.question, top_k=top_k)
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            answer = " ".join(c.text for c in chunks[:1])
            sources_list = chunks
        else:
            outcome = rag_service.query(pair.question, top_k=top_k)
            elapsed_ms = outcome.latency_ms
            answer = outcome.answer
            sources_list = outcome.sources

        latencies.append(elapsed_ms)
        records.append(
            EvalSampleRecord(
                question=pair.question,
                answer=answer,
                contexts=[src.text for src in sources_list],
                ground_truth=pair.ground_truth,
                latency_ms=elapsed_ms,
                sources=[
                    {
                        "chunk_id": src.chunk_id,
                        "filename": src.filename,
                        "page": src.page,
                        "score": src.score,
                    }
                    for src in sources_list
                ],
            )
        )

    avg_latency_ms = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    return records, avg_latency_ms


def compute_retrieval_proxy_metrics(records: Sequence[EvalSampleRecord]) -> dict[str, float]:
    """Compute non-LLM retrieval quality metrics (token recall & context hit rate)."""
    if not records:
        return {"retrieval_hit_rate": 0.0, "retrieval_token_recall": 0.0}

    hits = 0
    recalls: list[float] = []
    for rec in records:
        gt_tokens = {
            tok.strip(".,!?:;()[]{}\"'").lower()
            for tok in rec.ground_truth.split()
            if len(tok.strip(".,!?:;()[]{}\"'")) > 3
        }
        ctx_text = " ".join(rec.contexts).lower()
        if not gt_tokens:
            recalls.append(1.0)
            hits += 1
            continue
        matched = sum(1 for tok in gt_tokens if tok in ctx_text)
        ratio = matched / len(gt_tokens)
        recalls.append(ratio)
        if ratio >= 0.4:
            hits += 1

    return {
        "retrieval_hit_rate": round(hits / len(records), 4),
        "retrieval_token_recall": round(sum(recalls) / len(recalls), 4),
    }


def _make_local_ragas_embeddings(embedding_client: EmbeddingClient) -> Any:
    """Wrap DocuMind's local EmbeddingClient for RAGAS without reloading the model."""
    from ragas.embeddings import BaseRagasEmbeddings

    class _LocalRagasEmbeddings(BaseRagasEmbeddings):
        def embed_query(self, text: str) -> list[float]:
            return embedding_client.embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embedding_client.embed_documents(texts)

        async def aembed_query(self, text: str) -> list[float]:
            return embedding_client.embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return embedding_client.embed_documents(texts)

    return _LocalRagasEmbeddings()


def _build_ragas_gemini_judge(
    api_key: str,
    model_name: str = "gemini-2.0-flash",
    requests_per_second: float = 0.25,
    max_retries: int = 6,
) -> Any:
    """Create a rate-limited Gemini judge LLM wrapped for RAGAS."""
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.llms import LangchainLLMWrapper

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )
    chat_model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0,
        retries=max_retries,
        rate_limiter=rate_limiter,
    )
    return LangchainLLMWrapper(chat_model)


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if parsed != parsed:  # NaN check
        return 0.0
    return round(parsed, 4)


def compute_ragas_scores(
    records: Sequence[EvalSampleRecord],
    judge_llm: Any,
    ragas_embeddings: Any,
    max_workers: int = 1,
    max_retries: int = 6,
    base_delay_seconds: float = 2.0,
    max_delay_seconds: float = 32.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, float]:
    """Compute RAGAS faithfulness, answer_relevancy, context_precision, and context_recall."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.run_config import RunConfig

    samples = [
        SingleTurnSample(
            user_input=rec.question,
            response=rec.answer,
            retrieved_contexts=rec.contexts,
            reference=rec.ground_truth,
        )
        for rec in records
    ]
    dataset = EvaluationDataset(samples=samples)
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]
    run_config = RunConfig(
        timeout=180,
        max_retries=8,
        max_wait=60,
        max_workers=max_workers,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=judge_llm,
                embeddings=ragas_embeddings,
                run_config=run_config,
                raise_exceptions=True,
                show_progress=False,
            )
            df = result.to_pandas()
            metric_keys = [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]
            scores: dict[str, float] = {}
            for key in metric_keys:
                if key in df.columns:
                    scores[key] = _safe_float(df[key].mean())
                else:
                    scores[key] = 0.0
            scores["mean_ragas_score"] = round(
                sum(scores[k] for k in metric_keys) / len(metric_keys), 4
            )
            return scores
        except Exception as exc:
            last_error = exc
            if not is_rate_limit_error(exc) or attempt >= max_retries:
                raise
            delay = min(base_delay_seconds * (2**attempt), max_delay_seconds)
            jitter = random.uniform(0.0, delay * 0.25)
            wait_s = delay + jitter
            logger.warning(
                "RAGAS hit rate limit on attempt %d/%d (%s). Retrying in %.2fs.",
                attempt + 1,
                max_retries + 1,
                exc,
                wait_s,
            )
            sleep_fn(wait_s)

    raise RuntimeError(f"RAGAS evaluation failed after retries: {last_error}")


def log_evaluation_to_mlflow(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: Sequence[Path],
) -> str:
    """Log evaluation parameters, metrics, and artifacts to MLflow."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for path in artifact_paths:
            artifact_path = Path(path)
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
        return str(run.info.run_id)


def evaluate_configuration(
    settings: Settings,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    samples_dir: Path = DEFAULT_SAMPLES_DIR,
    output_path: Path = DEFAULT_RESULTS_PATH,
    quick: bool = False,
    retrieval_only: bool = False,
    throttle_seconds: float = 1.5,
    judge_rps: float = 0.25,
    run_name: str | None = None,
    embedding_client: EmbeddingClient | None = None,
    llm_client: LLMClient | None = None,
    reranker_client: RerankerClient | None = None,
    judge_llm: Any | None = None,
    ragas_embeddings: Any | None = None,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Run full evaluation pipeline for a configuration and save/log results."""
    limit = 5 if quick else None
    qa_pairs = load_qa_dataset(dataset_path, limit=limit)

    embedder = embedding_client or SentenceTransformerEmbedding(
        model_name=settings.embeddings.model_name,
        device=settings.embeddings.device,
        normalize_embeddings=settings.embeddings.normalize_embeddings,
    )

    reranker = reranker_client
    if reranker is None and settings.retrieval.use_reranker:
        reranker = CrossEncoderReranker(
            model_name=settings.retrieval.reranker_model,
            device=settings.embeddings.device,
        )

    active_llm: LLMClient | None = None
    if not retrieval_only:
        if llm_client is not None:
            active_llm = ThrottledLLMClient(llm_client, min_interval_seconds=throttle_seconds)
        else:
            if not settings.llm.api_key:
                raise LLMConfigurationError(
                    "GEMINI_API_KEY is required for RAG + RAGAS evaluation. "
                    "Set it in .env or use --retrieval-only for offline retrieval checks."
                )
            raw_gemini = GeminiLLMClient(config=settings.llm)
            active_llm = ThrottledLLMClient(raw_gemini, min_interval_seconds=throttle_seconds)

    rag_service = build_rag_service_for_eval(
        settings=settings,
        embedding_client=embedder,
        llm_client=active_llm,
        reranker_client=reranker,
        collection_name=f"eval_cs{settings.chunking.chunk_size}_k{settings.retrieval.top_k}",
    )
    indexed_chunks = index_corpus(rag_service, samples_dir=samples_dir)
    logger.info(
        "Indexed %d chunks (chunk_size=%d, overlap=%d, top_k=%d, reranker=%s)",
        indexed_chunks,
        settings.chunking.chunk_size,
        settings.chunking.chunk_overlap,
        settings.retrieval.top_k,
        settings.retrieval.use_reranker,
    )

    records, avg_latency_ms = run_rag_on_dataset(
        rag_service=rag_service,
        qa_pairs=qa_pairs,
        top_k=settings.retrieval.top_k,
        retrieval_only=retrieval_only,
    )

    proxy_metrics = compute_retrieval_proxy_metrics(records)
    if retrieval_only:
        scores: dict[str, float] = {**proxy_metrics}
    else:
        active_judge = judge_llm
        if active_judge is None:
            if not settings.llm.api_key:
                raise LLMConfigurationError("GEMINI_API_KEY is required for Gemini RAGAS judge.")
            active_judge = _build_ragas_gemini_judge(
                api_key=settings.llm.api_key,
                model_name=settings.llm.model_name,
                requests_per_second=judge_rps,
            )
        active_ragas_emb = ragas_embeddings or _make_local_ragas_embeddings(embedder)
        ragas_scores = compute_ragas_scores(
            records=records,
            judge_llm=active_judge,
            ragas_embeddings=active_ragas_emb,
        )
        scores = {**ragas_scores, **proxy_metrics}

    metrics: dict[str, float] = {
        **scores,
        "avg_latency_ms": avg_latency_ms,
        "indexed_chunks": float(indexed_chunks),
        "num_questions": float(len(qa_pairs)),
    }

    params: dict[str, Any] = {
        "chunk_size": settings.chunking.chunk_size,
        "chunk_overlap": settings.chunking.chunk_overlap,
        "top_k": settings.retrieval.top_k,
        "embedding_model": settings.embeddings.model_name,
        "use_reranker": settings.retrieval.use_reranker,
        "llm_model": settings.llm.model_name,
        "quick": quick,
        "retrieval_only": retrieval_only,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": params,
        "metrics": metrics,
        "records": [asdict(rec) for rec in records],
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    mlflow_run_id: str | None = None
    if log_mlflow:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "documind-rag")
        resolved_run_name = run_name or (
            f"eval_cs{settings.chunking.chunk_size}_k{settings.retrieval.top_k}"
            + ("_quick" if quick else "")
        )
        mlflow_run_id = log_evaluation_to_mlflow(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=resolved_run_name,
            params=params,
            metrics=metrics,
            artifact_paths=[out_path],
        )

    payload["mlflow_run_id"] = mlflow_run_id
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DocuMind RAG pipeline with RAGAS.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--quick", action="store_true", help="Evaluate only the first 5 questions.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip Gemini LLM & RAGAS judge; evaluate retrieval + latency only.",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--use-reranker", action="store_true", default=None)
    parser.add_argument("--throttle-seconds", type=float, default=1.5)
    parser.add_argument("--judge-rps", type=float, default=0.25)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_settings = load_settings()

    chunk_size = args.chunk_size or base_settings.chunking.chunk_size
    chunk_overlap = (
        args.chunk_overlap
        if args.chunk_overlap is not None
        else min(base_settings.chunking.chunk_overlap, max(0, chunk_size - 1))
    )
    top_k = args.top_k or base_settings.retrieval.top_k
    use_reranker = (
        args.use_reranker
        if args.use_reranker is not None
        else base_settings.retrieval.use_reranker
    )

    updated_settings = base_settings.model_copy(
        update={
            "chunking": ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            "retrieval": RetrievalConfig(
                top_k=top_k,
                use_reranker=use_reranker,
                reranker_model=base_settings.retrieval.reranker_model,
                rerank_candidate_multiplier=base_settings.retrieval.rerank_candidate_multiplier,
            ),
        }
    )

    result = evaluate_configuration(
        settings=updated_settings,
        dataset_path=args.dataset,
        samples_dir=args.samples_dir,
        output_path=args.output,
        quick=args.quick,
        retrieval_only=args.retrieval_only,
        throttle_seconds=args.throttle_seconds,
        judge_rps=args.judge_rps,
        log_mlflow=not args.no_mlflow,
    )
    logger.info("Evaluation complete. Metrics: %s", json.dumps(result["metrics"]))
    logger.info("Results saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
