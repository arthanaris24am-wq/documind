"""Grid search experiment runner over chunk_size x top_k logged to MLflow."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.config import ChunkingConfig, RetrievalConfig, Settings, load_settings
from app.embeddings import EmbeddingClient, SentenceTransformerEmbedding
from app.llm import LLMClient
from app.reranker import CrossEncoderReranker, RerankerClient
from pipelines.dataset import DEFAULT_DATASET_PATH, DEFAULT_SAMPLES_DIR
from pipelines.evaluate import evaluate_configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZES = (256, 512, 1024)
DEFAULT_TOP_KS = (3, 5)
DEFAULT_BEST_CONFIG_PATH = Path("best_config.json")
DEFAULT_ARTIFACTS_DIR = Path("data/eval/experiments")


def build_grid(
    chunk_sizes: Sequence[int] = DEFAULT_CHUNK_SIZES,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
) -> list[tuple[int, int]]:
    """Return all (chunk_size, top_k) grid combinations."""
    return [(cs, k) for cs in chunk_sizes for k in top_ks]


def rank_experiment_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank experiment entries by primary score descending, then latency ascending."""

    def _sort_key(entry: dict[str, Any]) -> tuple[float, float]:
        metrics = entry.get("metrics", {})
        if "mean_ragas_score" in metrics:
            primary = float(metrics["mean_ragas_score"])
        else:
            primary = (
                float(metrics.get("retrieval_hit_rate", 0.0)) * 0.5
                + float(metrics.get("retrieval_token_recall", 0.0)) * 0.5
            )
        latency = float(metrics.get("avg_latency_ms", 0.0))
        return (primary, -latency)

    return sorted(results, key=_sort_key, reverse=True)


def format_ranked_table(ranked: Sequence[dict[str, Any]]) -> str:
    """Format ranked experiment results as an ASCII table without emojis."""
    headers = [
        "Rank",
        "chunk_size",
        "top_k",
        "mean_ragas",
        "faithfulness",
        "ans_relevancy",
        "ctx_precision",
        "ctx_recall",
        "hit_rate",
        "avg_ms",
    ]
    rows: list[list[str]] = []
    for idx, entry in enumerate(ranked, start=1):
        params = entry["params"]
        metrics = entry["metrics"]
        rows.append(
            [
                str(idx),
                str(params["chunk_size"]),
                str(params["top_k"]),
                f"{metrics.get('mean_ragas_score', 0.0):.4f}",
                f"{metrics.get('faithfulness', 0.0):.4f}",
                f"{metrics.get('answer_relevancy', 0.0):.4f}",
                f"{metrics.get('context_precision', 0.0):.4f}",
                f"{metrics.get('context_recall', 0.0):.4f}",
                f"{metrics.get('retrieval_hit_rate', 0.0):.4f}",
                f"{metrics.get('avg_latency_ms', 0.0):.2f}",
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows)) if rows else len(headers[col])
        for col in range(len(headers))
    ]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    lines = [sep, header_line, sep]
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |")
    lines.append(sep)
    return "\n".join(lines)


def run_grid_experiments(
    settings: Settings,
    chunk_sizes: Sequence[int] = DEFAULT_CHUNK_SIZES,
    top_ks: Sequence[int] = DEFAULT_TOP_KS,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    samples_dir: Path = DEFAULT_SAMPLES_DIR,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    best_config_path: Path = DEFAULT_BEST_CONFIG_PATH,
    quick: bool = False,
    retrieval_only: bool = False,
    throttle_seconds: float = 1.5,
    judge_rps: float = 0.25,
    embedding_client: EmbeddingClient | None = None,
    llm_client: LLMClient | None = None,
    reranker_client: RerankerClient | None = None,
    judge_llm: Any | None = None,
    ragas_embeddings: Any | None = None,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Run grid search over chunk_size x top_k, print ranked table, and write best_config.json."""
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

    grid = build_grid(chunk_sizes=chunk_sizes, top_ks=top_ks)
    raw_results: list[dict[str, Any]] = []
    artifacts_root = Path(artifacts_dir)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    for chunk_size, top_k in grid:
        overlap = min(settings.chunking.chunk_overlap, max(0, chunk_size - 1))
        run_settings = settings.model_copy(
            update={
                "chunking": ChunkingConfig(chunk_size=chunk_size, chunk_overlap=overlap),
                "retrieval": RetrievalConfig(
                    top_k=top_k,
                    use_reranker=settings.retrieval.use_reranker,
                    reranker_model=settings.retrieval.reranker_model,
                    rerank_candidate_multiplier=settings.retrieval.rerank_candidate_multiplier,
                ),
            }
        )
        run_output = artifacts_root / f"eval_cs{chunk_size}_k{top_k}.json"
        outcome = evaluate_configuration(
            settings=run_settings,
            dataset_path=dataset_path,
            samples_dir=samples_dir,
            output_path=run_output,
            quick=quick,
            retrieval_only=retrieval_only,
            throttle_seconds=throttle_seconds,
            judge_rps=judge_rps,
            run_name=f"grid_cs{chunk_size}_k{top_k}" + ("_quick" if quick else ""),
            embedding_client=embedder,
            llm_client=llm_client,
            reranker_client=reranker,
            judge_llm=judge_llm,
            ragas_embeddings=ragas_embeddings,
            log_mlflow=log_mlflow,
        )
        raw_results.append(
            {
                "params": outcome["params"],
                "metrics": outcome["metrics"],
                "mlflow_run_id": outcome.get("mlflow_run_id"),
                "artifact_path": str(run_output),
            }
        )

    ranked = rank_experiment_results(raw_results)
    best = ranked[0]
    best_payload = {
        "best_params": best["params"],
        "best_metrics": best["metrics"],
        "mlflow_run_id": best.get("mlflow_run_id"),
        "ranked_runs": ranked,
    }

    best_path = Path(best_config_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with best_path.open("w", encoding="utf-8") as handle:
        json.dump(best_payload, handle, indent=2)

    table = format_ranked_table(ranked)
    print("\nRanked Experiment Results:\n" + table)
    logger.info("Saved best configuration to %s", best_path)
    return best_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chunk_size x top_k RAG grid experiments.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_BEST_CONFIG_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--quick", action="store_true", help="Use first 5 questions per run.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip Gemini LLM & RAGAS judge; rank by retrieval metrics + latency.",
    )
    parser.add_argument("--throttle-seconds", type=float, default=1.5)
    parser.add_argument("--judge-rps", type=float, default=0.25)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    run_grid_experiments(
        settings=settings,
        dataset_path=args.dataset,
        samples_dir=args.samples_dir,
        artifacts_dir=args.artifacts_dir,
        best_config_path=args.output,
        quick=args.quick,
        retrieval_only=args.retrieval_only,
        throttle_seconds=args.throttle_seconds,
        judge_rps=args.judge_rps,
        log_mlflow=not args.no_mlflow,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

