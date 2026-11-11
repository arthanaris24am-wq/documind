"""Offline tests for Phase 2 evaluation dataset, throttling, MLflow, and grid runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app.config import QdrantConfig, Settings
from pipelines.dataset import DEFAULT_DATASET_PATH, list_corpus_files, load_qa_dataset
from pipelines.evaluate import (
    EvalSampleRecord,
    ThrottledLLMClient,
    build_rag_service_for_eval,
    compute_ragas_scores,
    compute_retrieval_proxy_metrics,
    index_corpus,
    is_rate_limit_error,
    log_evaluation_to_mlflow,
    run_rag_on_dataset,
)
from pipelines.run_experiments import (
    build_grid,
    format_ranked_table,
    rank_experiment_results,
    run_grid_experiments,
)
from tests.conftest import DeterministicEmbeddingClient, FakeLLMClient


def test_qa_dataset_has_20_plus_valid_pairs() -> None:
    pairs = load_qa_dataset(DEFAULT_DATASET_PATH)
    assert len(pairs) >= 20
    for pair in pairs:
        assert pair.question
        assert pair.ground_truth

    quick_pairs = load_qa_dataset(DEFAULT_DATASET_PATH, limit=5)
    assert len(quick_pairs) == 5


def test_list_corpus_files_finds_samples() -> None:
    files = list_corpus_files()
    names = {p.name for p in files}
    assert {
        "documind_architecture.md",
        "rag_best_practices.txt",
        "deployment_guide.pdf",
    }.issubset(names)


def test_throttled_llm_client_enforces_interval() -> None:
    now = [100.0]
    sleeps: list[float] = []

    def fake_clock() -> float:
        return now[0]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    inner = FakeLLMClient()
    throttled = ThrottledLLMClient(
        inner=inner,
        min_interval_seconds=2.0,
        sleep_fn=fake_sleep,
        clock_fn=fake_clock,
    )

    throttled.generate("q1")
    now[0] += 0.5
    throttled.generate("q2")

    assert len(inner.prompts) == 2
    assert sleeps == [pytest.approx(1.5)]


def test_is_rate_limit_error_detection() -> None:
    assert is_rate_limit_error(RuntimeError("429 RESOURCE_EXHAUSTED")) is True
    assert is_rate_limit_error(RuntimeError("Too Many Requests quota exceeded")) is True
    assert is_rate_limit_error(ValueError("Invalid JSON schema")) is False


def test_run_rag_on_dataset_and_proxy_metrics() -> None:
    settings = Settings(qdrant=QdrantConfig(mode="memory", collection_name="eval_unit"))
    embedder = DeterministicEmbeddingClient(dim=32)
    service = build_rag_service_for_eval(
        settings=settings,
        embedding_client=embedder,
        llm_client=FakeLLMClient(),
        collection_name="eval_unit",
    )
    indexed = index_corpus(service)
    assert indexed >= 3

    pairs = load_qa_dataset(DEFAULT_DATASET_PATH, limit=5)
    records, avg_latency = run_rag_on_dataset(
        rag_service=service,
        qa_pairs=pairs,
        top_k=3,
        retrieval_only=False,
    )
    assert len(records) == 5
    assert avg_latency >= 0.0
    assert all(rec.contexts for rec in records)

    metrics = compute_retrieval_proxy_metrics(records)
    assert 0.0 <= metrics["retrieval_hit_rate"] <= 1.0
    assert 0.0 <= metrics["retrieval_token_recall"] <= 1.0


def test_compute_retrieval_proxy_metrics_empty() -> None:
    assert compute_retrieval_proxy_metrics([]) == {
        "retrieval_hit_rate": 0.0,
        "retrieval_token_recall": 0.0,
    }
    rec = EvalSampleRecord(
        question="What port?",
        answer="7860",
        contexts=["Hugging Face Spaces uses port 7860 for Docker SDK."],
        ground_truth="Hugging Face Spaces uses port 7860.",
        latency_ms=10.0,
        sources=[],
    )
    scores = compute_retrieval_proxy_metrics([rec])
    assert scores["retrieval_hit_rate"] == 1.0
    assert scores["retrieval_token_recall"] > 0.5


def test_grid_ranking_table_and_best_config_offline(tmp_path: Path) -> None:
    grid = build_grid()
    assert grid == [
        (256, 3),
        (256, 5),
        (512, 3),
        (512, 5),
        (1024, 3),
        (1024, 5),
    ]

    best_file = tmp_path / "best_config.json"
    artifacts_dir = tmp_path / "artifacts"
    settings = Settings(qdrant=QdrantConfig(mode="memory", collection_name="grid_unit"))

    result = run_grid_experiments(
        settings=settings,
        artifacts_dir=artifacts_dir,
        best_config_path=best_file,
        quick=True,
        retrieval_only=True,
        embedding_client=DeterministicEmbeddingClient(dim=32),
        log_mlflow=False,
    )

    assert best_file.is_file()
    saved = json.loads(best_file.read_text(encoding="utf-8"))
    assert len(saved["ranked_runs"]) == 6
    assert saved["best_params"] == result["best_params"]

    ranked = rank_experiment_results(saved["ranked_runs"])
    assert ranked[0]["params"] == saved["best_params"]

    table = format_ranked_table(ranked)
    assert "chunk_size" in table
    assert "top_k" in table


def test_log_evaluation_to_mlflow_records_params_metrics_and_artifact(tmp_path: Path) -> None:
    import mlflow

    db_path = tmp_path / "mlflow.db"
    artifact_file = tmp_path / "eval_results.json"
    artifact_file.write_text('{"status": "ok"}', encoding="utf-8")

    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    run_id = log_evaluation_to_mlflow(
        tracking_uri=tracking_uri,
        experiment_name="documind-unit-test",
        run_name="unit_eval_run",
        params={"chunk_size": 512, "top_k": 3, "use_reranker": False},
        metrics={"mean_ragas_score": 0.85, "avg_latency_ms": 12.5},
        artifact_paths=[artifact_file],
    )

    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(run_id)
    assert run.data.params["chunk_size"] == "512"
    assert run.data.params["top_k"] == "3"
    assert run.data.metrics["mean_ragas_score"] == pytest.approx(0.85)
    assert run.data.metrics["avg_latency_ms"] == pytest.approx(12.5)


def test_compute_ragas_scores_retries_on_429() -> None:
    class _FakeResult:
        def to_pandas(self) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "faithfulness": 0.9,
                        "answer_relevancy": 0.8,
                        "context_precision": 0.85,
                        "context_recall": 0.95,
                    }
                ]
            )

    calls = [0]
    sleeps: list[float] = []

    def fake_evaluate(**kwargs):
        calls[0] += 1
        if calls[0] < 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return _FakeResult()

    records = [
        EvalSampleRecord(
            question="What port?",
            answer="Port 7860.",
            contexts=["Exposes port 7860."],
            ground_truth="Port 7860.",
            latency_ms=5.0,
            sources=[],
        )
    ]

    with patch("ragas.evaluate", side_effect=fake_evaluate):
        scores = compute_ragas_scores(
            records=records,
            judge_llm=object(),
            ragas_embeddings=object(),
            max_retries=2,
            base_delay_seconds=0.5,
            sleep_fn=sleeps.append,
        )

    assert calls[0] == 2
    assert len(sleeps) == 1
    assert scores["faithfulness"] == pytest.approx(0.9)
    assert scores["answer_relevancy"] == pytest.approx(0.8)
    assert scores["context_precision"] == pytest.approx(0.85)
    assert scores["context_recall"] == pytest.approx(0.95)
    assert scores["mean_ragas_score"] == pytest.approx(0.875)

