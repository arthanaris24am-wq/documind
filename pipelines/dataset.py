"""Evaluation dataset and sample corpus loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.loaders import SUPPORTED_EXTENSIONS

DEFAULT_DATASET_PATH = Path("data/eval/qa_dataset.json")
DEFAULT_SAMPLES_DIR = Path("data/raw_samples")


@dataclass(frozen=True)
class QAPair:
    """Single question-answer pair for RAG evaluation."""

    question: str
    ground_truth: str


def load_qa_dataset(path: Path = DEFAULT_DATASET_PATH, limit: int | None = None) -> list[QAPair]:
    """Load and validate evaluation Q&A pairs from JSON."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Evaluation dataset must be a non-empty JSON list: {dataset_path}")

    pairs: list[QAPair] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} in {dataset_path} must be an object")
        question = str(item.get("question", "")).strip()
        ground_truth = str(item.get("ground_truth", "")).strip()
        if not question or not ground_truth:
            raise ValueError(
                f"Item {idx} in {dataset_path} must contain non-empty 'question' and 'ground_truth'"
            )
        pairs.append(QAPair(question=question, ground_truth=ground_truth))

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        return pairs[:limit]
    return pairs


def list_corpus_files(samples_dir: Path = DEFAULT_SAMPLES_DIR) -> list[Path]:
    """Return sorted supported document paths from a directory."""
    directory = Path(samples_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {directory}")
    files = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise ValueError(f"No supported documents found in {directory}")
    return files

