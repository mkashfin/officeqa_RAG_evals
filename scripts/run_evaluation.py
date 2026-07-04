#!/usr/bin/env python3
"""Run evaluation for baseline and/or engineered systems."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.evaluation.evaluator import (
    compare_results,
    evaluate_pipeline,
    save_evaluation_result,
)
from src.pipelines.baseline import BaselinePipeline
from src.pipelines.engineered import EngineeredPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Treasury RAG systems.")
    parser.add_argument(
        "--system",
        choices=["baseline", "engineered", "both"],
        default="both",
        help="Which system(s) to evaluate.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print comparison table when both systems are evaluated.",
    )
    args = parser.parse_args()

    settings = load_settings()
    settings.ensure_directories()
    results_dir = settings.data.results_dir

    baseline_result = None
    engineered_result = None

    if args.system in ("baseline", "both"):
        pipeline = BaselinePipeline(settings)
        pipeline.build_index(reset=False)
        baseline_result = evaluate_pipeline(
            pipeline, settings, "baseline", use_metadata_filters=False
        )
        save_evaluation_result(baseline_result, results_dir / "baseline_eval.json")
        print("\nBaseline evaluation:")
        print(f"  Hit Rate@5: {baseline_result.retrieval.hit_rate_at_k:.4f}")
        print(f"  MRR: {baseline_result.retrieval.mrr:.4f}")
        print(f"  Recall: {baseline_result.retrieval.recall:.4f}")
        print(f"  Groundedness: {baseline_result.generation.groundedness:.4f}")
        print(f"  Factual Accuracy: {baseline_result.generation.factual_accuracy:.4f}")
        print(f"  Hallucination Rate: {baseline_result.generation.hallucination_rate:.4f}")

    if args.system in ("engineered", "both"):
        pipeline = EngineeredPipeline(settings)
        pipeline.build_index(reset=False)
        engineered_result = evaluate_pipeline(
            pipeline, settings, "engineered", use_metadata_filters=True
        )
        save_evaluation_result(engineered_result, results_dir / "engineered_eval.json")
        print("\nEngineered evaluation:")
        print(f"  Hit Rate@5: {engineered_result.retrieval.hit_rate_at_k:.4f}")
        print(f"  MRR: {engineered_result.retrieval.mrr:.4f}")
        print(f"  Recall: {engineered_result.retrieval.recall:.4f}")
        print(f"  Groundedness: {engineered_result.generation.groundedness:.4f}")
        print(f"  Factual Accuracy: {engineered_result.generation.factual_accuracy:.4f}")
        print(f"  Hallucination Rate: {engineered_result.generation.hallucination_rate:.4f}")

    if args.compare and baseline_result and engineered_result:
        table = compare_results(baseline_result, engineered_result)
        comparison_path = results_dir / "comparison.csv"
        table.to_csv(comparison_path, index=False)
        print("\nComparison table:")
        print(table.to_string(index=False))
        print(f"\nSaved comparison to {comparison_path}")


if __name__ == "__main__":
    main()
