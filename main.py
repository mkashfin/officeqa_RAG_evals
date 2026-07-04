#!/usr/bin/env python3
"""Treasury RAG system entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treasury Financial Question Answering RAG System",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download dataset artifacts.")
    download_parser.add_argument("--force", action="store_true")

    index_parser = subparsers.add_parser("index", help="Build vector indexes.")
    index_parser.add_argument(
        "--system",
        choices=["baseline", "engineered", "both"],
        default="both",
    )
    index_parser.add_argument("--reset", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation metrics.")
    eval_parser.add_argument(
        "--system",
        choices=["baseline", "engineered", "both"],
        default="both",
    )
    eval_parser.add_argument("--compare", action="store_true")

    ask_parser = subparsers.add_parser("ask", help="Ask a question.")
    ask_parser.add_argument("question")
    ask_parser.add_argument(
        "--system",
        choices=["baseline", "engineered"],
        default="engineered",
    )
    ask_parser.add_argument("--year", type=int)
    ask_parser.add_argument("--month")

    args = parser.parse_args()
    settings = load_settings()
    settings.ensure_directories()

    if args.command == "download":
        from src.ingestion.downloader import DatasetDownloadError, prepare_data

        try:
            summary = prepare_data(settings, force=args.force)
        except DatasetDownloadError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        print(summary)
        return

    if args.command == "index":
        from src.pipelines.baseline import BaselinePipeline
        from src.pipelines.engineered import EngineeredPipeline

        if args.system in ("baseline", "both"):
            print(f"Baseline chunks: {BaselinePipeline(settings).build_index(reset=args.reset)}")
        if args.system in ("engineered", "both"):
            print(f"Engineered chunks: {EngineeredPipeline(settings).build_index(reset=args.reset)}")
        return

    if args.command == "evaluate":
        from src.evaluation.evaluator import compare_results, evaluate_pipeline, save_evaluation_result
        from src.pipelines.baseline import BaselinePipeline
        from src.pipelines.engineered import EngineeredPipeline

        baseline_result = engineered_result = None
        if args.system in ("baseline", "both"):
            baseline_result = evaluate_pipeline(
                BaselinePipeline(settings), settings, "baseline", use_metadata_filters=False
            )
            save_evaluation_result(baseline_result, settings.data.results_dir / "baseline_eval.json")

        if args.system in ("engineered", "both"):
            engineered_result = evaluate_pipeline(
                EngineeredPipeline(settings), settings, "engineered", use_metadata_filters=True
            )
            save_evaluation_result(
                engineered_result, settings.data.results_dir / "engineered_eval.json"
            )

        if args.compare and baseline_result and engineered_result:
            print(compare_results(baseline_result, engineered_result).to_string(index=False))
        return

    if args.command == "ask":
        from src.pipelines.baseline import BaselinePipeline
        from src.pipelines.engineered import EngineeredPipeline

        pipeline = (
            BaselinePipeline(settings)
            if args.system == "baseline"
            else EngineeredPipeline(settings)
        )
        pipeline.build_index(reset=False)
        response = pipeline.ask(args.question, year=args.year, month=args.month)
        print(response.answer)


if __name__ == "__main__":
    main()
