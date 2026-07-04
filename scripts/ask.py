#!/usr/bin/env python3
"""Interactive question answering CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.pipelines.baseline import BaselinePipeline
from src.pipelines.engineered import EngineeredPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a Treasury financial question.")
    parser.add_argument("question", help="Question to ask.")
    parser.add_argument(
        "--system",
        choices=["baseline", "engineered"],
        default="engineered",
    )
    parser.add_argument("--year", type=int, help="Filter retrieval by year.")
    parser.add_argument("--month", help="Filter retrieval by month.")
    args = parser.parse_args()

    settings = load_settings()
    pipeline = (
        BaselinePipeline(settings)
        if args.system == "baseline"
        else EngineeredPipeline(settings)
    )
    pipeline.build_index(reset=False)

    if pipeline.vector_store.count == 0:
        print(
            "\nERROR: Vector index is empty. Build it first:\n"
            "  python scripts/build_index.py --system both --reset\n"
        )
        sys.exit(1)

    response = pipeline.ask(
        question=args.question,
        year=args.year,
        month=args.month,
    )

    print("\nAnswer:")
    print(response.answer)
    print("\nSources:")
    if not response.retrieved_chunks:
        print("  (no chunks retrieved)")
    for chunk in response.retrieved_chunks:
        meta = chunk.metadata
        print(
            f"  - {meta.get('filename')} ({meta.get('month')} {meta.get('year')}) "
            f"[score={chunk.score:.3f}]"
        )


if __name__ == "__main__":
    main()
