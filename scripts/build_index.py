#!/usr/bin/env python3
"""Build vector indexes for baseline and engineered RAG pipelines."""

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
    parser = argparse.ArgumentParser(description="Build Treasury RAG vector indexes.")
    parser.add_argument(
        "--system",
        choices=["baseline", "engineered", "both"],
        default="both",
        help="Which index to build.",
    )
    parser.add_argument("--reset", action="store_true", help="Rebuild index from scratch.")
    args = parser.parse_args()

    settings = load_settings()
    settings.ensure_directories()

    if args.system in ("baseline", "both"):
        baseline = BaselinePipeline(settings)
        count = baseline.build_index(reset=args.reset)
        print(f"Baseline index: {count} chunks indexed.")

    if args.system in ("engineered", "both"):
        print(
            "Building engineered index (BGE-large). "
            "On CPU this may take 10–20 minutes for ~4000 chunks; "
            "a GPU will be much faster. Progress bars will appear below.\n"
        )
        engineered = EngineeredPipeline(settings)
        count = engineered.build_index(reset=args.reset)
        print(f"Engineered index: {count} chunks indexed.")


if __name__ == "__main__":
    main()
