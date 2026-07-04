#!/usr/bin/env python3
"""Download OfficeQA Treasury reports and evaluation CSV."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.ingestion.downloader import DatasetDownloadError, prepare_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OfficeQA dataset artifacts.")
    parser.add_argument("--force", action="store_true", help="Re-download data.")
    args = parser.parse_args()

    settings = load_settings()
    try:
        summary = prepare_data(settings, force=args.force)
    except DatasetDownloadError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    print("\nData preparation complete:")
    print(f"  Report files found: {summary['report_files']}")
    print(f"  Reports in indexed years: {summary['indexed_year_reports']}")
    print(f"  Evaluation questions: {summary['eval_questions']}")
    print(f"  Reports directory: {settings.data.reports_dir}")
    print(f"  Evaluation CSV: {settings.data.eval_csv}")

    if summary["report_files"] == 0:
        print(
            "\nNo Treasury report files were found.\n"
            "Ensure you have Hugging Face access and run:\n"
            "  huggingface-cli login\n"
            "  python scripts/download_data.py --force"
        )


if __name__ == "__main__":
    main()
