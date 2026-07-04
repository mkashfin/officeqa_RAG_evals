"""Download OfficeQA Treasury reports and evaluation CSV."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import pandas as pd

from src.config import Settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt", ".markdown"}
HF_DATASET = "databricks/officeqa"
HF_EVAL_CSV = "officeqa_full.csv"
HF_REPORTS_PATTERN = "treasury_bulletins_parsed/transformed/*.txt"

HF_ACCESS_HELP = """
OfficeQA data is hosted on Hugging Face (not GitHub) and requires access approval.

Steps:
  1. Visit https://huggingface.co/datasets/databricks/officeqa
  2. Log in and accept the dataset terms
  3. Run: huggingface-cli login
  4. Re-run: python scripts/download_data.py
"""


class DatasetDownloadError(RuntimeError):
    """Raised when OfficeQA artifacts cannot be downloaded."""


def _is_gated_dataset_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "gated",
            "401",
            "403",
            "unauthorized",
            "access",
            "authenticated",
            "permission",
        )
    )


def download_eval_csv(settings: Settings, force: bool = False) -> Path:
    """Download officeqa_full.csv from the Hugging Face dataset."""
    target = settings.data.eval_csv
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        logger.info("Evaluation CSV already exists: %s", target)
        return target

    logger.info("Downloading %s from Hugging Face dataset %s...", HF_EVAL_CSV, HF_DATASET)
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=settings.data.huggingface_dataset,
            repo_type="dataset",
            filename=HF_EVAL_CSV,
            local_dir=str(settings.data.raw_dir),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != target.resolve():
            shutil.copy2(downloaded_path, target)

        logger.info("Saved evaluation CSV to %s", target)
        return target
    except Exception as exc:
        if target.exists():
            logger.warning("Hugging Face download failed, using existing CSV: %s", exc)
            return target

        hint = HF_ACCESS_HELP if _is_gated_dataset_error(exc) else ""
        raise DatasetDownloadError(
            f"Could not download {HF_EVAL_CSV} from Hugging Face.\n"
            f"Error: {exc}\n{hint}"
        ) from exc


def download_treasury_reports(settings: Settings, force: bool = False) -> Path:
    """Download parsed Treasury bulletin text files from Hugging Face."""
    reports_root = settings.data.reports_dir
    transformed_dir = reports_root / "treasury_bulletins_parsed" / "transformed"
    reports_root.mkdir(parents=True, exist_ok=True)

    existing = list(transformed_dir.glob("*.txt")) if transformed_dir.exists() else []
    if existing and not force:
        logger.info("Treasury reports already present (%d files).", len(existing))
        return reports_root

    logger.info("Downloading Treasury bulletin text files from Hugging Face (~460MB)...")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=settings.data.huggingface_dataset,
            repo_type="dataset",
            allow_patterns=HF_REPORTS_PATTERN,
            local_dir=str(reports_root),
        )
        count = len(list(transformed_dir.glob("*.txt"))) if transformed_dir.exists() else 0
        logger.info("Downloaded %d Treasury report files.", count)
        return reports_root
    except Exception as exc:
        hint = HF_ACCESS_HELP if _is_gated_dataset_error(exc) else ""
        raise DatasetDownloadError(
            f"Could not download Treasury reports from Hugging Face.\n"
            f"Error: {exc}\n{hint}"
        ) from exc


def discover_report_files(settings: Settings) -> list[Path]:
    """Find markdown/text report files under raw data directories."""
    search_roots = [
        settings.data.reports_dir,
        settings.data.raw_dir,
    ]

    files: list[Path] = []
    seen: set[Path] = set()

    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if "officeqa" in path.name.lower() and path.suffix.lower() == ".csv":
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)

    return sorted(files)


def _count_reports_for_years(settings: Settings, report_files: list[Path]) -> int:
    """Count report files matching configured index years."""
    from src.metadata.extractor import extract_year_month_from_filename

    allowed = set(settings.data.years)
    count = 0
    for path in report_files:
        year, _, _ = extract_year_month_from_filename(path.name)
        if year in allowed:
            count += 1
    return count


def prepare_data(settings: Settings, force: bool = False) -> dict[str, int]:
    """
    Download and prepare dataset artifacts from Hugging Face.

    Returns summary counts for logging/reporting.
    """
    settings.ensure_directories()

    download_eval_csv(settings, force=force)
    download_treasury_reports(settings, force=force)

    report_files = discover_report_files(settings)
    indexed_reports = _count_reports_for_years(settings, report_files)

    if settings.data.eval_csv.exists():
        df = pd.read_csv(settings.data.eval_csv)
        eval_rows = len(df)
    else:
        eval_rows = 0

    return {
        "report_files": len(report_files),
        "indexed_year_reports": indexed_reports,
        "eval_questions": eval_rows,
    }
