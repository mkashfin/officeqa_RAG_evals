"""Load Treasury markdown/text documents."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import Settings
from src.ingestion.downloader import discover_report_files
from src.metadata.extractor import extract_metadata
from src.models import TreasuryDocument

logger = logging.getLogger(__name__)


def load_document(path: Path) -> TreasuryDocument | None:
    """Read a single document and attach metadata."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None

    if not content:
        return None

    metadata = extract_metadata(path, content)
    return TreasuryDocument(content=content, metadata=metadata)


def load_documents(
    settings: Settings,
    years: list[int] | None = None,
) -> list[TreasuryDocument]:
    """Load all Treasury reports, optionally filtered by year."""
    target_years = set(years or settings.data.years)
    documents: list[TreasuryDocument] = []

    for path in discover_report_files(settings):
        document = load_document(path)
        if document is None:
            continue
        if document.metadata.year not in target_years:
            continue
        documents.append(document)

    logger.info(
        "Loaded %d documents for years %s",
        len(documents),
        sorted(target_years),
    )
    return documents
