"""Extract year, month, and other metadata from Treasury report filenames and content."""

from __future__ import annotations

import re
from pathlib import Path

from src.metadata.query_parser import PERIOD_TO_PUBLICATION_MONTH
from src.models import DocumentMetadata

MONTH_NAMES = {
    "january": "January",
    "february": "February",
    "march": "March",
    "april": "April",
    "may": "May",
    "june": "June",
    "july": "July",
    "august": "August",
    "september": "September",
    "october": "October",
    "november": "November",
    "december": "December",
    "jan": "January",
    "feb": "February",
    "mar": "March",
    "apr": "April",
    "jun": "June",
    "jul": "July",
    "aug": "August",
    "sep": "September",
    "sept": "September",
    "oct": "October",
    "nov": "November",
    "dec": "December",
}

YEAR_PATTERN = re.compile(r"(20\d{2}|19\d{2})")
TREASURY_BULLETIN_PATTERN = re.compile(
    r"treasury_bulletin_(\d{4})_(\d{1,2})",
    re.IGNORECASE,
)
MONTH_NUMERIC_PATTERN = re.compile(r"(?<!\d)(0?[1-9]|1[0-2])(?!\d)")
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def _normalize_month(raw: str) -> str | None:
    key = raw.strip().lower()
    if key in MONTH_NAMES:
        return MONTH_NAMES[key]
    if key.isdigit():
        return _month_from_numeric(key)
    return None


def _month_from_numeric(value: str) -> str | None:
    month_num = int(value)
    month_list = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    if 1 <= month_num <= 12:
        return month_list[month_num - 1]
    return None


def extract_year_month_from_filename(filename: str) -> tuple[int | None, str | None, str | None]:
    """Parse year, publication month, and quarterly period code from filename."""
    bulletin_match = TREASURY_BULLETIN_PATTERN.search(filename)
    if bulletin_match:
        year = int(bulletin_match.group(1))
        period_code = bulletin_match.group(2).zfill(2)
        month = PERIOD_TO_PUBLICATION_MONTH.get(period_code)
        return year, month, period_code

    lowered = filename.lower().replace("-", "_").replace(" ", "_")

    year_match = YEAR_PATTERN.search(lowered)
    year = int(year_match.group(1)) if year_match else None

    month: str | None = None
    for token, normalized in MONTH_NAMES.items():
        if token in lowered:
            month = normalized
            break

    if month is None:
        numeric_match = MONTH_NUMERIC_PATTERN.search(lowered)
        if numeric_match:
            month = _month_from_numeric(numeric_match.group(1))

    period_code = None
    return year, month, period_code


def extract_first_heading(content: str) -> str:
    """Return the first markdown heading as a section label."""
    match = HEADING_PATTERN.search(content)
    return match.group(1).strip() if match else ""


def extract_metadata(path: Path, content: str) -> DocumentMetadata:
    """Build metadata for a Treasury report document."""
    year, month, period_code = extract_year_month_from_filename(path.name)

    if year is None:
        content_year = YEAR_PATTERN.search(content[:2000])
        if content_year:
            year = int(content_year.group(1))

    if month is None:
        for token, normalized in MONTH_NAMES.items():
            if re.search(rf"\b{token}\b", content[:2000], re.IGNORECASE):
                month = normalized
                break

    if year is None:
        year = 0
    if month is None:
        month = "Unknown"

    return DocumentMetadata(
        year=year,
        month=month,
        filename=path.name,
        section=extract_first_heading(content),
        period_code=period_code or "",
    )
