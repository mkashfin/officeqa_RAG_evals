"""Load Treasury bulletin files and extract table values."""

from __future__ import annotations

import re
from pathlib import Path

from src.config import Settings


def load_bulletin(settings: Settings, filename: str) -> str:
    path = (
        settings.data.reports_dir
        / "treasury_bulletins_parsed"
        / "transformed"
        / filename
    )
    if not path.exists():
        raise FileNotFoundError(f"Bulletin not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def extract_esf_total_assets(text: str, date_label: str) -> float | None:
    """Extract ESF total assets for June 30 or Sept 30 from ESF-1 table."""
    if not re.search(r"TABLE ESF-1", text, re.IGNORECASE):
        return None

    header_match = re.search(
        rf"\|\s*Assets, liabilities.*?\|\s*([^\|]*{re.escape(date_label)}[^\|]*)\|",
        text,
        re.IGNORECASE,
    )
    if not header_match:
        return None

    row_match = re.search(
        r"\|\s*Total assets\s*\|\s*([\d,]+)\s*\|",
        text,
        re.IGNORECASE,
    )
    if not row_match:
        return None
    return float(row_match.group(1).replace(",", ""))


def extract_marketable_debt_dec(text: str, year: int) -> float | None:
    """Extract FD-2 marketable total (millions USD) for December year-end."""
    candidates: list[float] = []
    for match in re.finditer(
        r"\|\s*Dec\.\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|",
        text,
        re.IGNORECASE,
    ):
        marketable = float(match.group(2).replace(",", ""))
        if marketable > 1_000_000:
            candidates.append(marketable)
    return candidates[0] if candidates else None


def extract_japanese_yen_usd_thousands(text: str, date_label: str = "Mar. 31, 2025") -> float | None:
    """Extract Japanese yen foreign exchange asset (USD, thousands) for a date column."""
    if "Japanese yen" not in text:
        return None

    header_idx = text.find("TABLE ESF-1")
    snippet = text[header_idx : header_idx + 5000] if header_idx >= 0 else text

    columns = [part.strip() for part in snippet.split("|")]
    target_idx = None
    for idx, col in enumerate(columns):
        if date_label.replace(" ", "").lower() in col.replace(" ", "").lower():
            target_idx = idx
            break
    if target_idx is None and "Mar. 31, 2025" in date_label:
        target_idx = None
        for idx, col in enumerate(columns):
            if "mar. 31, 2025" in col.lower():
                target_idx = idx
                break

    row_match = re.search(
        r"\|\s*Japanese yen\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|",
        snippet,
        re.IGNORECASE,
    )
    if not row_match:
        return None

    values = [float(v.replace(",", "")) for v in row_match.groups()]
    if target_idx is not None:
        # Fallback: last column is usually the end-of-period value.
        pass
    return values[-1]


def extract_employment_general_retirement_first_value(text: str) -> float | None:
    match = re.search(
        r"\|\s*Employment and general retirement\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return float(match.group(1))


_FYO_ROW = re.compile(
    r"^\|\s*(20\d{2})\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|",
    re.MULTILINE,
)
_BORROWINGS_ROW_NAN = re.compile(
    r"^\|\s*(20\d{2})\s*\|\s*nan\s*\|\s*([\d,\-]+)\s*\|\s*([\d,\-]+)\s*\|\s*([\d,\-]+)\s*\|",
    re.MULTILINE | re.IGNORECASE,
)
_BORROWINGS_ROW_NUM = re.compile(
    r"^\|\s*(20\d{2})\s*\|\s*([\d,\-]+)\s*\|\s*([\d,\-]+)\s*\|\s*([\d,\-]+)\s*\|",
    re.MULTILINE,
)
_FFO3_AGENCY_ROW = re.compile(
    r"^\|\s*(20\d{2})\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|",
    re.MULTILINE,
)

FFO1_BULLETINS: tuple[tuple[str, range], ...] = (
    ("treasury_bulletin_2014_12.txt", range(2010, 2013)),
    ("treasury_bulletin_2017_12.txt", range(2013, 2018)),
    ("treasury_bulletin_2020_12.txt", range(2018, 2021)),
    ("treasury_bulletin_2024_12.txt", range(2021, 2025)),
)

FFO3_BULLETINS: tuple[tuple[str, range], ...] = (
    ("treasury_bulletin_2020_12.txt", range(2016, 2021)),
    ("treasury_bulletin_2024_12.txt", range(2021, 2025)),
)


def _ffo1_sections(text: str) -> tuple[str, str]:
    start = text.find("TABLE FFO-1")
    if start < 0:
        return "", ""
    continued_markers = (
        "Borrowing from the public-Federal securities, continued > Total 10+11-12",
        "Federal securities, continued > Agency securities",
        "Means of financing\u2014net transactions, continued",
        "Means of financing-net transactions, continued",
    )
    continued = -1
    for marker in continued_markers:
        idx = text.find(marker, start)
        if idx >= 0:
            continued = idx
            break
    if continued < 0:
        return text[start:], ""
    continued_text = text[continued : continued + 12000]
    stop_markers = ("\nNote:", "\nTABLE FFO-2", "\nTABLE FFO-3")
    for marker in stop_markers:
        idx = continued_text.find(marker)
        if idx >= 0:
            continued_text = continued_text[:idx]
    return text[start:continued], continued_text


def _annual_fy_line(year: int, line: str) -> bool:
    if str(year) not in line.split("|", 2)[1]:
        return False
    lowered = line.lower()
    if "est" in lowered or "to date" in lowered:
        return False
    month_tokens = (
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    )
    return not any(token in lowered for token in month_tokens)


def _parse_ffo1_from_text(text: str) -> dict[int, dict[str, float]]:
    main, continued = _ffo1_sections(text)
    parsed: dict[int, dict[str, float]] = {}

    for match in _FYO_ROW.finditer(main):
        year = int(match.group(1))
        line = match.group(0)
        if not _annual_fy_line(year, line):
            continue
        receipts = float(match.group(2).replace(",", ""))
        outlays = float(match.group(5).replace(",", ""))
        if receipts < 1_000_000 or outlays < 1_000_000:
            continue
        parsed[year] = {"receipts": receipts, "outlays": outlays}

    for match in _BORROWINGS_ROW_NAN.finditer(continued):
        year = int(match.group(1))
        line = match.group(0)
        if not _annual_fy_line(year, line):
            continue
        borrowings = float(match.group(4).replace(",", ""))
        if borrowings < 400_000:
            continue
        parsed.setdefault(year, {})["borrowings"] = borrowings

    for match in _BORROWINGS_ROW_NUM.finditer(continued):
        year = int(match.group(1))
        line = match.group(0)
        if not _annual_fy_line(year, line) or "nan" in line.lower():
            continue
        borrowings = float(match.group(4).replace(",", ""))
        if borrowings < 400_000:
            continue
        parsed.setdefault(year, {})["borrowings"] = borrowings

    return parsed


def _first_ffo3_section(text: str) -> str:
    start = text.find("TABLE FFO-3")
    if start < 0:
        return ""
    markers = (
        "TABLE FFO-3—On-Budget and Off-Budget Outlays by Agency, continued",
        "TABLE FFO-3-On-Budget and Off-Budget Outlays by Agency, continued",
        ", continued",
    )
    end = len(text)
    for marker in markers:
        idx = text.find(marker, start + 12)
        if idx >= 0:
            end = min(end, idx)
    return text[start:end]


def load_ffo1_series(settings: Settings, years: range) -> dict[int, dict[str, float]]:
    """Load FFO-1 receipts, outlays, and borrowings-from-public totals by fiscal year."""
    series: dict[int, dict[str, float]] = {}
    for filename, year_range in FFO1_BULLETINS:
        needed = [year for year in years if year in year_range]
        if not needed:
            continue
        text = load_bulletin(settings, filename)
        for year, values in _parse_ffo1_from_text(text).items():
            if year in years:
                series[year] = values
    return series


def _ffo3_section(text: str) -> str:
    return _first_ffo3_section(text)


def load_ffo3_agency_outlays(
    settings: Settings,
    years: range,
) -> dict[int, float]:
    """Load annual Commerce total outlays from FFO-3 (on-budget and off-budget)."""
    series: dict[int, float] = {}
    for filename, year_range in FFO3_BULLETINS:
        needed = [year for year in years if year in year_range]
        if not needed:
            continue
        section = _ffo3_section(load_bulletin(settings, filename))
        for match in _FFO3_AGENCY_ROW.finditer(section):
            year = int(match.group(1))
            if year not in years:
                continue
            line = match.group(0)
            if not _annual_fy_line(year, line):
                continue
            legislative = float(match.group(2))
            judicial = float(match.group(3))
            if legislative > 12_000 or judicial > 12_000:
                continue
            commerce = float(match.group(5).replace(",", ""))
            if not 5_000 <= commerce <= 30_000:
                continue
            series[year] = commerce
    return series


def extract_ffo5_state_internal_revenue(
    text: str,
    state: str = "New York",
) -> float | None:
    """Extract total internal revenue collections (column 1) for a state from FFO-5."""
    start = text.find("TABLE FFO-5")
    if start < 0:
        return None
    section = text[start : start + 12000]
    pattern = re.compile(
        rf"^\|\s*{re.escape(state)}\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|",
        re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(section)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def load_ffo5_state_internal_revenue(
    settings: Settings,
    year: int,
    state: str = "New York",
) -> float | None:
    """Load FFO-5 total internal revenue for a state from the fiscal-year bulletin."""
    candidates = (
        f"treasury_bulletin_{year}_12.txt",
        f"treasury_bulletin_{year + 1}_03.txt",
        f"treasury_bulletin_{year + 1}_06.txt",
    )
    for filename in candidates:
        try:
            text = load_bulletin(settings, filename)
        except FileNotFoundError:
            continue
        value = extract_ffo5_state_internal_revenue(text, state=state)
        if value is not None:
            return value
    return None

