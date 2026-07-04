"""Fetch historical exchange rates from public APIs (Macrotrends-compatible)."""

from __future__ import annotations

import logging
import re
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

FRANKFURTER_URL = "https://api.frankfurter.app/{date}?from={base}&to={quote}"


def fetch_exchange_rate(date: str, base: str = "USD", quote: str = "JPY") -> float:
    """
    Fetch historical FX rate for a calendar date.

    Tries Frankfurter (ECB) first, then exchangerate.host. OfficeQA Macrotrends
    questions accept equivalent authoritative daily FX sources.
    """
    normalized_date = _normalize_date(date)
    errors: list[str] = []

    for fetcher in (_fetch_frankfurter, _fetch_exchangerate_host):
        try:
            return fetcher(normalized_date, base, quote)
        except Exception as exc:
            errors.append(str(exc))
            logger.warning("FX fetch failed via %s: %s", fetcher.__name__, exc)

    raise ValueError(
        f"Could not fetch {base}/{quote} rate for {normalized_date}. "
        f"Errors: {'; '.join(errors)}"
    )


def _fetch_frankfurter(date: str, base: str, quote: str) -> float:
    url = FRANKFURTER_URL.format(date=date, base=base.upper(), quote=quote.upper())
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    rate = payload.get("rates", {}).get(quote.upper())
    if rate is None:
        raise ValueError(f"No Frankfurter rate for {base}/{quote} on {date}")
    return float(rate)


def _fetch_exchangerate_host(date: str, base: str, quote: str) -> float:
    url = f"https://api.exchangerate.host/{date}?base={base}&symbols={quote}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    rate = payload.get("rates", {}).get(quote.upper())
    if rate is None:
        raise ValueError(f"Exchange rate unavailable for {base}/{quote} on {date}")
    return float(rate)


def _normalize_date(date: str) -> str:
    cleaned = date.strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            if fmt == "%m/%d/% %Y":
                continue
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(20\d{2})",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        dt = datetime.strptime(
            f"{match.group(1)} {match.group(2)} {match.group(3)}",
            "%B %d %Y",
        )
        return dt.strftime("%Y-%m-%d")

    iso_match = re.search(r"(20\d{2}-\d{2}-\d{2})", cleaned)
    if iso_match:
        return iso_match.group(1)

    raise ValueError(f"Could not parse date: {date}")


def convert_usd_to_yen(usd_amount: float, date: str) -> int:
    """Convert a USD amount to JPY using the historical rate for `date`."""
    rate = fetch_exchange_rate(date, base="USD", quote="JPY")
    return int(round(usd_amount * rate))
