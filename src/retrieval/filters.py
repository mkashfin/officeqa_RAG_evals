"""Resolve retrieval filters from CLI args and question text."""

from __future__ import annotations

from src.metadata.query_parser import TemporalFilters, parse_temporal_filters


def resolve_temporal_filters(
    question: str,
    year: int | list[int] | None = None,
    month: str | None = None,
    period_code: str | None = None,
) -> TemporalFilters:
    parsed = parse_temporal_filters(question)
    return parsed.merge(year=year, month=month, period_code=period_code)
