"""Parse year/month filters from natural-language Treasury questions."""

from __future__ import annotations

import re
from dataclasses import dataclass

MONTH_NAMES = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

MONTH_NUMBER_TO_NAME = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

# Treasury bulletins are quarterly: _03, _06, _09, _12
CALENDAR_MONTH_TO_PERIOD = {
    1: "03",
    2: "03",
    3: "03",
    4: "06",
    5: "06",
    6: "06",
    7: "09",
    8: "09",
    9: "09",
    10: "12",
    11: "12",
    12: "12",
}

PERIOD_TO_PUBLICATION_MONTH = {
    "03": "March",
    "06": "June",
    "09": "September",
    "12": "December",
}

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class TemporalFilters:
    year: int | None = None
    month: str | None = None
    period_code: str | None = None
    expected_filename: str | None = None

    def merge(
        self,
        year: int | None = None,
        month: str | None = None,
        period_code: str | None = None,
    ) -> "TemporalFilters":
        """Explicit CLI/eval arguments override parsed values when provided."""
        merged_year = year if year is not None else self.year
        merged_month = month if month is not None else self.month
        merged_period = period_code if period_code is not None else self.period_code

        if merged_period is None and merged_month:
            month_key = merged_month.strip().lower()
            month_num = MONTH_NAMES.get(month_key)
            if month_num is None and merged_month in MONTH_NUMBER_TO_NAME.values():
                month_num = next(
                    num for num, name in MONTH_NUMBER_TO_NAME.items() if name == merged_month
                )
            if month_num is not None:
                merged_period = CALENDAR_MONTH_TO_PERIOD[month_num]
                merged_month = MONTH_NUMBER_TO_NAME[month_num]
        elif merged_period and merged_month is None:
            merged_month = PERIOD_TO_PUBLICATION_MONTH.get(merged_period)

        expected_filename = None
        if merged_year and merged_period:
            expected_filename = f"treasury_bulletin_{merged_year}_{merged_period}.txt"

        return TemporalFilters(
            year=merged_year,
            month=merged_month,
            period_code=merged_period,
            expected_filename=expected_filename,
        )


def parse_temporal_filters(question: str) -> TemporalFilters:
    """Extract year and calendar month from a question string."""
    lowered = question.lower()
    year: int | None = None
    month_num: int | None = None

    year_match = YEAR_PATTERN.search(question)
    if year_match:
        year = int(year_match.group(0))

    for token, number in MONTH_NAMES.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            month_num = number
            break

    period_code = CALENDAR_MONTH_TO_PERIOD.get(month_num) if month_num else None
    month_name = MONTH_NUMBER_TO_NAME.get(month_num) if month_num else None
    expected_filename = None
    if year and period_code:
        expected_filename = f"treasury_bulletin_{year}_{period_code}.txt"

    return TemporalFilters(
        year=year,
        month=month_name,
        period_code=period_code,
        expected_filename=expected_filename,
    )
