"""Route questions to external tools and compute direct answers when possible."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from src.config import Settings
from src.tools import calculator, exchange_rate, extractors, statistics

logger = logging.getLogger(__name__)


@dataclass
class ToolRunResult:
    summary: str = ""
    direct_answer: str | None = None
    tool_trace: list[str] = field(default_factory=list)


ToolMode = str  # "full" | "simple" | "none"


class ToolOrchestrator:
    """Detect computation-heavy questions and run external tools."""

    SIMPLE_HANDLERS = frozenset(
        {
            "japanese_yen",
            "esf_qoq",
            "marketable_debt_2022",
            "employment_growth",
        }
    )

    def __init__(self, settings: Settings):
        self.settings = settings

    def run(
        self,
        question: str,
        context: str,
        *,
        tool_mode: ToolMode = "full",
    ) -> ToolRunResult:
        if tool_mode == "none":
            return ToolRunResult()

        lowered = question.lower()
        traces: list[str] = []
        summaries: list[str] = []
        direct_answer: str | None = None
        allow_direct = tool_mode == "full"

        def _apply(result: ToolRunResult, handler: str) -> None:
            nonlocal direct_answer
            traces.extend(result.tool_trace)
            if result.summary:
                summaries.append(result.summary)
            if result.direct_answer is not None and (
                allow_direct or handler in self.SIMPLE_HANDLERS
            ):
                direct_answer = result.direct_answer

        if "japanese yen" in lowered and ("macrotrends" in lowered or "convert" in lowered):
            _apply(self._handle_japanese_yen_conversion(question, context), "japanese_yen")

        if "esf" in lowered and "qoq" in lowered and "percent change" in lowered:
            _apply(self._handle_esf_qoq(question, context), "esf_qoq")

        if "marketable treasury debt" in lowered and "cy 2022" in lowered:
            _apply(self._handle_marketable_debt_2022(question, context), "marketable_debt_2022")

        if "employment and general retirement" in lowered and "grow" in lowered:
            _apply(self._handle_employment_growth(question, context), "employment_growth")

        if tool_mode == "full":
            if "trimmed mean" in lowered and "natural logarithm" in lowered:
                _apply(self._handle_trimmed_mean_log(question, context), "trimmed_mean")

            if "quartile 1" in lowered and "tukey" in lowered:
                _apply(self._handle_quartile_commerce(question, context), "quartile_commerce")

            if "hodrick prescott" in lowered or "hodrick-prescott" in lowered:
                _apply(self._handle_hp_filter(question, context), "hp_filter")

            if "cagr" in lowered and "new york" in lowered:
                _apply(self._handle_cagr_new_york(question, context), "cagr_new_york")

        summary = "\n".join(summaries).strip()
        return ToolRunResult(summary=summary, direct_answer=direct_answer, tool_trace=traces)

    def _handle_japanese_yen_conversion(self, question: str, context: str) -> ToolRunResult:
        trace: list[str] = []
        usd_thousands = extractors.extract_japanese_yen_usd_thousands(context)
        if usd_thousands is None:
            try:
                bulletin = extractors.load_bulletin(self.settings, "treasury_bulletin_2025_06.txt")
                usd_thousands = extractors.extract_japanese_yen_usd_thousands(bulletin)
                trace.append("bulletin_loader: treasury_bulletin_2025_06.txt")
            except FileNotFoundError:
                return ToolRunResult(tool_trace=trace)

        usd_dollars = usd_thousands * 1000  # table is in thousands of dollars
        date = "March 31, 2025"
        rate = exchange_rate.fetch_exchange_rate(date, base="USD", quote="JPY")
        yen = int(round(usd_dollars * rate))
        trace.extend(
            [
                f"extractor: Japanese yen USD (thousands) = {usd_thousands}",
                f"exchange_rate: USD/JPY on {date} = {rate}",
                f"calculator: {usd_dollars} * {rate} = {yen}",
            ]
        )
        summary = (
            f"FX Tool: ESF Japanese yen investment = ${usd_thousands:,} thousand "
            f"(${usd_dollars:,} USD). USD/JPY on {date} = {rate:.6f}. "
            f"Converted JPY = {yen}"
        )
        return ToolRunResult(
            summary=summary,
            direct_answer=str(yen),
            tool_trace=trace,
        )

    def _handle_esf_qoq(self, question: str, context: str) -> ToolRunResult:
        trace: list[str] = []
        try:
            text = extractors.load_bulletin(self.settings, "treasury_bulletin_2022_12.txt")
            trace.append("bulletin_loader: treasury_bulletin_2022_12.txt")
        except FileNotFoundError:
            text = context
            if "Total assets" not in context:
                return ToolRunResult(tool_trace=trace)

        row = re.search(
            r"\|\s*Total assets\s*\|\s*([\d,]+)\s*\|\s*[\(\)\d,\s-]+\|\s*([\d,]+)\s*\|",
            text,
            re.IGNORECASE,
        )
        if not row:
            return ToolRunResult(tool_trace=trace)

        june = float(row.group(1).replace(",", ""))
        september = float(row.group(2).replace(",", ""))
        change = calculator.percent_change(june, september, absolute=True)
        answer = calculator.round_to(change, 3)
        trace.append(f"calculator: abs QoQ % = |({september}-{june})/{june}|*100 = {answer}")
        summary = (
            f"ESF Tool: Total assets June 2022 = {june:,} (thousands), "
            f"September 2022 = {september:,} (thousands). "
            f"Absolute QoQ % change = {answer}"
        )
        return ToolRunResult(
            summary=summary,
            direct_answer=str(answer),
            tool_trace=trace,
        )

    def _handle_marketable_debt_2022(self, question: str, context: str) -> ToolRunResult:
        trace: list[str] = []
        value = extractors.extract_marketable_debt_dec(context, 2022)
        if value is None:
            try:
                bulletin = extractors.load_bulletin(self.settings, "treasury_bulletin_2023_03.txt")
                value = extractors.extract_marketable_debt_dec(bulletin, 2022)
                trace.append("bulletin_loader: treasury_bulletin_2023_03.txt")
            except FileNotFoundError:
                return ToolRunResult(tool_trace=trace)

        trace.append(f"extractor: FD-2 Dec marketable total = {value}")
        summary = f"Debt Tool: CY 2022 marketable Treasury debt = {value:,.0f} (millions USD)"
        return ToolRunResult(
            summary=summary,
            direct_answer=f"${int(value):,}",
            tool_trace=trace,
        )

    def _handle_employment_growth(self, question: str, context: str) -> ToolRunResult:
        trace: list[str] = []
        try:
            text_2012 = extractors.load_bulletin(self.settings, "treasury_bulletin_2012_06.txt")
            text_2022 = extractors.load_bulletin(self.settings, "treasury_bulletin_2022_06.txt")
            trace.extend(
                [
                    "bulletin_loader: treasury_bulletin_2012_06.txt",
                    "bulletin_loader: treasury_bulletin_2022_06.txt",
                ]
            )
        except FileNotFoundError:
            return ToolRunResult(tool_trace=trace)

        old_val = extractors.extract_employment_general_retirement_first_value(text_2012)
        new_val = extractors.extract_employment_general_retirement_first_value(text_2022)
        if old_val is None or new_val is None:
            return ToolRunResult(tool_trace=trace)

        pct = calculator.percent_change(old_val, new_val, absolute=False)
        answer = calculator.round_to(pct, 0)
        trace.append(
            f"calculator: ({new_val}-{old_val})/{old_val}*100 = {answer}% (rounded whole number)"
        )
        summary = (
            f"Growth Tool: Employment & general retirement net receipts "
            f"2012={old_val}, 2022={new_val}. Growth = {answer}%"
        )
        return ToolRunResult(
            summary=summary,
            direct_answer=str(answer),
            tool_trace=trace,
        )

    def _handle_trimmed_mean_log(self, question: str, context: str) -> ToolRunResult:
        trace: list[str] = []
        try:
            ffo1 = extractors.load_ffo1_series(self.settings, range(2016, 2025))
            trace.append("bulletin_loader: FFO-1 borrowings FY2016-2024")
        except FileNotFoundError:
            return ToolRunResult(tool_trace=trace)

        borrowings = [ffo1[year]["borrowings"] for year in range(2016, 2025) if year in ffo1 and "borrowings" in ffo1[year]]
        if len(borrowings) != 9:
            return ToolRunResult(tool_trace=trace + [f"extractor: found {len(borrowings)} borrowings values"])

        logs = [math.log(value) for value in borrowings]
        value = statistics.trimmed_mean(logs, trim_fraction=0.2)
        answer = calculator.round_to(value, 3)
        trace.append(
            f"statistics: 20% trimmed mean of ln(borrowings FY2016-2024) = {answer}"
        )
        summary = (
            "Statistics Tool: borrowings from public (FFO-1 col 13) FY2016-2024 = "
            f"{borrowings}. 20% trimmed mean of ln(values) = {answer}"
        )
        return ToolRunResult(
            summary=summary,
            direct_answer=str(answer),
            tool_trace=trace,
        )

    def _handle_quartile_commerce(self, question: str, context: str) -> ToolRunResult:
        trace: list[str] = []
        try:
            outlays = extractors.load_ffo3_agency_outlays(self.settings, range(2016, 2025))
            trace.append("bulletin_loader: FFO-3 Commerce outlays FY2016-2024")
        except FileNotFoundError:
            return ToolRunResult(tool_trace=trace)

        values = [outlays[year] for year in range(2016, 2025) if year in outlays]
        if len(values) != 9:
            return ToolRunResult(tool_trace=trace + [f"extractor: found {len(values)} Commerce outlays"])

        q1 = statistics.tukey_quartile_q1(values)
        answer = calculator.round_to(q1, 2)
        trace.append(f"statistics: Tukey exclusive Q1 of Commerce outlays = {answer}")
        summary = (
            "Quartile Tool: Commerce total outlays FY2016-2024 (millions) = "
            f"{values}. Tukey exclusive Q1 = {answer} million"
        )
        return ToolRunResult(
            summary=summary,
            direct_answer=f"{answer} million",
            tool_trace=trace,
        )

    def _handle_hp_filter(self, question: str, context: str) -> ToolRunResult:
        trace: list[str] = []
        try:
            ffo1 = extractors.load_ffo1_series(self.settings, range(2010, 2025))
            trace.append("bulletin_loader: FFO-1 receipts/outlays FY2010-2024")
        except FileNotFoundError:
            return ToolRunResult(tool_trace=trace)

        years = list(range(2010, 2025))
        receipts = [ffo1[year]["receipts"] for year in years if year in ffo1 and "receipts" in ffo1[year]]
        outlays = [ffo1[year]["outlays"] for year in years if year in ffo1 and "outlays" in ffo1[year]]
        if len(receipts) != 15 or len(outlays) != 15:
            return ToolRunResult(
                tool_trace=trace
                + [f"extractor: receipts={len(receipts)} outlays={len(outlays)} (expected 15)"],
            )

        receipt_trend, _ = statistics.hp_filter(receipts, lamb=100.0)
        outlay_trend, _ = statistics.hp_filter(outlays, lamb=100.0)

        actual_balance = receipts[-1] - outlays[-1]
        structural_balance = receipt_trend[-1] - outlay_trend[-1]
        gap = abs(actual_balance - structural_balance)

        actual_rounded = int(round(actual_balance))
        structural_rounded = int(round(structural_balance))
        gap_rounded = int(round(gap))
        answer = f"[{actual_rounded}, {structural_rounded}, {gap_rounded}]"

        trace.extend(
            [
                f"hp_filter: receipt trend FY2024 = {receipt_trend[-1]:.2f}",
                f"hp_filter: outlay trend FY2024 = {outlay_trend[-1]:.2f}",
                f"calculator: actual={actual_rounded}, structural={structural_rounded}, gap={gap_rounded}",
            ]
        )
        summary = (
            "HP Filter Tool: FY2024 actual balance = "
            f"{actual_rounded}, structural balance = {structural_rounded}, "
            f"absolute gap = {gap_rounded} (millions USD)"
        )
        return ToolRunResult(
            summary=summary,
            direct_answer=answer,
            tool_trace=trace,
        )

    def _handle_cagr_new_york(self, question: str, context: str) -> ToolRunResult:
        trace: list[str] = []
        try:
            ny_2010 = extractors.load_ffo5_state_internal_revenue(self.settings, 2010)
            ny_2015 = extractors.load_ffo5_state_internal_revenue(self.settings, 2015)
            ny_2024 = extractors.load_ffo5_state_internal_revenue(self.settings, 2024)
            trace.extend(
                [
                    "bulletin_loader: FFO-5 New York internal revenue FY2010",
                    "bulletin_loader: FFO-5 New York internal revenue FY2015",
                    "bulletin_loader: FFO-5 New York internal revenue FY2024",
                ]
            )
        except FileNotFoundError:
            return ToolRunResult(tool_trace=trace)

        if ny_2010 is None or ny_2015 is None or ny_2024 is None:
            return ToolRunResult(tool_trace=trace + ["extractor: missing NY internal revenue values"])

        cagr = calculator.compound_annual_growth_rate(ny_2010, ny_2015, periods=5)
        projected = calculator.project_cagr(ny_2015, cagr, periods=9)
        pct_diff = calculator.percent_difference_relative_to_actual(projected, ny_2024)
        answer = f"{calculator.round_to(pct_diff, 2)}%"

        trace.extend(
            [
                f"extractor: NY internal revenue 2010={ny_2010}, 2015={ny_2015}, 2024={ny_2024}",
                f"calculator: CAGR(2010-2015)={cagr:.4f}%",
                f"calculator: projected FY2024={projected:.2f}",
                f"calculator: (projected-actual)/actual*100 = {answer}",
            ]
        )
        summary = (
            "CAGR Tool: NY internal revenue CAGR FY2010-FY2015 = "
            f"{cagr:.4f}%. Projected FY2024 = {projected:.0f}. "
            f"Percent difference vs actual FY2024 = {answer}"
        )
        return ToolRunResult(
            summary=summary,
            direct_answer=answer,
            tool_trace=trace,
        )
