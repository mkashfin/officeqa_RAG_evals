"""Post-generation guardrails for grounded, factual answers."""

from __future__ import annotations

import re

HEDGE_PATTERNS = (
    r"\bassume\b",
    r"\bestimate\b",
    r"\bprojection\b",
    r"\bunfortunately\b",
    r"\bcannot calculate\b",
    r"\bcannot be determined\b",
    r"\bimpossible to\b",
    r"\bwithout additional information\b",
    r"\blet'?s assume\b",
    r"\bfor example, if\b",
)

REFUSAL_TEXT = "Unable to determine from the provided documents."

FINAL_ANSWER_PATTERN = re.compile(
    r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>",
    re.IGNORECASE | re.DOTALL,
)
NUMBER_PATTERN = re.compile(
    r"[-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?",
)


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    matches = FINAL_ANSWER_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def _parse_number(raw: str) -> float | None:
    token = raw.strip().replace("$", "").replace(",", "").replace("%", "")
    if not token or token in {"+", "-"}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in NUMBER_PATTERN.findall(text):
        parsed = _parse_number(match)
        if parsed is not None:
            values.append(parsed)
    return values


def _contains_hedge(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in HEDGE_PATTERNS)


def _clean_final_value(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip('"').strip("'"))


def _format_final_answer(value: str) -> str:
    return f"<FINAL_ANSWER>{_clean_final_value(value)}</FINAL_ANSWER>"


def _compact_numeric_answer(text: str) -> str:
    """Reduce verbose FINAL_ANSWER prose to a concise numeric value when possible."""
    cleaned = _clean_final_value(text)
    if len(cleaned) <= 100:
        return cleaned

    nums = extract_numbers(cleaned)
    if not nums:
        return cleaned

    value = nums[-1]
    if "%" in cleaned and "%" not in str(value):
        rendered = f"{value:g}%"
    elif "$" in cleaned:
        rendered = f"${int(value):,}" if value == int(value) else f"${value:g}"
    elif value == int(value):
        rendered = str(int(value))
    else:
        rendered = str(value)

    return rendered


def enforce_final_answer(raw: str, *, compact_verbose: bool = True) -> str:
    if not raw or not raw.strip():
        return _format_final_answer(REFUSAL_TEXT)

    extracted = extract_final_answer(raw)
    if extracted:
        if _contains_hedge(extracted):
            return _format_final_answer(REFUSAL_TEXT)
        if compact_verbose and len(extracted) > 100:
            extracted = _compact_numeric_answer(extracted)
        if len(extracted) > 250:
            return _format_final_answer(REFUSAL_TEXT)
        return _format_final_answer(extracted)

    if len(raw.strip()) > 80 or _contains_hedge(raw):
        return _format_final_answer(REFUSAL_TEXT)

    return _format_final_answer(raw.strip())


def build_evidence_context(context: str, tool_outputs: str = "") -> str:
    parts = [part.strip() for part in (context, tool_outputs) if part and part.strip()]
    return "\n\n".join(parts)


def _number_variants(value: float) -> set[str]:
    variants: set[str] = set()
    if value == int(value):
        variants.add(str(int(value)))
        variants.add(f"{int(value):,}")
    variants.add(str(value))
    variants.add(f"{value:.3f}".rstrip("0").rstrip("."))
    return {v.replace(",", "").lower() for v in variants if v}


def is_answer_grounded(answer: str, evidence: str) -> bool:
    final = extract_final_answer(answer)
    if not final or final.lower() == REFUSAL_TEXT.lower():
        return True
    if _contains_hedge(final):
        return False

    evidence_norm = evidence.lower().replace(",", "")
    nums = extract_numbers(final)

    if nums:
        for num in nums:
            for variant in _number_variants(num):
                if variant and variant in evidence_norm:
                    return True
        if "tool:" in evidence.lower() or "calculator:" in evidence.lower():
            return True

    if not nums and len(final) >= 4:
        return final.lower() in evidence.lower()

    return False


def apply_guardrails(
    raw_answer: str,
    context: str,
    tool_outputs: str = "",
    *,
    require_grounding: bool = True,
    compact_verbose: bool = True,
) -> str:
    formatted = enforce_final_answer(raw_answer, compact_verbose=compact_verbose)
    if not require_grounding:
        return formatted

    evidence = build_evidence_context(context, tool_outputs)
    if is_answer_grounded(formatted, evidence):
        return formatted

    final = extract_final_answer(formatted)
    if final and extract_numbers(final) and tool_outputs:
        return formatted

    return _format_final_answer(REFUSAL_TEXT)
