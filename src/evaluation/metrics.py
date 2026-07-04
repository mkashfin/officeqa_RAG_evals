"""Evaluation metrics for retrieval and generation."""

from __future__ import annotations

import re
from typing import Iterable

from src.models import GenerationMetrics, RetrievalMetrics, RetrievedChunk

FINAL_ANSWER_PATTERN = re.compile(
    r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>",
    re.IGNORECASE | re.DOTALL,
)
NUMBER_PATTERN = re.compile(
    r"[-+]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?",
)


def _normalize_document_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\u2212", "-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_final_answer(text: str) -> str:
    """Extract the model's final answer from optional XML tags."""
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
    """Extract numeric values from answer text."""
    values: list[float] = []
    for match in NUMBER_PATTERN.findall(text):
        parsed = _parse_number(match)
        if parsed is not None:
            values.append(parsed)
    return values


def _is_likely_year(value: float) -> bool:
    if value != int(value):
        return False
    year = int(value)
    return 1900 <= year <= 2100


def _has_significant_text(text: str) -> bool:
    stripped = re.sub(NUMBER_PATTERN, " ", text)
    stripped = re.sub(r"[$%]", " ", stripped)
    words = [w for w in re.findall(r"[A-Za-z]{3,}", stripped)]
    return len(words) >= 1


def document_hit_at_k(
    retrieved_chunks: list[RetrievedChunk],
    reference_documents: Iterable[str],
    k: int = 5,
) -> bool:
    """Return True if any reference document appears in top-k retrieved chunks."""
    refs = {_normalize_document_name(doc) for doc in reference_documents if doc}
    if not refs:
        return False

    for chunk in retrieved_chunks[:k]:
        filename = _normalize_document_name(str(chunk.metadata.get("filename", "")))
        if not filename:
            continue
        if filename in refs:
            return True
    return False


def reciprocal_rank(
    retrieved_chunks: list[RetrievedChunk],
    reference_documents: Iterable[str],
) -> float:
    """Compute reciprocal rank of the first relevant document."""
    refs = {_normalize_document_name(doc) for doc in reference_documents if doc}
    if not refs:
        return 0.0

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        filename = _normalize_document_name(str(chunk.metadata.get("filename", "")))
        if filename in refs:
            return 1.0 / rank
    return 0.0


def chunk_recall_at_k(
    retrieved_chunks: list[RetrievedChunk],
    reference_documents: Iterable[str],
    k: int = 5,
) -> float:
    """Fraction of reference documents found in top-k."""
    refs = [doc for doc in reference_documents if doc]
    if not refs:
        return 0.0

    matched = sum(
        1 for ref in refs if document_hit_at_k(retrieved_chunks, [ref], k=k)
    )
    return matched / len(refs)


def aggregate_retrieval_metrics(
    hits: list[bool],
    reciprocal_ranks: list[float],
    recalls: list[float],
) -> RetrievalMetrics:
    total = max(len(hits), 1)
    return RetrievalMetrics(
        hit_rate_at_k=sum(hits) / total,
        mrr=sum(reciprocal_ranks) / total,
        recall=sum(recalls) / total,
    )


def factual_accuracy_score(
    generated: str,
    reference: str,
    tolerance: float = 0.01,
) -> bool:
    """
    Check factual accuracy using OfficeQA-style numeric matching.

    - Extracts numbers from both answers
    - Ignores year-like spurious values in long model responses
    - Compares absolute values for percent-style answers
    - Uses ±tolerance relative error (PRD: ±1%)
    """
    predicted = extract_final_answer(generated)
    if not predicted:
        return False

    if "unable to determine" in predicted.lower() or "cannot find" in predicted.lower():
        return False

    if "cannot be determined" in predicted.lower() or "impossible to" in predicted.lower():
        return False

    reference_nums = extract_numbers(reference)
    predicted_nums = extract_numbers(predicted)

    if not reference_nums:
        ref_norm = _normalize_text(reference)
        pred_norm = _normalize_text(predicted)
        return bool(ref_norm) and (ref_norm in pred_norm or pred_norm in ref_norm)

    if not predicted_nums:
        return False

    target = reference_nums[0]
    gt_has_text = _has_significant_text(reference)
    filter_years = not (_is_likely_year(target) or gt_has_text)

    candidate_values = [
        value
        for value in predicted_nums
        if not (filter_years and _is_likely_year(value))
    ]
    if not candidate_values:
        candidate_values = predicted_nums

    best_diff = float("inf")
    for value in candidate_values:
        comparisons = [value]
        if target < 0 or value < 0 or "%" in reference or "percent" in reference.lower():
            comparisons.append(abs(value))

        for candidate in comparisons:
            if target == 0:
                if abs(candidate) <= tolerance:
                    return True
                continue

            diff = abs(target - candidate) / abs(target)
            best_diff = min(best_diff, diff)
            if diff <= tolerance:
                return True

    return False


def split_claims(text: str) -> list[str]:
    """Split answer text into simple claim sentences."""
    parts = re.split(r"[.!?;\n]+", text)
    return [part.strip() for part in parts if len(part.strip()) > 8]


def _number_variants(value: float) -> set[str]:
    variants: set[str] = set()
    if value == int(value):
        variants.add(str(int(value)))
        variants.add(f"{int(value):,}")
    variants.add(str(value))
    variants.add(f"{value:.3f}".rstrip("0").rstrip("."))
    return {v.replace(",", "").lower() for v in variants if v}


def _build_evidence(context: str, tool_outputs: str = "") -> str:
    parts = [part.strip() for part in (context, tool_outputs) if part and part.strip()]
    return "\n\n".join(parts)


def _numeric_answer_grounded(answer_text: str, evidence: str) -> bool | None:
    """
    Return True/False for numeric-only answers, or None when claim-based scoring applies.
    """
    nums = extract_numbers(answer_text)
    if not nums:
        return None

    evidence_norm = evidence.lower().replace(",", "")
    for num in nums:
        for variant in _number_variants(num):
            if variant and variant in evidence_norm:
                return True

    evidence_lower = evidence.lower()
    if "tool:" in evidence_lower or "calculator:" in evidence_lower:
        for num in nums:
            if str(num) in evidence_norm or str(int(num)) in evidence_norm:
                return True

    # Short numeric answers without prose fall through to numeric check.
    if len(re.findall(r"[a-z]{4,}", answer_text.lower())) == 0:
        return False

    return None


def groundedness_score(
    answer: str,
    context: str,
    tool_outputs: str = "",
) -> float:
    """Estimate groundedness as support in retrieved context and tool evidence."""
    answer_text = extract_final_answer(answer)
    if not answer_text:
        return 0.0

    if "unable to determine" in answer_text.lower():
        return 1.0

    evidence = _build_evidence(context, tool_outputs)
    numeric_grounded = _numeric_answer_grounded(answer_text, evidence)
    if numeric_grounded is not None:
        return 1.0 if numeric_grounded else 0.0

    claims = split_claims(answer_text)
    if not claims:
        return 0.0

    context_lower = evidence.lower()
    supported = 0
    for claim in claims:
        words = [w for w in re.findall(r"\w+", claim.lower()) if len(w) > 3]
        if not words:
            continue
        overlap = sum(1 for word in words if word in context_lower)
        if overlap / len(words) >= 0.5:
            supported += 1

    return supported / len(claims)


def hallucination_rate(
    answer: str,
    context: str,
    tool_outputs: str = "",
) -> float:
    """Fraction of claims not supported by retrieved context."""
    grounded = groundedness_score(answer, context, tool_outputs=tool_outputs)
    return max(0.0, 1.0 - grounded)


def aggregate_generation_metrics(
    groundedness_values: list[float],
    factual_flags: list[bool],
    hallucination_values: list[float],
) -> GenerationMetrics:
    total = max(len(groundedness_values), 1)
    return GenerationMetrics(
        groundedness=sum(groundedness_values) / total,
        factual_accuracy=sum(factual_flags) / max(len(factual_flags), 1),
        hallucination_rate=sum(hallucination_values) / total,
    )
