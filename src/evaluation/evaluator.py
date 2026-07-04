"""Run evaluation for baseline and engineered RAG systems."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Protocol

import pandas as pd

from src.config import Settings, load_settings
from src.evaluation.metrics import (
    aggregate_generation_metrics,
    aggregate_retrieval_metrics,
    chunk_recall_at_k,
    document_hit_at_k,
    factual_accuracy_score,
    groundedness_score,
    hallucination_rate,
    reciprocal_rank,
)
from src.generation.generator import format_context
from src.models import EvalQuestion, EvaluationResult

logger = logging.getLogger(__name__)

QUESTION_COLUMNS = ("question", "Question", "query", "prompt")
ANSWER_COLUMNS = ("answer", "Answer", "ground_truth", "target", "correct_answer")
DOC_COLUMNS = (
    "source_files",
    "document",
    "documents",
    "source",
    "filename",
    "file_name",
    "doc_name",
    "reference",
    "report",
)
YEAR_COLUMNS = ("year", "Year", "report_year")
MONTH_COLUMNS = ("month", "Month", "report_month")


class RAGPipeline(Protocol):
    def ask(
        self,
        question: str,
        year: int | list[int] | None = None,
        month: str | None = None,
    ): ...


def _first_present(row: pd.Series, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in row and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value:
                return value
    return None


def _parse_documents(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[;,|\n]+", raw)
    return [part.strip() for part in parts if part.strip()]


def _metadata_from_filename(filename: str) -> tuple[int | None, str | None, str | None]:
    from src.metadata.extractor import TREASURY_BULLETIN_PATTERN, _month_from_numeric

    match = TREASURY_BULLETIN_PATTERN.search(filename)
    if not match:
        return None, None, None
    year = int(match.group(1))
    period_code = match.group(2).zfill(2)
    month = _month_from_numeric(period_code)
    return year, month, period_code


def _years_from_filenames(filenames: list[str]) -> set[int]:
    from src.metadata.extractor import TREASURY_BULLETIN_PATTERN

    years: set[int] = set()
    for name in filenames:
        match = TREASURY_BULLETIN_PATTERN.search(name)
        if match:
            years.add(int(match.group(1)))
    return years


def _month_from_filenames(filenames: list[str]) -> str | None:
    from src.metadata.extractor import TREASURY_BULLETIN_PATTERN, _month_from_numeric

    for name in filenames:
        match = TREASURY_BULLETIN_PATTERN.search(name)
        if match:
            return _month_from_numeric(match.group(2))
    return None


def load_eval_questions(settings: Settings) -> list[EvalQuestion]:
    """Load and filter evaluation questions from officeqa_full.csv."""
    csv_path = settings.data.eval_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found at {csv_path}. Run download_data.py first."
        )

    df = pd.read_csv(csv_path)
    allowed_years = set(settings.data.years)
    questions: list[EvalQuestion] = []

    for _, row in df.iterrows():
        question = _first_present(row, QUESTION_COLUMNS)
        answer = _first_present(row, ANSWER_COLUMNS)
        if not question or not answer:
            continue

        year_raw = _first_present(row, YEAR_COLUMNS)
        year = int(float(year_raw)) if year_raw else None

        doc_raw = _first_present(row, DOC_COLUMNS)
        documents = _parse_documents(doc_raw)
        file_years = _years_from_filenames(documents)

        matching_docs = [
            doc
            for doc in documents
            if _years_from_filenames([doc]).intersection(allowed_years)
        ]

        if file_years:
            matching_years = file_years.intersection(allowed_years)
            if not matching_years:
                continue
        elif year is not None and year not in allowed_years:
            continue

        primary_doc = matching_docs[0] if matching_docs else (documents[0] if documents else None)
        month = _first_present(row, MONTH_COLUMNS)
        period_code: str | None = None
        if primary_doc:
            doc_year, doc_month, doc_period = _metadata_from_filename(primary_doc)
            if doc_year in allowed_years:
                year = doc_year
            if month is None:
                month = doc_month
            period_code = doc_period
        elif month is None:
            month = _month_from_filenames(documents)

        questions.append(
            EvalQuestion(
                question=question,
                answer=answer,
                reference_documents=documents,
                year=year,
                month=month,
                period_code=period_code,
                raw=row.to_dict(),
            )
        )

    max_samples = settings.evaluation.max_eval_samples
    if max_samples is not None:
        questions = questions[: int(max_samples)]

    logger.info("Loaded %d evaluation questions.", len(questions))
    return questions


def evaluate_pipeline(
    pipeline: RAGPipeline,
    settings: Settings,
    system_name: str,
    questions: list[EvalQuestion] | None = None,
    use_metadata_filters: bool = True,
) -> EvaluationResult:
    """Evaluate a RAG pipeline on the OfficeQA question set."""
    eval_questions = questions or load_eval_questions(settings)
    top_k = settings.evaluation.top_k
    tolerance = settings.evaluation.factual_tolerance

    hits: list[bool] = []
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    groundedness_values: list[float] = []
    factual_flags: list[bool] = []
    hallucination_values: list[float] = []
    details: list[dict] = []

    for item in eval_questions:
        if use_metadata_filters:
            response = pipeline.ask(
                question=item.question,
                year=item.year,
                month=item.month,
                period_code=item.period_code,
            )
        else:
            response = pipeline.ask(question=item.question)

        hit = document_hit_at_k(response.retrieved_chunks, item.reference_documents, k=top_k)
        rr = reciprocal_rank(response.retrieved_chunks, item.reference_documents)
        recall = chunk_recall_at_k(response.retrieved_chunks, item.reference_documents, k=top_k)

        context = format_context(response.retrieved_chunks)
        tool_outputs = getattr(response, "tool_outputs", "") or ""
        grounded = groundedness_score(
            response.answer,
            context,
            tool_outputs=tool_outputs,
        )
        factual = factual_accuracy_score(response.answer, item.answer, tolerance)
        hallucinated = hallucination_rate(
            response.answer,
            context,
            tool_outputs=tool_outputs,
        )

        hits.append(hit)
        reciprocal_ranks.append(rr)
        recalls.append(recall)
        groundedness_values.append(grounded)
        factual_flags.append(factual)
        hallucination_values.append(hallucinated)

        details.append(
            {
                "question": item.question,
                "reference_answer": item.answer,
                "generated_answer": response.answer,
                "hit_at_k": hit,
                "reciprocal_rank": rr,
                "recall": recall,
                "groundedness": grounded,
                "factual_accuracy": factual,
                "hallucination_rate": hallucinated,
                "retrieved_files": [
                    chunk.metadata.get("filename") for chunk in response.retrieved_chunks
                ],
            }
        )

    return EvaluationResult(
        system_name=system_name,
        retrieval=aggregate_retrieval_metrics(hits, reciprocal_ranks, recalls),
        generation=aggregate_generation_metrics(
            groundedness_values,
            factual_flags,
            hallucination_values,
        ),
        num_questions=len(eval_questions),
        details=details,
    )


def save_evaluation_result(result: EvaluationResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "system_name": result.system_name,
        "num_questions": result.num_questions,
        "retrieval": {
            "hit_rate_at_k": result.retrieval.hit_rate_at_k,
            "mrr": result.retrieval.mrr,
            "recall": result.retrieval.recall,
        },
        "generation": {
            "groundedness": result.generation.groundedness,
            "factual_accuracy": result.generation.factual_accuracy,
            "hallucination_rate": result.generation.hallucination_rate,
        },
        "details": result.details,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compare_results(
    baseline: EvaluationResult,
    engineered: EvaluationResult,
) -> pd.DataFrame:
    """Build comparison table required by the PRD."""
    rows = [
        ("Hit Rate@5", baseline.retrieval.hit_rate_at_k, engineered.retrieval.hit_rate_at_k),
        ("MRR", baseline.retrieval.mrr, engineered.retrieval.mrr),
        ("Recall", baseline.retrieval.recall, engineered.retrieval.recall),
        ("Groundedness", baseline.generation.groundedness, engineered.generation.groundedness),
        (
            "Factual Accuracy",
            baseline.generation.factual_accuracy,
            engineered.generation.factual_accuracy,
        ),
        (
            "Hallucination Rate",
            baseline.generation.hallucination_rate,
            engineered.generation.hallucination_rate,
        ),
    ]

    records = []
    for metric, base_value, eng_value in rows:
        if base_value == 0:
            improvement = eng_value - base_value
        else:
            improvement = (eng_value - base_value) / abs(base_value)

        # Lower is better for hallucination rate
        if metric == "Hallucination Rate":
            improvement = -improvement

        records.append(
            {
                "Metric": metric,
                "Baseline": round(base_value, 4),
                "Engineered": round(eng_value, 4),
                "Improvement": round(improvement, 4),
            }
        )

    return pd.DataFrame(records)
