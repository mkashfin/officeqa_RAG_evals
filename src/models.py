"""Shared data models for the Treasury RAG system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentMetadata:
    year: int
    month: str
    filename: str
    section: str = ""
    page_number: int | None = None
    period_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "year": int(self.year),
            "month": self.month,
            "filename": self.filename,
        }
        if self.section:
            payload["section"] = self.section
        if self.page_number is not None:
            payload["page_number"] = int(self.page_number)
        if self.period_code:
            payload["period_code"] = self.period_code
        return payload


@dataclass
class TreasuryDocument:
    content: str
    metadata: DocumentMetadata


@dataclass
class DocumentChunk:
    text: str
    metadata: DocumentMetadata
    chunk_id: int

    def to_dict(self) -> dict[str, Any]:
        payload = self.metadata.to_dict()
        payload["chunk_id"] = self.chunk_id
        return payload


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float
    rank: int = 0


@dataclass
class RAGResponse:
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    expanded_query: str | None = None
    tool_outputs: str = ""


@dataclass
class EvalQuestion:
    question: str
    answer: str
    reference_documents: list[str] = field(default_factory=list)
    year: int | None = None
    month: str | None = None
    period_code: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalMetrics:
    hit_rate_at_k: float
    mrr: float
    recall: float


@dataclass
class GenerationMetrics:
    groundedness: float
    factual_accuracy: float
    hallucination_rate: float


@dataclass
class EvaluationResult:
    system_name: str
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    num_questions: int
    details: list[dict[str, Any]] = field(default_factory=list)
