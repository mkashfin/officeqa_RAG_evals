"""LLM answer generation via Ollama."""

from __future__ import annotations

import logging

import ollama

from src.config import LLMConfig
from src.generation.guardrails import apply_guardrails
from src.models import RAGResponse, RetrievedChunk

logger = logging.getLogger(__name__)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a single context block."""
    blocks: list[str] = []
    for chunk in chunks:
        meta = chunk.metadata
        header = (
            f"[{meta.get('filename', 'unknown')} | "
            f"{meta.get('month', '?')} {meta.get('year', '?')} | "
            f"chunk {meta.get('chunk_id', '?')}]"
        )
        if meta.get("section"):
            header += f" Section: {meta['section']}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


class AnswerGenerator:
    def __init__(self, config: LLMConfig, *, require_grounding: bool = True):
        self.config = config
        self.require_grounding = require_grounding
        self.client = ollama.Client(host=config.base_url)

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        prompt_template: str,
        expanded_query: str | None = None,
    ) -> RAGResponse:
        context = format_context(chunks)
        prompt = prompt_template.format(context=context, question=question)

        try:
            response = self.client.generate(
                model=self.config.model,
                prompt=prompt,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            )
            raw_answer = response.get("response", "").strip()
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raw_answer = (
                "Unable to generate an answer. Ensure Ollama is running and the "
                f"model '{self.config.model}' is available."
            )

        answer = apply_guardrails(
            raw_answer,
            context=context,
            require_grounding=self.require_grounding,
            compact_verbose=True,
        )

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
            expanded_query=expanded_query,
        )
