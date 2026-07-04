"""LLM generation augmented with external tool outputs."""

from __future__ import annotations

import logging

from src.config import Settings
from src.generation.generator import AnswerGenerator, format_context
from src.generation.guardrails import apply_guardrails
from src.models import RAGResponse, RetrievedChunk
from src.tools.orchestrator import ToolOrchestrator

logger = logging.getLogger(__name__)


class ToolAugmentedGenerator(AnswerGenerator):
    """Run external tools before LLM generation; use tool answer when confident."""

    def __init__(
        self,
        settings: Settings,
        *,
        require_grounding: bool = True,
        allow_direct_tool_answer: bool = True,
        tool_mode: str = "full",
    ):
        super().__init__(settings.llm, require_grounding=require_grounding)
        self.settings = settings
        self.allow_direct_tool_answer = allow_direct_tool_answer
        self.tool_mode = tool_mode
        self.orchestrator = ToolOrchestrator(settings)

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        prompt_template: str,
        expanded_query: str | None = None,
        use_tools: bool = True,
    ) -> RAGResponse:
        context = format_context(chunks)
        tool_summary = ""
        tool_trace: list[str] = []

        if use_tools:
            tool_result = self.orchestrator.run(
                question,
                context,
                tool_mode=self.tool_mode,
            )
            tool_summary = tool_result.summary
            tool_trace = tool_result.tool_trace

            if self.allow_direct_tool_answer and tool_result.direct_answer is not None:
                logger.info(
                    "Using direct tool answer for question: %s",
                    question[:80],
                )
                raw_answer = f"<FINAL_ANSWER>{tool_result.direct_answer}</FINAL_ANSWER>"
                answer = apply_guardrails(
                    raw_answer,
                    context=context,
                    tool_outputs=tool_summary,
                    require_grounding=self.require_grounding,
                )
                return RAGResponse(
                    question=question,
                    answer=answer,
                    retrieved_chunks=chunks,
                    expanded_query=expanded_query,
                    tool_outputs=tool_summary,
                )

        if "{tool_outputs}" in prompt_template:
            prompt_body = prompt_template.format(
                context=context,
                question=question,
                tool_outputs=tool_summary or "No external tool outputs.",
            )
        else:
            augmented_context = context
            if tool_summary:
                augmented_context += f"\n\n--- EXTERNAL TOOL OUTPUTS ---\n{tool_summary}"
            prompt_body = prompt_template.format(context=augmented_context, question=question)

        try:
            response = self.client.generate(
                model=self.config.model,
                prompt=prompt_body,
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
            tool_outputs=tool_summary,
            require_grounding=self.require_grounding,
        )

        if tool_trace:
            logger.debug("Tool trace: %s", "; ".join(tool_trace))

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
            expanded_query=expanded_query,
            tool_outputs=tool_summary,
        )
