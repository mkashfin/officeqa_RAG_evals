"""Engineered RAG pipeline with optimized retrieval and generation."""

from __future__ import annotations

import logging

from src.chunking import engineered as engineered_chunking
from src.config import Settings
from src.embeddings.embedder import Embedder
from src.generation.tool_augmented_generator import ToolAugmentedGenerator
from src.ingestion.loader import load_documents
from src.models import RAGResponse
from src.retrieval.engineered import EngineeredRetriever
from src.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class EngineeredPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.config = settings.engineered
        self.embedder = Embedder(self.config.embedding_model)
        self.vector_store = ChromaVectorStore(
            settings=settings,
            pipeline_config=self.config,
            embedder=self.embedder,
            persist_subdir="engineered",
        )
        self.retriever = EngineeredRetriever(
            vector_store=self.vector_store,
            config=self.config,
            embedder=self.embedder,
        )
        self.generator = ToolAugmentedGenerator(settings)

    def build_index(self, reset: bool = False) -> int:
        if reset:
            self.vector_store.reset()

        if self.vector_store.count > 0 and not reset:
            logger.info("Engineered index already contains %d chunks.", self.vector_store.count)
            return self.vector_store.count

        documents = load_documents(self.settings)
        if not documents:
            raise RuntimeError(
                "No Treasury reports found. Run `python scripts/download_data.py` first."
            )

        chunks = engineered_chunking.chunk_documents(documents, self.config)
        self.vector_store.add_chunks(chunks)
        count = self.vector_store.count
        logger.info("Engineered index built with %d chunks from %d documents.", count, len(documents))
        return count

    def ask(
        self,
        question: str,
        year: int | list[int] | None = None,
        month: str | None = None,
        period_code: str | None = None,
    ) -> RAGResponse:
        chunks, expanded_query = self.retriever.retrieve(
            query=question,
            top_k=self.config.top_k,
            year=year,
            month=month,
            period_code=period_code,
        )
        return self.generator.generate(
            question=question,
            chunks=chunks,
            prompt_template=self.settings.prompts.engineered,
            expanded_query=expanded_query,
        )
