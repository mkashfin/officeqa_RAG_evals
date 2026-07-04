"""Baseline retriever: pure vector similarity search without metadata filtering."""

from __future__ import annotations

from src.models import RetrievedChunk
from src.vectorstore.chroma_store import ChromaVectorStore


class BaselineRetriever:
    """MiniLM embeddings + vector search only. No metadata filters (by design)."""

    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        year: int | list[int] | None = None,
        month: str | None = None,
        period_code: str | None = None,
    ) -> list[RetrievedChunk]:
        del year, month, period_code
        k = top_k or self.vector_store.config.top_k
        return self.vector_store.similarity_search(
            query=query,
            top_k=k,
            apply_metadata_filter=False,
        )
