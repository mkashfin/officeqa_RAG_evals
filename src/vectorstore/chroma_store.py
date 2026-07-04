"""ChromaDB vector store with metadata filtering support."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from tqdm import tqdm

from src.config import PipelineConfig, Settings
from src.embeddings.embedder import Embedder
from src.models import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Persist and query Treasury report chunk embeddings."""

    def __init__(
        self,
        settings: Settings,
        pipeline_config: PipelineConfig,
        embedder: Embedder,
        persist_subdir: str,
    ):
        self.settings = settings
        self.config = pipeline_config
        self.embedder = embedder
        persist_path = settings.data.indices_dir / persist_subdir
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=pipeline_config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        name = self.config.collection_name
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embed_batch_size: int | None = None,
        chroma_batch_size: int | None = None,
    ) -> None:
        """Embed all chunks in one pass, then store in ChromaDB with progress."""
        if not chunks:
            logger.warning("No chunks to index.")
            return

        embed_bs = embed_batch_size or self.config.embed_batch_size
        chroma_bs = chroma_batch_size or self.config.chroma_batch_size
        total = len(chunks)

        logger.info(
            "Indexing %d chunks into %s (embed_batch=%d, store_batch=%d, device=%s)",
            total,
            self.config.collection_name,
            embed_bs,
            chroma_bs,
            self.embedder.device,
        )

        texts = [chunk.text for chunk in chunks]
        ids = [f"{chunk.metadata.filename}::{chunk.chunk_id}" for chunk in chunks]
        metadatas = [chunk.to_dict() for chunk in chunks]

        logger.info("Generating embeddings for %d chunks...", total)
        embeddings = self.embedder.embed_documents(
            texts,
            batch_size=embed_bs,
            show_progress=True,
        )

        logger.info("Writing embeddings to ChromaDB...")
        for start in tqdm(
            range(0, total, chroma_bs),
            desc="Storing in ChromaDB",
            unit="batch",
        ):
            end = min(start + chroma_bs, total)
            self.collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=metadatas[start:end],
            )

        logger.info("Indexed %d chunks total.", self.count)

    def _build_where_clause(
        self,
        year: int | list[int] | None = None,
        month: str | None = None,
        period_code: str | None = None,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []

        if year is not None:
            if isinstance(year, list):
                clauses.append({"year": {"$in": [int(y) for y in year]}})
            else:
                clauses.append({"year": int(year)})

        if period_code is not None:
            clauses.append({"period_code": period_code})
        elif month is not None:
            clauses.append({"month": month})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def similarity_search(
        self,
        query: str,
        top_k: int | None = None,
        year: int | list[int] | None = None,
        month: str | None = None,
        period_code: str | None = None,
        apply_metadata_filter: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Vector similarity search with optional metadata filters."""
        k = top_k or self.config.top_k
        use_filter = (
            self.config.metadata_filtering
            if apply_metadata_filter is None
            else apply_metadata_filter
        )

        where = (
            self._build_where_clause(year, month, period_code)
            if use_filter and (year is not None or month is not None or period_code is not None)
            else None
        )
        query_embedding = self.embedder.embed_query(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for rank, (text, metadata, distance) in enumerate(
            zip(documents, metadatas, distances), start=1
        ):
            score = 1.0 - float(distance)
            retrieved.append(
                RetrievedChunk(
                    text=text or "",
                    metadata=metadata or {},
                    score=score,
                    rank=rank,
                )
            )

        return retrieved

    def get_all_documents(self) -> tuple[list[str], list[dict[str, Any]]]:
        """Return all indexed documents for BM25 indexing."""
        result = self.collection.get(include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return documents, metadatas
