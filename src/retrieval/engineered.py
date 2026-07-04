"""Engineered retriever: hybrid search, query expansion, reranking, compression."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import PipelineConfig
from src.embeddings.embedder import Embedder
from src.metadata.query_parser import TemporalFilters
from src.models import RetrievedChunk
from src.retrieval.filters import resolve_temporal_filters
from src.vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class EngineeredRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        config: PipelineConfig,
        embedder: Embedder,
    ):
        self.vector_store = vector_store
        self.config = config
        self.embedder = embedder
        self._bm25: BM25Okapi | None = None
        self._bm25_texts: list[str] = []
        self._bm25_metadatas: list[dict] = []

    def _ensure_bm25_index(self) -> None:
        if self._bm25 is not None:
            return

        documents, metadatas = self.vector_store.get_all_documents()
        if not documents:
            self._bm25 = BM25Okapi([[]])
            return

        tokenized = [_tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_texts = documents
        self._bm25_metadatas = metadatas
        logger.info("Built BM25 index over %d documents.", len(documents))

    def expand_query(self, query: str) -> str:
        """Expand query with domain terms to improve recall."""
        if not self.config.query_expansion:
            return query

        keywords = [
            "U.S. Treasury Bulletin",
            "Table FFO",
            "receipts outlays deficit debt",
            "Exchange Stabilization Fund ESF",
            "Foreign exchange and securities",
        ]
        lowered = query.lower()
        selected = [term for term in keywords if any(token in lowered for token in term.lower().split())]
        if not selected:
            selected = keywords[:2]
        return f"{query}\nFocus on: {'; '.join(selected)}"

    def _metadata_matches(
        self,
        metadata: dict,
        filters: TemporalFilters,
    ) -> bool:
        if filters.year is not None:
            doc_year = int(metadata.get("year", -1))
            if doc_year != int(filters.year):
                return False

        if filters.period_code is not None:
            if metadata.get("period_code") != filters.period_code:
                return False
        elif filters.month is not None and metadata.get("month") != filters.month:
            return False
        return True

    def _hybrid_search(
        self,
        query: str,
        candidate_k: int,
        filters: TemporalFilters,
    ) -> list[RetrievedChunk]:
        vector_hits = self.vector_store.similarity_search(
            query=query,
            top_k=candidate_k,
            year=filters.year,
            month=filters.month,
            period_code=filters.period_code,
            apply_metadata_filter=True,
        )

        if not vector_hits and filters.year is not None and filters.period_code is not None:
            logger.info("No hits for year=%s period=%s; retrying year-only filter.", filters.year, filters.period_code)
            vector_hits = self.vector_store.similarity_search(
                query=query,
                top_k=candidate_k,
                year=filters.year,
                apply_metadata_filter=True,
            )

        if not vector_hits and filters.year is not None:
            logger.info("No hits for year=%s; retrying without metadata filter.", filters.year)
            vector_hits = self.vector_store.similarity_search(
                query=query,
                top_k=candidate_k,
                apply_metadata_filter=False,
            )

        if not self.config.hybrid_search:
            return vector_hits

        self._ensure_bm25_index()
        if not self._bm25_texts:
            return vector_hits

        bm25_scores = self._bm25.get_scores(_tokenize(query))
        max_bm25 = float(max(bm25_scores)) if len(bm25_scores) else 1.0
        if max_bm25 <= 0:
            max_bm25 = 1.0

        combined: dict[str, RetrievedChunk] = {}

        for hit in vector_hits:
            key = f"{hit.metadata.get('filename')}::{hit.metadata.get('chunk_id')}"
            combined[key] = RetrievedChunk(
                text=hit.text,
                metadata=hit.metadata,
                score=self.config.vector_weight * hit.score,
                rank=0,
            )

        top_bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]
        for idx in top_bm25_indices:
            metadata = self._bm25_metadatas[idx]
            if filters.year is not None or filters.period_code is not None or filters.month is not None:
                if not self._metadata_matches(metadata, filters):
                    continue

            key = f"{metadata.get('filename')}::{metadata.get('chunk_id')}"
            normalized_bm25 = float(bm25_scores[idx]) / max_bm25
            text = self._bm25_texts[idx]

            if key in combined:
                combined[key].score += self.config.bm25_weight * normalized_bm25
            else:
                combined[key] = RetrievedChunk(
                    text=text,
                    metadata=metadata,
                    score=self.config.bm25_weight * normalized_bm25,
                    rank=0,
                )

        ranked = sorted(combined.values(), key=lambda item: item.score, reverse=True)
        for rank, item in enumerate(ranked[:candidate_k], start=1):
            item.rank = rank
        return ranked[:candidate_k]

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not self.config.reranking or not candidates:
            return candidates[:top_k]

        try:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(self.config.reranker_model)
            pairs = [[query, chunk.text] for chunk in candidates]
            scores = reranker.predict(pairs)

            reranked = sorted(
                zip(candidates, scores),
                key=lambda pair: float(pair[1]),
                reverse=True,
            )
            output: list[RetrievedChunk] = []
            for rank, (chunk, score) in enumerate(reranked[:top_k], start=1):
                output.append(
                    RetrievedChunk(
                        text=chunk.text,
                        metadata=chunk.metadata,
                        score=float(score),
                        rank=rank,
                    )
                )
            return output
        except Exception as exc:
            logger.warning("Reranking failed, using hybrid scores: %s", exc)
            return candidates[:top_k]

    def _boost_expected_document(
        self,
        candidates: list[RetrievedChunk],
        filters: TemporalFilters,
    ) -> list[RetrievedChunk]:
        if not filters.expected_filename:
            return candidates

        expected = filters.expected_filename.lower()
        boosted: list[RetrievedChunk] = []
        for chunk in candidates:
            filename = str(chunk.metadata.get("filename", "")).lower()
            score = chunk.score
            if filename == expected:
                score += 1.0
            boosted.append(
                RetrievedChunk(
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=score,
                    rank=chunk.rank,
                )
            )
        return sorted(boosted, key=lambda item: item.score, reverse=True)

    def _compress_context(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove near-duplicate chunks from the same file/section."""
        if not self.config.context_compression:
            return chunks

        seen: set[str] = set()
        compressed: list[RetrievedChunk] = []
        for chunk in chunks:
            fingerprint = f"{chunk.metadata.get('filename')}::{chunk.metadata.get('section', '')}"
            text_key = chunk.text[:200]
            key = f"{fingerprint}::{text_key}"
            if key in seen:
                continue
            seen.add(key)
            compressed.append(chunk)
        return compressed

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        year: int | list[int] | None = None,
        month: str | None = None,
        period_code: str | None = None,
    ) -> tuple[list[RetrievedChunk], str]:
        expanded = self.expand_query(query)
        k = top_k or self.config.top_k
        candidate_k = max(self.config.retrieve_candidates, k)
        filters = resolve_temporal_filters(
            query,
            year=year,
            month=month,
            period_code=period_code,
        )

        if filters.expected_filename:
            logger.info("Targeting bulletin: %s", filters.expected_filename)

        candidates = self._hybrid_search(
            query=expanded,
            candidate_k=candidate_k,
            filters=filters,
        )
        candidates = self._boost_expected_document(candidates, filters)
        reranked = self._rerank(query=query, candidates=candidates, top_k=k)
        compressed = self._compress_context(reranked)

        for rank, chunk in enumerate(compressed, start=1):
            chunk.rank = rank

        return compressed, expanded
