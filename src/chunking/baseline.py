"""Baseline fixed-size token chunking."""

from __future__ import annotations

import tiktoken

from src.config import PipelineConfig
from src.models import DocumentChunk, TreasuryDocument


def _get_encoder(tokenizer_name: str):
    if tokenizer_name.startswith("tiktoken:"):
        encoding_name = tokenizer_name.split(":", 1)[1]
        return tiktoken.get_encoding(encoding_name)
    return tiktoken.get_encoding("cl100k_base")


def chunk_document(
    document: TreasuryDocument,
    config: PipelineConfig,
    start_chunk_id: int = 0,
) -> list[DocumentChunk]:
    """
    Split a document into fixed-size token chunks with overlap.

    Strategy: fixed window (baseline per PRD).
    """
    encoder = _get_encoder(config.tokenizer)
    tokens = encoder.encode(document.content)

    chunks: list[DocumentChunk] = []
    chunk_id = start_chunk_id
    step = max(config.chunk_size - config.chunk_overlap, 1)

    for start in range(0, len(tokens), step):
        end = min(start + config.chunk_size, len(tokens))
        token_slice = tokens[start:end]
        if not token_slice:
            continue

        text = encoder.decode(token_slice).strip()
        if not text:
            continue

        chunks.append(
            DocumentChunk(
                text=text,
                metadata=document.metadata,
                chunk_id=chunk_id,
            )
        )
        chunk_id += 1

        if end >= len(tokens):
            break

    return chunks


def chunk_documents(
    documents: list[TreasuryDocument],
    config: PipelineConfig,
) -> list[DocumentChunk]:
    """Chunk all documents sequentially."""
    all_chunks: list[DocumentChunk] = []
    next_id = 0
    for document in documents:
        doc_chunks = chunk_document(document, config, start_chunk_id=next_id)
        all_chunks.extend(doc_chunks)
        next_id += len(doc_chunks)
    return all_chunks
