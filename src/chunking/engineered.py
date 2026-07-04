"""Engineered heading-aware recursive chunking."""

from __future__ import annotations

import re

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import PipelineConfig
from src.models import DocumentChunk, TreasuryDocument

HEADING_SPLIT_PATTERN = re.compile(r"(?=^#{1,3}\s+)", re.MULTILINE)


def _get_encoder(tokenizer_name: str):
    if tokenizer_name.startswith("tiktoken:"):
        encoding_name = tokenizer_name.split(":", 1)[1]
        return tiktoken.get_encoding(encoding_name)
    return tiktoken.get_encoding("cl100k_base")


def _split_by_headings(content: str) -> list[tuple[str, str]]:
    """Split markdown into (section_title, section_text) pairs."""
    parts = HEADING_SPLIT_PATTERN.split(content)
    sections: list[tuple[str, str]] = []

    if parts and parts[0].strip():
        sections.append(("", parts[0].strip()))

    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        title = lines[0].lstrip("#").strip() if lines else ""
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        sections.append((title, body or part))

    if not sections:
        sections.append(("", content))
    return sections


def chunk_document(
    document: TreasuryDocument,
    config: PipelineConfig,
    start_chunk_id: int = 0,
) -> list[DocumentChunk]:
    """
    Heading-aware recursive chunking.

    1. Split document by markdown headings.
    2. Apply recursive character splitting within each section.
    3. Preserve section title in chunk metadata.
    """
    encoder = _get_encoder(config.tokenizer)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=lambda text: len(encoder.encode(text)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[DocumentChunk] = []
    chunk_id = start_chunk_id

    for section_title, section_text in _split_by_headings(document.content):
        if not section_text.strip():
            continue

        section_meta = document.metadata
        if section_title:
            section_meta = document.metadata.__class__(
                year=document.metadata.year,
                month=document.metadata.month,
                filename=document.metadata.filename,
                section=section_title,
                page_number=document.metadata.page_number,
                period_code=document.metadata.period_code,
            )

        for text in splitter.split_text(section_text):
            cleaned = text.strip()
            if not cleaned:
                continue
            chunks.append(
                DocumentChunk(
                    text=cleaned,
                    metadata=section_meta,
                    chunk_id=chunk_id,
                )
            )
            chunk_id += 1

    return chunks


def chunk_documents(
    documents: list[TreasuryDocument],
    config: PipelineConfig,
) -> list[DocumentChunk]:
    """Chunk all documents with heading-aware strategy."""
    all_chunks: list[DocumentChunk] = []
    next_id = 0
    for document in documents:
        doc_chunks = chunk_document(document, config, start_chunk_id=next_id)
        all_chunks.extend(doc_chunks)
        next_id += len(doc_chunks)
    return all_chunks
