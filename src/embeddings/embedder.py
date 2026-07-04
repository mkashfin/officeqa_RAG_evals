"""Generate embeddings for document chunks and queries."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _resolve_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class Embedder:
    """Wrapper around SentenceTransformer for batch encoding."""

    def __init__(self, model_name: str, device: str | None = None):
        self.device = device or _resolve_device()
        logger.info("Loading embedding model: %s on %s", model_name, self.device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model_name = model_name

    def embed_documents(
        self,
        texts: Sequence[str],
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode all texts in one optimized pass."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        progress_bar = show_progress and len(texts) > 1
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents_batched(
        self,
        texts: Sequence[str],
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode texts with an explicit tqdm bar (fallback for older ST versions)."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        batches = range(0, len(texts), batch_size)
        iterator = batches
        if show_progress and len(texts) > batch_size:
            iterator = tqdm(
                batches,
                total=(len(texts) + batch_size - 1) // batch_size,
                desc="Embedding chunks",
                unit="batch",
            )

        chunks: list[np.ndarray] = []
        for start in iterator:
            end = min(start + batch_size, len(texts))
            batch_vectors = self.model.encode(
                list(texts[start:end]),
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            chunks.append(np.asarray(batch_vectors, dtype=np.float32))

        return np.vstack(chunks)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vector[0], dtype=np.float32)
