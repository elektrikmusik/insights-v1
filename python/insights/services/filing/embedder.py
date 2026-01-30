"""
Embedding Service - Generates vector embeddings for text chunks.
"""

from insights.adapters.models.embedding import GoogleEmbeddingAdapter
from insights.adapters.models.interface import BaseEmbeddingModel


class EmbeddingService:
    """Service for generating text embeddings used in vector search."""

    def __init__(self, adapter: BaseEmbeddingModel | None = None):
        """
        Initialize embedding service.

        Args:
            adapter: Embedding model adapter (defaults to GoogleEmbeddingAdapter)
        """
        self.adapter = adapter or GoogleEmbeddingAdapter()

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return await self.adapter.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Handles batch processing if necessary.
        """
        return await self.adapter.embed_batch(texts)
