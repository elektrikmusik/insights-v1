"""
Embedding adapter implementation using Google Gemini API.
"""
import logging
from typing import cast

import httpx

from insights.adapters.models.interface import BaseEmbeddingModel
from insights.core.config import settings
from insights.core.errors import LLMProviderError
from insights.core.retry import retry_llm_call

logger = logging.getLogger(__name__)


class GoogleEmbeddingAdapter(BaseEmbeddingModel):
    """
    Adapter for Google Gemini Embeddings (via REST API).
    """

    def __init__(self, model_id: str | None = None):
        self._model_id = model_id or settings.EMBEDDING_MODEL
        self._api_key = settings.GOOGLE_API_KEY
        self._base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def dimension(self) -> int:
        return 768  # text-embedding-004 is 768

    @retry_llm_call
    async def embed(self, text: str) -> list[float]:
        """Embed a single string."""
        if not text.strip():
            return [0.0] * self.dimension

        url = f"{self._base_url}/{self._model_id}:embedContent"
        params = {"key": self._api_key}
        payload = {
            "model": f"models/{self._model_id}",
            "content": {"parts": [{"text": text}]}
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, params=params, json=payload, timeout=10.0)

                if response.status_code != 200:
                    raise LLMProviderError(
                        "google",
                        f"Embedding failed: {response.text}",
                        status_code=response.status_code
                    )

                data = response.json()
                if "embedding" not in data:
                    raise LLMProviderError("google", f"No embedding returned: {data}")

                return cast(list[float], data["embedding"]["values"])

        except Exception as e:
            logger.error(f"Google embedding failed: {e}")
            raise LLMProviderError("google", f"Google embedding failed: {e}") from e

    @retry_llm_call
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings (sequentially for now)."""
        # Google Batch Embed API exists but for simplicity/safety with rate limits
        # we can do sequential or simple gather.
        # For now, simplistic loop (retried individually).
        results = []
        for text in texts:
            results.append(await self.embed(text))
        return results
