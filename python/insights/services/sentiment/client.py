"""
FinBERT HTTP Client for remote sentiment analysis.
"""
import logging

import httpx

from insights.core.config import settings
from insights.core.errors import InsightsError
from insights.services.sentiment.finbert_analyzer import SentimentResult

logger = logging.getLogger(__name__)


class FinBERTClient:
    """Client for remote FinBERT microservice."""

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.FINBERT_SERVICE_URL

    async def predict(self, texts: list[str]) -> list[SentimentResult]:
        """
        Send texts to FinBERT service for analysis.

        Args:
            texts: List of strings to analyze

        Returns:
            List of SentimentResult objects
        """
        if not texts:
            return []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/analyze",
                    json={"texts": texts}
                )
                response.raise_for_status()

                data = response.json()
                # data should be list of dicts matching SentimentResult
                return [SentimentResult(**item) for item in data]

        except httpx.HTTPError as e:
            logger.error(f"FinBERT service failed: {e}")
            raise InsightsError(f"Sentiment analysis failed: {e}") from e
