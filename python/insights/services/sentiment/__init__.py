"""
Sentiment Analysis Services for InSights-ai.

This module provides multiple implementations for financial sentiment analysis:
- FinBERTAnalyzer: Local model inference using transformers (requires torch)
- FinBERTClient: HTTP client for remote FinBERT microservice
- HuggingFaceInferenceAnalyzer: HuggingFace Inference API (lightweight, no local model)

The `get_sentiment_analyzer()` factory function returns the appropriate
implementation based on the FINBERT_MODE configuration.
"""
from typing import Protocol, List, TYPE_CHECKING

from insights.services.sentiment.finbert_analyzer import SentimentResult

if TYPE_CHECKING:
    from insights.services.sentiment.finbert_analyzer import FinBERTAnalyzer
    from insights.services.sentiment.client import FinBERTClient
    from insights.services.sentiment.hf_inference import HuggingFaceInferenceAnalyzer


class SentimentAnalyzerProtocol(Protocol):
    """Protocol for sentiment analyzers."""
    
    async def predict(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment for a batch of texts."""
        ...


class _NoOpSentimentAnalyzer:
    """No-op sentiment analyzer when HF_TOKEN is unset. Returns neutral for all texts."""

    async def predict(self, texts: List[str]) -> List[SentimentResult]:
        if not texts:
            return []
        return [
            SentimentResult(
                text=text[:100] + "..." if len(text) > 100 else text,
                label="neutral",
                score=0.0,
                scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            )
            for text in texts
        ]


def get_sentiment_analyzer() -> SentimentAnalyzerProtocol:
    """
    Factory function to get the appropriate sentiment analyzer.
    
    Based on FINBERT_MODE setting:
    - "local": Uses FinBERTAnalyzer (loads model locally, requires torch)
    - "http": Uses FinBERTClient (connects to remote FinBERT service)
    - "hf_inference": Uses HuggingFaceInferenceAnalyzer (HF Inference API)
    
    Returns:
        An instance implementing SentimentAnalyzerProtocol
    """
    from insights.core.config import settings
    
    mode = settings.FINBERT_MODE.lower()
    
    if mode == "local":
        from insights.services.sentiment.finbert_analyzer import FinBERTAnalyzer
        # Wrap sync predict in async
        analyzer = FinBERTAnalyzer()
        
        # Create async wrapper
        class AsyncWrapper:
            def __init__(self, sync_analyzer: FinBERTAnalyzer):
                self._analyzer = sync_analyzer
            
            async def predict(self, texts: List[str]) -> List[SentimentResult]:
                import asyncio
                return await asyncio.to_thread(self._analyzer.predict, texts)
        
        return AsyncWrapper(analyzer)
    
    elif mode == "http":
        from insights.services.sentiment.client import FinBERTClient
        return FinBERTClient()
    
    elif mode == "hf_inference":
        from insights.services.sentiment.hf_inference import HuggingFaceInferenceAnalyzer
        if not getattr(settings, "HF_TOKEN", None) or not settings.HF_TOKEN.strip():
            logger = __import__("logging").getLogger(__name__)
            logger.warning(
                "HF_TOKEN is not set; using no-op sentiment (neutral for all texts). "
                "Set HF_TOKEN for real FinBERT sentiment via HuggingFace Inference API."
            )
            return _NoOpSentimentAnalyzer()
        return HuggingFaceInferenceAnalyzer()
    
    else:
        raise ValueError(
            f"Invalid FINBERT_MODE: {mode}. "
            "Must be 'local', 'http', or 'hf_inference'"
        )


# Direct imports for backwards compatibility
from insights.services.sentiment.client import FinBERTClient
from insights.services.sentiment.finbert_analyzer import FinBERTAnalyzer

__all__ = [
    "FinBERTAnalyzer",
    "FinBERTClient", 
    "SentimentResult",
    "get_sentiment_analyzer",
    "SentimentAnalyzerProtocol",
]
