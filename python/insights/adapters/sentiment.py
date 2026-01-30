"""
Sentiment Adapter - Facade for FinBERT analysis via multiple backends.

Supports three modes (configured via FINBERT_MODE env var):
- "local": Load FinBERT model in-process (requires torch, heavy)
- "http": Connect to FinBERT microservice
- "hf_inference": Use HuggingFace Inference API (lightweight, recommended)
"""
import json
import logging
from typing import List

from agno.tools import Toolkit

from insights.core.events import publish_tool_used
from insights.services.sentiment import get_sentiment_analyzer, SentimentResult

logger = logging.getLogger(__name__)

# Class names used to identify analyzer type for streamed events
_FINBERT_ANALYZER_NAMES = ("HuggingFaceInferenceAnalyzer", "FinBERTAnalyzer", "FinBERTClient")
_NOOP_ANALYZER_NAME = "_NoOpSentimentAnalyzer"


class SentimentToolkit(Toolkit):
    """
    Agno toolkit for financial sentiment analysis.
    
    Wraps the configured FinBERT analyzer based on FINBERT_MODE setting.
    Uses HuggingFace Inference API by default (no local model required).
    """
    
    def __init__(self):
        super().__init__(name="sentiment_toolkit")
        self.analyzer = get_sentiment_analyzer()
        self.register(self.analyze_sentiment)
        self.register(self.get_sentiment_scores)
        
        logger.info(f"Initialized SentimentToolkit with {type(self.analyzer).__name__}")

    def _tool_used_payload(self, num_texts: int) -> tuple[str, dict]:
        analyzer_name = type(self.analyzer).__name__
        if analyzer_name in _FINBERT_ANALYZER_NAMES:
            analyzer, msg = "finbert", f"Running FinBERT on {num_texts} text(s)"
        elif analyzer_name == _NOOP_ANALYZER_NAME:
            analyzer, msg = "noop", f"No-op sentiment on {num_texts} text(s) (set HF_TOKEN for FinBERT)"
        else:
            analyzer, msg = analyzer_name.lower(), f"Sentiment ({analyzer_name}) on {num_texts} text(s)"
        return msg, {"analyzer": analyzer, "num_texts": num_texts}

    async def analyze_sentiment(self, texts: List[str]) -> str:
        """
        Analyze the sentiment of financial text passages.
        
        Use this tool to understand the tone and sentiment of risk factors, 
        earnings reports, news articles, or any financial text.
        
        Args:
            texts: List of text strings to analyze (max 10 recommended per call)
            
        Returns:
            JSON string with sentiment results for each text, including:
            - label: "positive", "negative", or "neutral"
            - score: confidence score (0.0 to 1.0)
            - scores: breakdown of all sentiment probabilities
        """
        if not texts:
            return json.dumps({"error": "No texts provided"})
        msg, extra = self._tool_used_payload(len(texts))
        publish_tool_used("sentiment_toolkit", "analyze_sentiment", message=msg, **extra)
        try:
            results = await self.analyzer.predict(texts)
            return json.dumps([r.to_dict() for r in results], indent=2)
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return json.dumps({"error": str(e)})
    
    async def get_sentiment_scores(self, text: str) -> dict:
        """
        Get detailed sentiment scores for a single text.
        
        Use this for quick sentiment assessment of a specific passage.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with label, score, and full probability breakdown
        """
        msg, extra = self._tool_used_payload(1)
        publish_tool_used("sentiment_toolkit", "get_sentiment_scores", message=msg, **extra)
        try:
            results = await self.analyzer.predict([text])
            if results:
                return results[0].to_dict()
            return {"label": "neutral", "score": 0.0, "scores": {}}
        except Exception as e:
            logger.error(f"Sentiment scoring failed: {e}")
            return {"error": str(e)}


# Re-export for convenience
__all__ = ["SentimentToolkit", "get_sentiment_analyzer", "SentimentResult"]
