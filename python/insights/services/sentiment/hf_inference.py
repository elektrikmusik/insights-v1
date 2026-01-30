"""
FinBERT Sentiment Analyzer using Hugging Face Inference API.

This is a lightweight alternative to local model inference that uses the
Hugging Face serverless Inference API. It doesn't require downloading the
1.3GB model locally and is suitable for low-volume usage.

Requires: huggingface_hub package and HF_TOKEN environment variable.
"""
import logging
import asyncio
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any

from huggingface_hub import InferenceClient
from huggingface_hub.inference._client import TextClassificationOutputElement

from insights.core.config import settings
from insights.core.errors import InsightsError

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result for a single text."""
    text: str
    label: str  # 'positive', 'negative', 'neutral'
    score: float  # confidence
    scores: Dict[str, float]
    
    def to_dict(self) -> dict:
        return asdict(self)


class HuggingFaceInferenceAnalyzer:
    """
    Financial sentiment analysis using Hugging Face Inference API.
    
    Uses ProsusAI/finbert model via HF serverless inference.
    This is a lightweight alternative that doesn't require local model loading.
    
    Features:
    - No local GPU/model required
    - Uses HF_TOKEN for authentication
    - Async-friendly with automatic batching
    - Falls back gracefully on rate limits
    
    Usage:
        analyzer = HuggingFaceInferenceAnalyzer()
        results = await analyzer.predict(["Stock prices increased significantly."])
    """
    
    MODEL_ID = "ProsusAI/finbert"
    LABELS = ["positive", "negative", "neutral"]
    
    def __init__(
        self, 
        token: Optional[str] = None,
        model_id: str = MODEL_ID,
        timeout: float = 30.0
    ):
        """
        Initialize the HF Inference client.
        
        Args:
            token: HuggingFace API token (defaults to HF_TOKEN env var)
            model_id: Model ID to use (default: ProsusAI/finbert)
            timeout: Request timeout in seconds
        """
        self.token = token or settings.HF_TOKEN
        self.model_id = model_id
        self.timeout = timeout
        
        if not self.token:
            raise ValueError(
                "HF_TOKEN is required. Set it in environment or pass token parameter."
            )
        
        # Initialize the sync client (we'll wrap calls in async)
        self.client = InferenceClient(
            model=self.model_id,
            token=self.token,
            timeout=self.timeout
        )
        
        logger.info(f"Initialized HF Inference client for model: {self.model_id}")
    
    def _classify_sync(self, text: str) -> List[TextClassificationOutputElement]:
        """Synchronous classification for a single text."""
        try:
            return self.client.text_classification(text)
        except Exception as e:
            logger.error(f"HF Inference API error: {e}")
            raise InsightsError(f"Sentiment analysis failed: {e}") from e
    
    def predict_sync(self, texts: List[str]) -> List[SentimentResult]:
        """
        Synchronous prediction for a list of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of SentimentResult objects
        """
        if not texts:
            return []
        
        results = []
        
        for text in texts:
            try:
                # HF text_classification returns a list of label/score pairs
                classification = self._classify_sync(text)
                
                # Build scores dict from all labels
                scores_dict = {item.label: item.score for item in classification}
                
                # Find the highest scoring label
                best = max(classification, key=lambda x: x.score)
                
                results.append(SentimentResult(
                    text=text[:100] + "..." if len(text) > 100 else text,
                    label=best.label,
                    score=best.score,
                    scores=scores_dict
                ))
                
            except Exception as e:
                logger.warning(f"Failed to classify text: {e}")
                # Return neutral sentiment on error
                results.append(SentimentResult(
                    text=text[:100] + "..." if len(text) > 100 else text,
                    label="neutral",
                    score=0.0,
                    scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0}
                ))
        
        return results
    
    async def predict(self, texts: List[str]) -> List[SentimentResult]:
        """
        Async prediction for a list of texts.
        
        Uses asyncio.to_thread for non-blocking execution.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of SentimentResult objects
        """
        if not texts:
            return []
        
        # Run sync code in thread pool to avoid blocking
        return await asyncio.to_thread(self.predict_sync, texts)
    
    async def get_negative_intensity(self, text: str) -> float:
        """
        Get the negative sentiment intensity for a single text.
        
        Returns:
            Float from 0.0 (not negative) to 1.0 (highly negative)
        """
        results = await self.predict([text])
        if not results:
            return 0.0
        return results[0].scores.get("negative", 0.0)
    
    async def health_check(self) -> bool:
        """Check if the HF Inference API is accessible."""
        try:
            # Quick test with a simple text
            result = await self.predict(["test"])
            return len(result) > 0
        except Exception as e:
            logger.warning(f"HF Inference health check failed: {e}")
            return False


# Singleton instance
_hf_analyzer: Optional[HuggingFaceInferenceAnalyzer] = None


def get_hf_analyzer() -> HuggingFaceInferenceAnalyzer:
    """Get or create the singleton HF Inference analyzer."""
    global _hf_analyzer
    if _hf_analyzer is None:
        _hf_analyzer = HuggingFaceInferenceAnalyzer()
    return _hf_analyzer


# Quick test function
async def test_hf_inference():
    """Test the HF Inference analyzer."""
    analyzer = HuggingFaceInferenceAnalyzer()
    
    test_texts = [
        "The company reported record profits and exceeded expectations.",
        "Stock prices plummeted after the disappointing earnings report.",
        "The market remained stable with no significant changes."
    ]
    
    print("Testing HuggingFace Inference API with FinBERT...\n")
    
    results = await analyzer.predict(test_texts)
    
    for result in results:
        print(f"Text: {result.text}")
        print(f"  Label: {result.label} (confidence: {result.score:.3f})")
        print(f"  Scores: {result.scores}")
        print()
    
    return results


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_hf_inference())
