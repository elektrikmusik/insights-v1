"""
Mock FinBERT for testing without the full model weights.
"""
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class MockSentimentResult:
    text: str
    label: str
    score: float
    scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label, 
            "score": self.score, 
            "scores": self.scores
        }


class MockFinBERTAnalyzer:
    """Mock FinBERT with simple keyword-based sentiment."""
    
    def predict(self, texts: List[str], batch_size: int = 16) -> List[MockSentimentResult]:
        results = []
        for text in texts:
            text_lower = text.lower()
            if any(w in text_lower for w in ["risk", "loss", "decline", "threat", "liability"]):
                results.append(MockSentimentResult(
                    text=text[:50], 
                    label="negative", 
                    score=0.85,
                    scores={"positive": 0.05, "negative": 0.85, "neutral": 0.10}
                ))
            elif any(w in text_lower for w in ["growth", "opportunity", "profit", "benefit"]):
                results.append(MockSentimentResult(
                    text=text[:50], 
                    label="positive", 
                    score=0.80,
                    scores={"positive": 0.80, "negative": 0.05, "neutral": 0.15}
                ))
            else:
                results.append(MockSentimentResult(
                    text=text[:50], 
                    label="neutral", 
                    score=0.75,
                    scores={"positive": 0.10, "negative": 0.15, "neutral": 0.75}
                ))
        return results
