"""
FinBERT Sentiment Analyzer - Core implementation using Hugging Face Transformers.
"""
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

# Conditional imports for optional dependencies
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Define dummy placeholders
    torch = None
    F = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    
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


class FinBERTAnalyzer:
    """
    Financial sentiment analysis using ProsusAI/finbert.
    """
    
    tokenizer: Any
    model: Any
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str | None = None):
        """
        Initialize analyzer and load model.
        
        Args:
            model_name: Hugging Face model ID
            device: 'cuda', 'mps', 'cpu', or None (auto)
        """
        self.tokenizer = None
        self.model = None

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers/torch not installed. "
                "Install with 'uv add transformers torch' or use FINBERT_MODE='http'."
            )
            
        self.model_name = model_name
        self._device = self._get_device(device)
        self._load_model()
        
    def _get_device(self, requested: str | None) -> str:
        if requested:
            return requested
        
        if torch and torch.cuda.is_available():
            return "cuda"
        if torch and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
        
    def _load_model(self):
        logger.info(f"Loading FinBERT model: {self.model_name} on {self._device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self._device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            raise
            
    def predict(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment for a batch of texts."""
        if not texts:
            return []
            
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("FinBERT model/tokenizer not initialized")

        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Apply softmax (dim=1)
            probs = F.softmax(outputs.logits, dim=1)
            
        results = []
        # Check model config for labels
        id2label = self.model.config.id2label
        
        cpu_probs = probs.cpu().numpy()
        
        for i, text in enumerate(texts):
            prob_dist = cpu_probs[i]
            # Get max
            pred_id = prob_dist.argmax()
            label = id2label[pred_id]
            score = float(prob_dist[pred_id])
            
            scores_dict = {
                id2label[j]: float(prob) 
                for j, prob in enumerate(prob_dist)
            }
            
            results.append(SentimentResult(
                text=text,
                label=label,
                score=score,
                scores=scores_dict
            ))
            
        return results
