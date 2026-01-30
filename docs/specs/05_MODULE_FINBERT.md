# 05. Module: FinBERT Service

## Overview

FinBERT provides specialized financial sentiment analysis using the `ProsusAI/finbert` model. This module handles:
- Sentiment classification (positive, negative, neutral)
- Confidence scoring
- Batch processing for efficiency
- Hybrid deployment (in-process for dev, microservice for prod)

---

## Architecture

### Development Mode (In-Process)

```
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI Backend Process                                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  SentimentToolkit (Agno)                                  │ │
│  │  └── FinBERTService (Singleton)                          │ │
│  │      └── Hugging Face Transformers                        │ │
│  │          └── ProsusAI/finbert (loaded once)              │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Production Mode (Microservice)

```
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI Backend (Cloud Run)                                    │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  SentimentToolkit (Agno)                                  │ │
│  │  └── FinBERTClient (HTTP)                                │ │
│  └───────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/gRPC
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  FinBERT Service (GPU Instance/TorchServe)                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  FastAPI                                                  │ │
│  │  └── FinBERTAnalyzer                                     │ │
│  │      └── PyTorch + Transformers                          │ │
│  │          └── CUDA/MPS acceleration                       │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## FinBERT Analyzer (Core Logic)

### `insights/services/sentiment/finbert_analyzer.py`

```python
"""
FinBERT Sentiment Analyzer - Core implementation.
This is the actual model loading and inference code.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result for a single text."""
    text: str
    label: str  # 'positive', 'negative', 'neutral'
    score: float  # 0.0 to 1.0 (confidence for the predicted label)
    scores: Dict[str, float]  # All class probabilities
    
    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": self.score,
            "scores": self.scores
        }


class FinBERTAnalyzer:
    """
    Financial sentiment analysis using ProsusAI/finbert.
    
    Implemented as a singleton to avoid reloading the model on every call.
    The model is ~1.3GB and takes several seconds to load.
    
    Usage:
        analyzer = FinBERTAnalyzer()
        results = analyzer.predict(["The company reported strong earnings."])
    """
    
    _instance: Optional["FinBERTAnalyzer"] = None
    _initialized: bool = False
    
    # Model configuration
    MODEL_NAME = "ProsusAI/finbert"
    MAX_LENGTH = 512  # BERT context window
    LABELS = ["positive", "negative", "neutral"]
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model and tokenizer (only once)."""
        if FinBERTAnalyzer._initialized:
            return
        
        logger.info(f"Loading FinBERT model: {self.MODEL_NAME}")
        
        # Detect best available device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple MPS")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU (consider GPU for production)")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        ).to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        
        FinBERTAnalyzer._initialized = True
        logger.info("FinBERT model loaded successfully")
    
    def predict(
        self, 
        texts: List[str],
        batch_size: int = 32
    ) -> List[SentimentResult]:
        """
        Predict sentiment for a list of texts.
        
        Args:
            texts: List of text strings to analyze
            batch_size: Batch size for inference (default: 32)
            
        Returns:
            List of SentimentResult objects
        """
        if not texts:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self._predict_batch(batch_texts)
            results.extend(batch_results)
        
        return results
    
    def _predict_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Process a single batch of texts."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_tensors="pt"
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
        
        # Convert to results
        results = []
        probs_cpu = probabilities.cpu().numpy()
        
        for idx, text in enumerate(texts):
            scores_dict = {
                label: float(probs_cpu[idx][i])
                for i, label in enumerate(self.LABELS)
            }
            
            # Get predicted label and confidence
            pred_idx = probs_cpu[idx].argmax()
            pred_label = self.LABELS[pred_idx]
            pred_score = float(probs_cpu[idx][pred_idx])
            
            results.append(SentimentResult(
                text=text[:100] + "..." if len(text) > 100 else text,
                label=pred_label,
                score=pred_score,
                scores=scores_dict
            ))
        
        return results
    
    def get_negative_intensity(self, text: str) -> float:
        """
        Get the negative sentiment intensity for a single text.
        
        Returns:
            Float from 0.0 (not negative) to 1.0 (highly negative)
        """
        results = self.predict([text])
        if not results:
            return 0.0
        return results[0].scores.get("negative", 0.0)
    
    def chunk_text(
        self, 
        full_text: str, 
        window_size: int = 450,  # Leave room for special tokens
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into chunks that fit FinBERT's context window.
        Respects sentence boundaries where possible.
        
        Args:
            full_text: Text to split
            window_size: Target chunk size in tokens (not chars)
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Simple sentence splitting
        import re
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            # Estimate tokens (rough: ~4 chars per token for English)
            sentence_tokens = len(sentence) // 4
            
            if current_tokens + sentence_tokens > window_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


# Singleton accessor
def get_finbert_analyzer() -> FinBERTAnalyzer:
    """Get the singleton FinBERT analyzer instance."""
    return FinBERTAnalyzer()
```

---

## HTTP Client (Production Mode)

### `insights/services/sentiment/finbert_client.py`

```python
"""
HTTP client for FinBERT microservice.
Used in production when FinBERT runs as a separate service.
"""
import logging
from typing import List, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from insights.core.errors import SentimentServiceError
from .finbert_analyzer import SentimentResult

logger = logging.getLogger(__name__)


class FinBERTClient:
    """
    HTTP client for FinBERT microservice.
    
    Usage:
        client = FinBERTClient("http://finbert-service:8000")
        results = await client.predict(["Stock price increased significantly."])
    """
    
    def __init__(
        self, 
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def predict(
        self, 
        texts: List[str],
        batch_size: int = 32
    ) -> List[SentimentResult]:
        """
        Send texts to FinBERT service for analysis.
        
        Args:
            texts: List of texts to analyze
            batch_size: Hint for server-side batching
            
        Returns:
            List of SentimentResult objects
        """
        try:
            response = await self.client.post(
                "/predict",
                json={
                    "texts": texts,
                    "batch_size": batch_size
                }
            )
            response.raise_for_status()
            
            data = response.json()
            return [
                SentimentResult(
                    text=r.get("text", ""),
                    label=r["label"],
                    score=r["score"],
                    scores=r["scores"]
                )
                for r in data.get("results", [])
            ]
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                logger.warning("FinBERT service is loading, retrying...")
                raise  # Will be retried
            raise SentimentServiceError(f"FinBERT service error: {e}")
        
        except httpx.TimeoutException:
            raise SentimentServiceError("FinBERT service timeout")
    
    async def health_check(self) -> bool:
        """Check if FinBERT service is healthy."""
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
```

---

## Agno Toolkit Wrapper

### `insights/services/sentiment/toolkit.py`

```python
"""
Agno Toolkit wrapper for sentiment analysis.
"""
from typing import List, Optional
from agno.tools import Toolkit
from pydantic import BaseModel

from insights.core.config import settings
from .finbert_analyzer import get_finbert_analyzer, SentimentResult
from .finbert_client import FinBERTClient


class SentimentAnalysisResult(BaseModel):
    """Structured result for agent consumption."""
    texts_analyzed: int
    average_sentiment: str
    negative_intensity: float
    results: List[dict]


class SentimentToolkit(Toolkit):
    """
    Sentiment analysis toolkit for Agno agents.
    
    Automatically uses in-process FinBERT in development
    and HTTP client in production.
    """
    
    def __init__(self, finbert_url: Optional[str] = None):
        super().__init__(name="sentiment_toolkit")
        
        self.use_service = settings.ENV == "production" and finbert_url
        self.finbert_url = finbert_url or settings.FINBERT_SERVICE_URL
        
        # Register tools
        self.register(self.analyze_sentiment)
        self.register(self.get_negative_score)
    
    async def analyze_sentiment(
        self, 
        texts: List[str]
    ) -> str:
        """
        Analyze sentiment of multiple texts using FinBERT.
        
        Use this to understand the sentiment of risk factors, news, or other financial text.
        Returns structured analysis with labels (positive/negative/neutral) and confidence scores.
        
        Args:
            texts: List of text passages to analyze
            
        Returns:
            JSON string with sentiment analysis results
        """
        if self.use_service:
            async with FinBERTClient(self.finbert_url) as client:
                results = await client.predict(texts)
        else:
            analyzer = get_finbert_analyzer()
            results = analyzer.predict(texts)
        
        # Aggregate results
        negative_scores = [r.scores.get("negative", 0) for r in results]
        avg_negative = sum(negative_scores) / len(negative_scores) if negative_scores else 0
        
        # Determine overall sentiment
        label_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            label_counts[r.label] += 1
        
        avg_sentiment = max(label_counts, key=label_counts.get)
        
        output = SentimentAnalysisResult(
            texts_analyzed=len(results),
            average_sentiment=avg_sentiment,
            negative_intensity=avg_negative,
            results=[r.to_dict() for r in results]
        )
        
        return output.model_dump_json()
    
    async def get_negative_score(self, text: str) -> float:
        """
        Get the negative sentiment intensity for a single text.
        
        Use this to quickly assess how negative a particular passage is.
        Returns a score from 0.0 (not negative) to 1.0 (highly negative).
        
        Args:
            text: Text to analyze
            
        Returns:
            Negative intensity score (0.0 to 1.0)
        """
        if self.use_service:
            async with FinBERTClient(self.finbert_url) as client:
                results = await client.predict([text])
                return results[0].scores.get("negative", 0.0) if results else 0.0
        else:
            analyzer = get_finbert_analyzer()
            return analyzer.get_negative_intensity(text)
```

---

## FinBERT Microservice

### `finbert/main.py`

```python
"""
FinBERT Microservice - Standalone FastAPI service for sentiment analysis.
Run this as a separate container with GPU access.
"""
import logging
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import after logging setup
from insights.services.sentiment.finbert_analyzer import FinBERTAnalyzer


class PredictRequest(BaseModel):
    texts: List[str]
    batch_size: int = 32


class PredictResponse(BaseModel):
    results: List[dict]
    model: str = "ProsusAI/finbert"


# Global analyzer instance
analyzer: FinBERTAnalyzer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global analyzer
    logger.info("Loading FinBERT model...")
    analyzer = FinBERTAnalyzer()
    logger.info("FinBERT model ready")
    yield
    logger.info("Shutting down FinBERT service")


app = FastAPI(
    title="FinBERT Sentiment Service",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "ProsusAI/finbert"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict sentiment for a list of texts.
    
    Returns:
        List of sentiment results with labels and scores
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts:
        return PredictResponse(results=[])
    
    try:
        results = analyzer.predict(request.texts, batch_size=request.batch_size)
        return PredictResponse(
            results=[r.to_dict() for r in results]
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### `finbert/Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Pre-download model during build
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
    AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `finbert/requirements.txt`

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
transformers==4.37.0
pydantic==2.5.0
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Model load time | <10s | Cached in container image |
| Inference latency (1 text) | <50ms | GPU required |
| Batch throughput (32 texts) | <500ms | Optimal batch size |
| Memory usage | ~2GB | Model + tokenizer |

---

## Configuration

### Environment Variables

```bash
# Development (in-process)
FINBERT_MODE=local

# Production (microservice)
FINBERT_MODE=service
FINBERT_SERVICE_URL=http://finbert-service:8000
```

### Docker Compose

```yaml
services:
  finbert:
    build: ./finbert
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8001:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```