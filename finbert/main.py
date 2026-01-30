import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure insights package is in path (handled by Docker structure)
from insights.services.sentiment.finbert_analyzer import FinBERTAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finbert-service")

app = FastAPI(title="FinBERT Sentiment Service")

# Initialize analyzer on startup
analyzer = None

@app.on_event("startup")
async def startup_event():
    global analyzer
    try:
        logger.info("Initializing FinBERT Analyzer...")
        analyzer = FinBERTAnalyzer()
        logger.info("FinBERT Analyzer ready.")
    except Exception as e:
        logger.error(f"Failed to initialize FinBERT: {e}")
        # We don't raise here to allow container to start and report unhealthiness?
        # Better to fail fast.
        raise RuntimeError(f"Model initialization failed: {e}") from e

class AnalyzeRequest(BaseModel):
    texts: list[str]

class SentimentResponse(BaseModel):
    text: str
    label: str
    score: float
    scores: dict[str, float]

@app.get("/health")
def health_check():
    if analyzer:
        return {"status": "healthy"}
    return {"status": "not_ready"}

@app.post("/analyze", response_model=list[SentimentResponse])
async def analyze(request: AnalyzeRequest):
    if not analyzer:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        results = analyzer.predict(request.texts)
        return [r.to_dict() for r in results]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
