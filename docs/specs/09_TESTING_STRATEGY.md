# 09. Testing Strategy

## Overview

Testing focuses on:
1. **Unit Tests**: Pure business logic (domain services)
2. **Integration Tests**: Component interactions (mocked externals)
3. **E2E Tests**: Critical workflows (limited)

---

## Test Pyramid

```
                    ┌───────────────────┐
                    │     E2E Tests     │  ~5%
                    └─────────────────────┘
                    ┌───────────────────────┐
                    │  Integration Tests     │  ~30%
                    └─────────────────────────┘
            ┌─────────────────────────────────────┐
            │            Unit Tests               │  ~65%
            └─────────────────────────────────────┘
```

---

## Shared Fixtures

### `tests/conftest.py`

```python
"""Shared pytest fixtures."""
import os
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

os.environ["ENV"] = "test"

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_filing_text() -> str:
    return """
    ITEM 1A. RISK FACTORS
    
    **Supply Chain Concentration**
    We depend on suppliers in Asia for key components.
    
    **Regulatory Uncertainty**
    Changes in government policies may affect operations.
    """


@pytest.fixture
def mock_mcp_client() -> AsyncMock:
    mock = AsyncMock()
    mock.call_tool = AsyncMock(return_value="ITEM 1A. RISK FACTORS...")
    return mock


@pytest.fixture
def mock_openrouter() -> AsyncMock:
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value='{"risks": []}')
    mock.model_id = "openai/gpt-4o"
    return mock


@pytest.fixture
def mock_finbert() -> MagicMock:
    from tests.mocks.mock_finbert import MockFinBERTAnalyzer
    return MockFinBERTAnalyzer()
```

---

## Mock FinBERT

### `tests/mocks/mock_finbert.py`

```python
"""Mock FinBERT for testing without model."""
from typing import List
from dataclasses import dataclass


@dataclass
class MockSentimentResult:
    text: str
    label: str
    score: float
    scores: dict
    
    def to_dict(self) -> dict:
        return {"label": self.label, "score": self.score, "scores": self.scores}


class MockFinBERTAnalyzer:
    """Mock FinBERT with keyword-based sentiment."""
    
    def predict(self, texts: List[str], batch_size: int = 32) -> List[MockSentimentResult]:
        results = []
        for text in texts:
            text_lower = text.lower()
            if any(w in text_lower for w in ["risk", "loss", "decline"]):
                results.append(MockSentimentResult(
                    text=text[:100], label="negative", score=0.85,
                    scores={"positive": 0.05, "negative": 0.85, "neutral": 0.10}
                ))
            else:
                results.append(MockSentimentResult(
                    text=text[:100], label="neutral", score=0.70,
                    scores={"positive": 0.15, "negative": 0.15, "neutral": 0.70}
                ))
        return results
```

---

## Unit Tests

### `tests/unit/services/test_drift_calculator.py`

```python
"""Unit tests for DriftCalculator."""
import pytest
from insights.services.risk.drift_calculator import DriftCalculator, RiskFactor


class TestDriftCalculator:
    @pytest.fixture
    def calculator(self):
        return DriftCalculator()
    
    def test_fuzzy_title_matching(self, calculator):
        score = calculator._fuzzy_match("Supply Chain Risk", "Supply Chain Concentration")
        assert score > 0.7
    
    def test_new_risk_detection(self, calculator):
        current = [RiskFactor(rank=1, title="AI Competition", content="...")]
        previous = [RiskFactor(rank=1, title="Market Risk", content="...")]
        
        results = calculator.analyze_drift(current, previous)
        new_risks = [r for r in results if r.drift_type == "new"]
        assert len(new_risks) == 1


class TestHeatScorer:
    @pytest.fixture
    def scorer(self):
        from insights.services.risk.heat_scorer import HeatScorer
        return HeatScorer()
    
    def test_high_heat_for_large_rank_change(self, scorer):
        score = scorer.compute(rank_delta=5, semantic_score=0.8, is_new=False)
        assert score > 0.7
    
    def test_low_heat_for_stable(self, scorer):
        score = scorer.compute(rank_delta=0, semantic_score=0.95, is_new=False)
        assert score < 0.3
```

---

## Integration Tests

### `tests/integration/test_api_endpoints.py`

```python
"""Integration tests for API endpoints."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestAnalyzeEndpoint:
    async def test_analyze_returns_job_id(self, test_client: AsyncClient):
        response = await test_client.post("/analyze", json={
            "ticker": "AAPL", "years": [2024, 2023]
        })
        assert response.status_code == 202
        assert "job_id" in response.json()
    
    async def test_health_check(self, test_client: AsyncClient):
        response = await test_client.get("/health")
        assert response.status_code == 200
```

---

## Running Tests

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest tests/unit -v

# With coverage
uv run pytest --cov=src --cov-report=html

# Parallel execution
uv run pytest -n auto
```

---

## Coverage Requirements

| Component | Minimum |
|-----------|---------|
| Domain Services | 90% |
| Adapters | 80% |
| API Routes | 80% |
| Overall | 80% |