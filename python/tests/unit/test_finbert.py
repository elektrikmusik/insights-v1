from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from insights.adapters.sentiment import LocalSentimentAdapter
from insights.services.sentiment import FinBERTClient, SentimentResult
from insights.services.sentiment.finbert_analyzer import FinBERTAnalyzer


class TestFinBERTClient:
    @pytest.mark.asyncio
    async def test_predict_success(self):
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "text": "Good news",
                    "label": "positive",
                    "score": 0.9,
                    "scores": {"positive": 0.9, "negative": 0.1, "neutral": 0.0}
                }
            ]
            mock_post.return_value = mock_response
            # Async context manager mock
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None

            with patch("httpx.AsyncClient", return_value=mock_client):
                client = FinBERTClient(base_url="http://test")
                results = await client.predict(["Good news"])

                assert len(results) == 1
                assert results[0].label == "positive"
                assert results[0].score == 0.9

class TestLocalSentimentAdapter:
    @pytest.mark.asyncio
    async def test_predict_offload(self):
        # Mock FinBERTAnalyzer to avoid loading real model
        with patch("insights.services.sentiment.finbert_analyzer.FinBERTAnalyzer") as MockAnalyzer:
            mock_instance = MockAnalyzer.return_value
            mock_instance.predict.return_value = [
                SentimentResult(
                    text="Bad news",
                    label="negative",
                    score=0.8,
                    scores={}
                )
            ]

            # Use patch for the inner analyzer creation in adapter
            with patch("insights.adapters.sentiment.FinBERTAnalyzer", return_value=mock_instance):
                adapter = LocalSentimentAdapter()
                results = await adapter.predict(["Bad news"])

                assert len(results) == 1
                assert results[0].label == "negative"
                # Check that it called the sync method
                mock_instance.predict.assert_called_once()

class TestFinBERTAnalyzer:
    def test_init_check(self):
        # Test usage when transformers is present (mocked)
        with patch("insights.services.sentiment.finbert_analyzer.TRANSFORMERS_AVAILABLE", True):
            with patch("insights.services.sentiment.finbert_analyzer.AutoTokenizer"):
                with patch("insights.services.sentiment.finbert_analyzer.AutoModelForSequenceClassification"):
                   analyzer = FinBERTAnalyzer()
                   assert analyzer.model is not None

    def test_init_fail(self):
         # Test usage when transformers missing
         with patch("insights.services.sentiment.finbert_analyzer.TRANSFORMERS_AVAILABLE", False):
             with pytest.raises(ImportError):
                 FinBERTAnalyzer()
