import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_health_endpoint(test_client):
    """Test the /health endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "insights-ai"
    assert "timestamp" in data

@pytest.mark.asyncio
async def test_ready_endpoint(test_client):
    """Test the /ready endpoint."""
    response = test_client.get("/ready")
    assert response.status_code == 200
    assert "status" in response.json()

@pytest.mark.asyncio
async def test_analyze_success(test_client):
    """Test /analyze with mocked dependencies."""
    with patch("insights.api.routes.analysis.get_db") as mock_get_db, \
         patch("insights.api.routes.analysis.analyze_risk_drift.delay") as mock_delay:
        
        # Mock DB response
        mock_db = MagicMock()
        mock_db.create_job = AsyncMock(return_value="550e8400-e29b-41d4-a716-446655440000")
        mock_get_db.return_value = mock_db
        
        response = test_client.post("/api/v1/analyze", json={
            "ticker": "AAPL",
            "years": [2024, 2023]
        })
        
        assert response.status_code == 200 # Current implementation returns 200 via Pydantic model
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        mock_delay.assert_called_once()
