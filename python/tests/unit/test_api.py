import pytest
from uuid import uuid4
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock

# Mock settings to avoid loading real env/secrets
with patch("insights.core.config.settings.DEBUG", True), \
     patch("insights.core.config.settings.SUPABASE_URL", "https://mock.supabase.co"), \
     patch("insights.core.config.settings.SUPABASE_ANON_KEY", "mock-key"), \
     patch("insights.core.config.settings.REDIS_URL", "redis://localhost:6379/0"):
    from insights.api.main import app
    from insights.core.types import JobStatus, Company

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

@pytest.mark.asyncio
async def test_analyze_endpoint():
    job_id = uuid4()
    
    # Mock DB and Celery
    with patch("insights.api.routes.analysis.get_db") as mock_get_db, \
         patch("insights.api.routes.analysis.analyze_risk_drift") as mock_task:
        
        mock_db = MagicMock()
        mock_db.create_job = AsyncMock(return_value=job_id)
        mock_get_db.return_value = mock_db
        
        # Dispatch
        payload = {"ticker": "AAPL", "years": [2024, 2023]}
        response = client.post("/api/v1/analyze", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == str(job_id)
        assert data["status"] == JobStatus.PENDING
        
        # Verify Celery delay was called
        mock_task.delay.assert_called_once_with(
            job_id=str(job_id),
            ticker="AAPL",
            years=[2024, 2023],
            options={"include_sentiment": True, "generate_heatmap": True, "depth": "deep"},
            callback_url=None
        )

@pytest.mark.asyncio
async def test_get_job_status():
    job_id = uuid4()
    
    with patch("insights.api.routes.analysis.get_db") as mock_get_db:
        mock_db = MagicMock()
        # Mock JobRecord-like object (or dict depending on what get_job returns)
        # JobRecord is a Pydantic model in DB manager
        from insights.adapters.db.manager import JobRecord
        mock_job = JobRecord(
            id=job_id,
            job_type="risk_drift",
            status="completed",
            progress=100,
            request_payload={"ticker": "AAPL", "years": [2024, 2023]},
            result_summary={"summary": "Excellent growth"},
            created_at=None,
            started_at=None,
            completed_at=None
        )
        mock_db.get_job = AsyncMock(return_value=mock_job)
        mock_get_db.return_value = mock_db
        
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(job_id)
        assert data["status"] == "completed"
        assert data["result_summary"]["summary"] == "Excellent growth"

def test_get_job_not_found():
    with patch("insights.api.routes.analysis.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_db.get_job = AsyncMock(return_value=None)
        mock_get_db.return_value = mock_db
        
        response = client.get(f"/api/v1/jobs/{uuid4()}")
        assert response.status_code == 404
