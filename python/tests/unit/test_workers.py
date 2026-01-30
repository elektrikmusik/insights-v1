import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, UTC

from insights.workers.tasks import _analyze_risk_drift_async
from insights.core.types import JobStatus, SynthesizedReport, Company

@pytest.mark.asyncio
async def test_analyze_risk_drift_task_success():
    # Mock dependencies
    job_id = "test-job-id"
    ticker = "AAPL"
    years = [2024, 2023]
    
    mock_task = MagicMock()
    mock_db = MagicMock()
    mock_db.update_job_status = AsyncMock()
    mock_task.db = mock_db
    
    mock_report = SynthesizedReport(
        company=Company(ticker="AAPL", name="Apple", id="AAPL"),
        summary="Risk level increased",
        expert_findings=[],
        risk_level="medium"
    )
    
    with patch("insights.workers.tasks.Orchestrator") as MockOrch, \
         patch("insights.workers.tasks.publish_progress") as mock_pub:
        
        mock_orch_instance = MockOrch.return_value
        mock_orch_instance.process_request = AsyncMock(return_value=mock_report)
        
        await _analyze_risk_drift_async(mock_task, job_id, ticker, years, None, None)
        
        # Verify status updates
        assert mock_db.update_job_status.call_count == 2
        
        # Verify first call (status=running)
        call_1 = mock_db.update_job_status.call_args_list[0]
        assert call_1.args[0] == job_id
        assert call_1.args[1] == JobStatus.RUNNING
        
        # Verify second call (status=completed)
        call_2 = mock_db.update_job_status.call_args_list[1]
        assert call_2.args[0] == job_id
        assert call_2.kwargs["status"] == JobStatus.COMPLETED
        assert call_2.kwargs["result_summary"] == "Risk level increased"
        assert call_2.kwargs["result"] == mock_report.model_dump()
        assert isinstance(call_2.kwargs["completed_at"], datetime)

@pytest.mark.asyncio
async def test_analyze_risk_drift_task_failure():
    job_id = "fail-job-id"
    ticker = "FAIL"
    years = [2024]
    
    mock_task = MagicMock()
    mock_db = MagicMock()
    mock_db.update_job_status = AsyncMock()
    mock_task.db = mock_db
    
    with patch("insights.workers.tasks.Orchestrator") as MockOrch:
        mock_orch_instance = MockOrch.return_value
        mock_orch_instance.process_request = AsyncMock(side_effect=ValueError("Test Error"))
        
        with pytest.raises(ValueError, match="Test Error"):
            await _analyze_risk_drift_async(mock_task, job_id, ticker, years, None, None)
            
        # Verify failure update
        mock_db.update_job_status.assert_any_call(
            job_id,
            JobStatus.FAILED,
            error_message="Test Error"
        )
