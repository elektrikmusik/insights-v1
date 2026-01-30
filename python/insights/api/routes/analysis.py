"""
Analysis routes for triggering the Expert System.
"""
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException
from insights.core.types import AnalyzeRequest, AnalyzeResponse, JobStatus
from insights.adapters.db.manager import get_db, JobCreate
from insights.workers.tasks import analyze_risk_drift

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Submit a ticker for analysis. Returns a Job ID.
    The analysis runs asynchronously via Celery worker.
    """
    db = get_db()
    
    try:
        # Create Job in DB to track status
        job_uuid = await db.create_job(JobCreate(
            job_type="risk_drift",
            request_payload=request.model_dump(),
        ))
        job_id = str(job_uuid)
        
        # Dispatch Celery Task
        analyze_risk_drift.delay(
            job_id=job_id,
            ticker=request.ticker,
            years=request.years,
            options=request.options.model_dump() if request.options else None,
            callback_url=request.webhook_url
        )
        
        return AnalyzeResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message=f"Analysis scheduled for {request.ticker}"
        )
        
    except Exception as e:
        logger.error(f"Analysis submission failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to submit analysis job")


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """
    Retrieve job status and result.
    """
    db = get_db()
    
    try:
        if not job_id:
             raise ValueError("Empty ID")
        uuid_obj = UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")
    
    job = await db.get_job(uuid_obj)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job
