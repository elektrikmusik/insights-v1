"""
Celery task definitions for async analysis.
"""
import logging
import asyncio
from datetime import datetime, UTC
from typing import Optional, List
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from .celery_app import celery_app
from insights.adapters.db.manager import DBManager
from insights.agents.orchestrator import Orchestrator
from insights.core.events import publish_progress, set_current_job_id
from insights.core.types import JobStatus

logger = logging.getLogger(__name__)


class AnalysisTask(Task):
    """Base task with error handling and DB access."""
    
    _db: Optional[DBManager] = None
    
    @property
    def db(self) -> DBManager:
        if self._db is None:
            self._db = DBManager()
        return self._db
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        job_id = kwargs.get("job_id") or args[0]
        
        # Update job status
        asyncio.run(self.db.update_job_status(
            job_id,
            status=JobStatus.FAILED,
            error_message=str(exc)
        ))
        
        # Publish failure event
        publish_progress(job_id, {
            "event": "job_failed",
            "error": str(exc)
        })


@celery_app.task(
    bind=True,
    base=AnalysisTask,
    name="insights.workers.tasks.analyze_risk_drift",
    max_retries=3,
    default_retry_delay=60
)
def analyze_risk_drift(
    self,
    job_id: str,
    ticker: str,
    years: List[int],
    options: dict | None = None,
    callback_url: Optional[str] = None
):
    """
    Execute risk drift analysis for a company.
    """
    asyncio.run(_analyze_risk_drift_async(
        self, job_id, ticker, years, options, callback_url
    ))


async def _analyze_risk_drift_async(
    task: AnalysisTask,
    job_id: str,
    ticker: str,
    years: List[int],
    options: dict | None = None,
    callback_url: Optional[str] = None
):
    """Async implementation of risk drift analysis."""
    from insights.adapters.mcp.client import get_mcp_client
    from insights.services.risk_drift import RiskDriftPipeline

    db = task.db
    set_current_job_id(job_id)

    try:
        # Update status: processing
        await db.update_job_status(job_id, JobStatus.RUNNING, progress=0)
        publish_progress(job_id, {"event": "job_started", "ticker": ticker})
        
        # Run Risk Drift Pipeline (DB-first approach)
        logger.info(f"Task {job_id}: Running risk drift pipeline for {ticker}")
        pipeline = RiskDriftPipeline(db_manager=db)
        
        # Execute pipeline
        pipeline_result = await pipeline.run(ticker, years)
        
        # Extract results
        full_analysis = pipeline_result.full_analysis
        markdown_report = pipeline_result.markdown_report
        company_id = pipeline_result.company_id
        
        logger.info(f"Task {job_id}: Pipeline completed. Company ID: {company_id}")
        
        # Update job status with full_analysis as result_summary
        await db.update_job_status(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            result_summary=full_analysis,  # Full analysis JSON
        )
        
        # Save report to reports table
        await db.save_report(
            company_id=company_id,
            title=f"Risk Drift Analysis: {ticker} ({years[0]} vs {years[1]})",
            report_type="risk_drift",
            markdown_content=markdown_report,
            job_id=job_id,
            parameters={"ticker": ticker, "years": years},
            summary=full_analysis,  # Full analysis JSON in summary field
        )
        
        logger.info(f"Task {job_id}: Report saved")
        
        publish_progress(job_id, {"event": "job_completed"})
        
        # Send webhook
        if callback_url:
            await _send_webhook(callback_url, job_id, JobStatus.COMPLETED, full_analysis)
        
    except SoftTimeLimitExceeded:
        logger.error(f"Job {job_id} exceeded time limit")
        await db.update_job_status(
            job_id, JobStatus.FAILED,
            error_message="Analysis exceeded time limit"
        )
        raise
    
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        await db.update_job_status(
            job_id, JobStatus.FAILED,
            error_message=str(e)
        )
        
        # Send failure webhook
        if callback_url:
            try:
                await _send_webhook(callback_url, job_id, JobStatus.FAILED, {"error": str(e)})
            except Exception as swe:
                logger.warning(f"Failed to send failure webhook: {swe}")
                
        raise

    finally:
        set_current_job_id(None)
        # Disconnect MCP so SSE context is closed in the same task (avoids cancel-scope error)
        try:
            client = get_mcp_client()
            await client.disconnect()
        except Exception as e:
            logger.debug(f"MCP disconnect: {e}")


async def _send_webhook(url: str, job_id: str, status: JobStatus, data: dict):
    """Send webhook notification."""
    import httpx
    import hmac
    import hashlib
    from insights.core.config import settings
    
    payload = {
        "event": f"job.{status}",
        "job_id": job_id,
        "status": status,
        "data": data,
        "timestamp": datetime.now(UTC).isoformat()
    }
    
    # Sign payload if secret is configured
    signature = None
    if settings.WEBHOOK_SECRET:
        signature = hmac.new(
            settings.WEBHOOK_SECRET.encode(),
            json.dumps(payload, sort_keys=True).encode(),
            hashlib.sha256
        ).hexdigest()
    
    headers = {"Content-Type": "application/json"}
    if signature:
        headers["X-Signature"] = signature
        
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=10
            )
        except Exception as e:
            logger.warning(f"Webhook delivery failed for {job_id}: {e}")
