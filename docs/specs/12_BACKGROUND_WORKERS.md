# 12. Background Workers (Celery)

## Overview

Background workers handle long-running analysis tasks asynchronously:
- Job queuing with Redis
- Celery workers for task execution
- SSE for real-time progress updates
- Webhook callbacks on completion

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Client Request: POST /analyze                                               │
│  → Returns: {job_id: "abc123", status: "queued"}                           │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │ Enqueue task
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Redis Queue                                                                 │
│  └── analysis_queue: [job_id: "abc123", ticker: "AAPL", years: [2024,2023]]│
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │ Dequeue & process
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Celery Worker (1-N instances)                                              │
│  └── analyze_risk_drift task                                                │
│      ├── Creates Research Agent                                             │
│      ├── Agent orchestrates toolkits                                        │
│      ├── Publishes progress to Redis PubSub                                │
│      └── Updates job_queue table                                            │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │ Completion
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Notifications                                                               │
│  ├── SSE: /stream/{job_id} receives completion event                       │
│  └── Webhook: POST to callback_url with results                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Celery Configuration

### `insights/workers/celery_app.py`

```python
"""
Celery application configuration.
"""
from celery import Celery
from insights.core.config import settings

# Create Celery app
celery_app = Celery(
    "insights",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# Configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Time limits
    task_time_limit=600,        # 10 minutes hard limit
    task_soft_time_limit=540,   # 9 minutes soft limit
    
    # Retry policy
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Prefetch
    worker_prefetch_multiplier=1,  # Fair distribution
    
    # Results
    result_expires=3600,  # 1 hour
    
    # Task routing
    task_routes={
        "insights.workers.tasks.analyze_risk_drift": {"queue": "analysis"},
        "insights.workers.tasks.generate_embeddings": {"queue": "embeddings"},
    },
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Autodiscover tasks
celery_app.autodiscover_tasks(["insights.workers"])
```

---

## Task Definitions

### `insights/workers/tasks.py`

```python
"""
Celery task definitions for async analysis.
"""
import logging
from datetime import datetime, UTC
from typing import Optional
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from .celery_app import celery_app
from insights.adapters.db.manager import DBManager
from insights.agents.research import get_research_agent
from insights.core.events import publish_progress

logger = logging.getLogger(__name__)


class AnalysisTask(Task):
    """Base task with error handling."""
    
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
        import asyncio
        asyncio.run(self.db.update_job_status(
            job_id,
            status="failed",
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
    years: list,
    callback_url: Optional[str] = None
):
    """
    Execute risk drift analysis for a company.
    
    Args:
        job_id: Unique job identifier
        ticker: Company ticker symbol
        years: [current_year, previous_year]
        callback_url: Optional webhook URL
    """
    import asyncio
    asyncio.run(_analyze_risk_drift_async(
        self, job_id, ticker, years, callback_url
    ))


async def _analyze_risk_drift_async(
    task: AnalysisTask,
    job_id: str,
    ticker: str,
    years: list,
    callback_url: Optional[str]
):
    """Async implementation of risk drift analysis."""
    db = task.db
    
    try:
        # Update status: processing
        await db.update_job_status(job_id, "processing", progress=0)
        publish_progress(job_id, {"event": "job_started", "ticker": ticker})
        
        # Create agent
        agent = get_research_agent(job_id=job_id)
        
        # Progress callback for agent steps
        def on_step(step_name: str, progress: int):
            publish_progress(job_id, {
                "event": "step_complete",
                "step": step_name,
                "progress": progress
            })
        
        # Construct prompt
        prompt = f"""
        Analyze risk drift for {ticker} comparing fiscal years {years[0]} and {years[1]}.
        
        Execute the full workflow:
        1. Fetch Risk Factors from both 10-K filings
        2. Extract and structure individual risks
        3. Analyze sentiment for each risk
        4. Calculate drift and changes
        5. Generate heatmap visualization data
        6. Provide executive summary and recommendations
        """
        
        # Run agent
        on_step("initializing", 10)
        response = await agent.arun(prompt)
        on_step("agent_complete", 80)
        
        # Parse results
        result_data = _extract_results(response)
        
        # Save to database
        await _save_results(db, job_id, ticker, years, result_data)
        on_step("saving", 90)
        
        # Update status: completed
        await db.update_job_status(
            job_id,
            status="completed",
            progress=100,
            result_summary=result_data.get("summary", ""),
            completed_at=datetime.now(UTC)
        )
        
        publish_progress(job_id, {"event": "job_completed"})
        
        # Send webhook
        if callback_url:
            await _send_webhook(callback_url, job_id, "completed", result_data)
        
    except SoftTimeLimitExceeded:
        logger.error(f"Job {job_id} exceeded time limit")
        await db.update_job_status(
            job_id, "failed",
            error_message="Analysis exceeded time limit"
        )
        raise
    
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        await db.update_job_status(
            job_id, "failed",
            error_message=str(e)
        )
        
        if callback_url:
            await _send_webhook(callback_url, job_id, "failed", {"error": str(e)})
        
        # Retry for transient errors
        if _is_retriable(e):
            raise task.retry(exc=e)
        raise


def _extract_results(response) -> dict:
    """Extract structured data from agent response."""
    import json
    import re
    
    content = response.content if hasattr(response, 'content') else str(response)
    
    # Try to find JSON in response
    json_match = re.search(r'\{[\s\S]*"heatmap"[\s\S]*\}', content)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    return {
        "summary": content[:500],
        "raw_response": content
    }


async def _save_results(db, job_id, ticker, years, data):
    """Save analysis results to database."""
    await db.upsert_analysis(
        ticker=ticker,
        year_current=years[0],
        year_previous=years[1],
        job_id=job_id,
        summary=data.get("summary"),
        heatmap=data.get("heatmap"),
        drifts=data.get("drifts", [])
    )


async def _send_webhook(url: str, job_id: str, status: str, data: dict):
    """Send webhook notification."""
    import httpx
    import hmac
    import hashlib
    from insights.core.config import settings
    
    payload = {
        "job_id": job_id,
        "status": status,
        "data": data,
        "timestamp": datetime.now(UTC).isoformat()
    }
    
    # Sign payload
    signature = hmac.new(
        settings.WEBHOOK_SECRET.encode(),
        str(payload).encode(),
        hashlib.sha256
    ).hexdigest()
    
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                url,
                json=payload,
                headers={"X-Signature": signature},
                timeout=10
            )
        except Exception as e:
            logger.warning(f"Webhook failed: {e}")


def _is_retriable(error: Exception) -> bool:
    """Check if error is transient and retriable."""
    retriable_types = (
        ConnectionError,
        TimeoutError,
    )
    return isinstance(error, retriable_types)
```

---

## Progress Events

### `insights/core/events.py`

```python
"""
Event publishing for real-time progress updates.
Uses Redis PubSub for SSE streaming.
"""
import json
import redis.asyncio as redis
from typing import Any, Dict
from insights.core.config import settings


class ProgressPublisher:
    """Publishes progress events to Redis PubSub."""
    
    def __init__(self):
        self.redis = redis.from_url(settings.REDIS_URL)
    
    async def publish(self, job_id: str, event: Dict[str, Any]):
        """Publish progress event for a job."""
        channel = f"job:{job_id}:progress"
        await self.redis.publish(channel, json.dumps(event))
    
    async def subscribe(self, job_id: str):
        """Subscribe to progress events for a job."""
        channel = f"job:{job_id}:progress"
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(channel)
        return pubsub


_publisher = None


def get_publisher() -> ProgressPublisher:
    global _publisher
    if _publisher is None:
        _publisher = ProgressPublisher()
    return _publisher


def publish_progress(job_id: str, event: dict):
    """Synchronous wrapper for publish."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(get_publisher().publish(job_id, event))
        else:
            loop.run_until_complete(get_publisher().publish(job_id, event))
    except RuntimeError:
        asyncio.run(get_publisher().publish(job_id, event))
```

---

## SSE Endpoint

### `insights/routes/stream.py`

```python
"""
Server-Sent Events endpoint for job progress.
"""
import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from insights.core.events import get_publisher
from insights.adapters.db.manager import DBManager

router = APIRouter()


@router.get("/stream/{job_id}")
async def stream_job_progress(job_id: str):
    """
    Stream job progress via Server-Sent Events.
    
    Events:
    - job_started: Analysis began
    - step_complete: A step finished
    - job_completed: Analysis done
    - job_failed: Analysis failed
    """
    db = DBManager()
    
    # Verify job exists
    job = await db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    # If already complete, return final status
    if job["status"] in ("completed", "failed"):
        return StreamingResponse(
            _yield_final_status(job),
            media_type="text/event-stream"
        )
    
    # Stream progress events
    return StreamingResponse(
        _stream_progress(job_id),
        media_type="text/event-stream"
    )


async def _stream_progress(job_id: str):
    """Generator for SSE events."""
    publisher = get_publisher()
    pubsub = await publisher.subscribe(job_id)
    
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                yield f"event: {data.get('event', 'progress')}\n"
                yield f"data: {json.dumps(data)}\n\n"
                
                # Stop on terminal events
                if data.get("event") in ("job_completed", "job_failed"):
                    break
    finally:
        await pubsub.unsubscribe()


async def _yield_final_status(job: dict):
    """Yield final status for completed jobs."""
    event = "job_completed" if job["status"] == "completed" else "job_failed"
    data = {
        "event": event,
        "status": job["status"],
        "result": job.get("result_summary")
    }
    yield f"event: {event}\n"
    yield f"data: {json.dumps(data)}\n\n"
```

---

## Running Workers

### Development

```bash
# Start single worker
celery -A insights.workers.celery_app worker --loglevel=info

# Start with specific queue
celery -A insights.workers.celery_app worker -Q analysis --loglevel=info

# Start Flower monitor
celery -A insights.workers.celery_app flower --port=5555
```

### Production (Docker)

```yaml
# docker-compose.yml
services:
  worker:
    build:
      dockerfile: docker/Dockerfile.worker
    command: celery -A insights.workers.celery_app worker -Q analysis --loglevel=info --concurrency=4
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    deploy:
      replicas: 2
```

---

## Monitoring

### Flower Dashboard

```bash
# Install
pip install flower

# Run
celery -A insights.workers.celery_app flower --port=5555
# Access at http://localhost:5555
```

### Key Metrics

| Metric | Target |
|--------|--------|
| Task latency (p95) | < 3 min |
| Queue depth | < 50 |
| Worker utilization | 60-80% |
| Failure rate | < 1% |
