"""
Server-Sent Events endpoint for job progress.
"""
import asyncio
import json
import logging
from uuid import UUID
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from insights.adapters.db.manager import DBManager, get_db, JobRecord
from insights.core.events import get_publisher

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stream/{job_id}")
async def stream_job_progress(job_id: str):
    """
    Stream job progress via Server-Sent Events.
    """
    db = get_db()
    
    try:
        uuid_obj = UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Job ID format")
    
    # Verify job exists
    job = await db.get_job(uuid_obj)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # If already complete, yield final status immediately
    if job.status in ("completed", "failed", "cancelled"):
        return StreamingResponse(
            _yield_final_status(job),
            media_type="text/event-stream"
        )
    
    # Stream progress events
    return StreamingResponse(
        _stream_progress(job_id),
        media_type="text/event-stream"
    )


async def _stream_progress(job_id: str) -> AsyncGenerator[str, None]:
    """Generator for SSE events."""
    publisher = get_publisher()
    pubsub = await publisher.subscribe(job_id)
    
    try:
        # Initial connection event
        yield f"event: connected\ndata: {json.dumps({'message': 'Connected to stream'})}\n\n"
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                event_type = data.get("event", "progress")
                
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(data)}\n\n"
                
                # Stop on terminal events
                if event_type in ("job_completed", "job_failed"):
                    break
    except asyncio.CancelledError:
        logger.info(f"Stream cancelled for job {job_id}")
    finally:
        await pubsub.unsubscribe()


async def _yield_final_status(job: JobRecord):
    """Yield final status for completed jobs."""
    event = "job_completed" if job.status == "completed" else "job_failed"
    data = {
        "event": event,
        "status": job.status,
        "result": job.result_summary
    }
    yield f"event: {event}\n"
    yield f"data: {json.dumps(data)}\n\n"
