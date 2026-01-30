"""
Event publishing for real-time progress updates.
Uses Redis PubSub for SSE streaming.
Publish is synchronous (sync Redis) so it works when called from sync tool code in worker threads.
"""
import json
import contextvars
import redis
import redis.asyncio as redis_async
from typing import Any, Dict, Optional
from insights.core.config import settings

# Current job_id for the running analysis (set by worker, read by toolkits for streaming)
_current_job_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("insights_job_id", default=None)


def get_current_job_id() -> Optional[str]:
    """Return the job_id for the current analysis context (if set)."""
    return _current_job_id.get()


def set_current_job_id(job_id: Optional[str]) -> None:
    """Set the job_id for the current analysis context (used by worker)."""
    _current_job_id.set(job_id)


class ProgressPublisher:
    """Publishes progress events to Redis PubSub. Sync publish for any-thread use; async subscribe for stream."""
    
    def __init__(self):
        self._redis_sync = redis.from_url(settings.REDIS_URL)
        self._redis_async = redis_async.from_url(settings.REDIS_URL)
    
    def publish_sync(self, job_id: str, event: Dict[str, Any]) -> None:
        """Publish progress event synchronously. Safe to call from any thread (e.g. sync toolkit code)."""
        channel = f"job:{job_id}:progress"
        self._redis_sync.publish(channel, json.dumps(event))
    
    async def subscribe(self, job_id: str):
        """Subscribe to progress events for a job (async, for stream endpoint)."""
        channel = f"job:{job_id}:progress"
        pubsub = self._redis_async.pubsub()
        await pubsub.subscribe(channel)
        return pubsub


_publisher = None


def get_publisher() -> ProgressPublisher:
    global _publisher
    if _publisher is None:
        _publisher = ProgressPublisher()
    return _publisher


def publish_progress(job_id: str, event: dict) -> None:
    """Publish a progress event. Uses sync Redis so it works from sync or async context and any thread."""
    get_publisher().publish_sync(job_id, event)


def publish_tool_used(
    toolkit_name: str,
    tool_name: str,
    message: str = "",
    **extra: Any,
) -> None:
    """
    Stream a tool_used event when any toolkit tool is invoked.
    Uses current job_id from context; no-op if not in a job context.
    """
    job_id = get_current_job_id()
    if not job_id:
        return
    event = {
        "event": "tool_used",
        "toolkit": toolkit_name,
        "tool": tool_name,
        "message": message or f"{toolkit_name}.{tool_name}",
        **extra,
    }
    publish_progress(job_id, event)
