"""
Retry utilities using Tenacity.

Provides decorators and helpers for resilient external calls.
"""
import logging
from collections.abc import Callable
from functools import wraps

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# Standard retry configuration for external services
RETRY_ATTEMPTS = 3
RETRY_MIN_WAIT = 1  # seconds
RETRY_MAX_WAIT = 10  # seconds


def retry_on_error(
    exceptions: tuple[type[Exception], ...] = (Exception,),
    attempts: int = RETRY_ATTEMPTS,
    min_wait: float = RETRY_MIN_WAIT,
    max_wait: float = RETRY_MAX_WAIT,
):
    """
    Decorator for retrying functions on specific exceptions.

    Args:
        exceptions: Tuple of exception types to retry on
        attempts: Maximum number of attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)

    Example:
        @retry_on_error(exceptions=(httpx.HTTPError,), attempts=3)
        async def fetch_data():
            ...
    """
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.DEBUG),
        reraise=True,
    )


def retry_llm_call(func: Callable) -> Callable:
    """
    Decorator specifically for LLM API calls.

    Retries on common transient errors:
    - Connection errors
    - Rate limits (with longer backoff)
    - Server errors (5xx)
    """
    import httpx

    @wraps(func)
    async def wrapper(*args, **kwargs):
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=1, max=30),
            retry=retry_if_exception_type((
                httpx.HTTPError,
                httpx.TimeoutException,
                ConnectionError,
            )),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _inner():
            return await func(*args, **kwargs)

        return await _inner()

    return wrapper


def retry_mcp_call(func: Callable) -> Callable:
    """
    Decorator for MCP tool calls.

    Handles:
    - Connection drops
    - Timeout on tool execution
    - Server unavailability
    """
    from insights.core.errors import MCPConnectionError, MCPTimeoutError

    @wraps(func)
    async def wrapper(*args, **kwargs):
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=1, max=10),
            retry=retry_if_exception_type((
                MCPConnectionError,
                MCPTimeoutError,
                ConnectionError,
                TimeoutError,
            )),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _inner():
            return await func(*args, **kwargs)

        return await _inner()

    return wrapper


def retry_db_call(func: Callable) -> Callable:
    """
    Decorator for database operations.

    Handles transient database errors with short retry.
    """
    from insights.core.errors import DatabaseConnectionError

    @wraps(func)
    async def wrapper(*args, **kwargs):
        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(min=0.5, max=2),
            retry=retry_if_exception_type((
                DatabaseConnectionError,
                ConnectionError,
            )),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _inner():
            return await func(*args, **kwargs)

        return await _inner()

    return wrapper
