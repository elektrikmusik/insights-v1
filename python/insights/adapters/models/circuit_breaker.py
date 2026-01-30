"""
Circuit Breaker for LLM providers.

Prevents cascading failures by temporarily disabling
failing providers/models.
"""
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"   # Normal operation
    OPEN = "open"       # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    reset_timeout: int = 300   # seconds


class CircuitBreaker:
    """
    Manages circuit states for different keys (provider/model).
    """

    _instance: Optional["CircuitBreaker"] = None

    def __init__(self):
        self._states: dict[str, CircuitState] = {}
        self._failures: dict[str, int] = {}
        self._last_failure_time: dict[str, float] = {}
        self._default_config = CircuitBreakerConfig()

    @classmethod
    def get_instance(cls) -> "CircuitBreaker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_state(self, key: str) -> CircuitState:
        """Get current state for a key, checking recovery timeout."""
        state = self._states.get(key, CircuitState.CLOSED)

        if state == CircuitState.OPEN:
            last_fail = self._last_failure_time.get(key, 0)
            if time.time() - last_fail > self._default_config.recovery_timeout:
                logger.info(f"Circuit {key} entering HALF_OPEN state")
                return CircuitState.HALF_OPEN

        return state

    def record_success(self, key: str):
        """Record a success, resetting failure count if needed."""
        if self.get_state(key) in (CircuitState.OPEN, CircuitState.HALF_OPEN):
            logger.info(f"Circuit {key} recovered (CLOSED)")
            self._states[key] = CircuitState.CLOSED
            self._failures[key] = 0

    def record_failure(self, key: str):
        """Record a failure, potentially opening the circuit."""
        self._failures[key] = self._failures.get(key, 0) + 1
        self._last_failure_time[key] = time.time()

        current_failures = self._failures[key]
        if (
            self.get_state(key) == CircuitState.HALF_OPEN
            or current_failures >= self._default_config.failure_threshold
        ):
            if self._states.get(key) != CircuitState.OPEN:
                logger.warning(f"Circuit {key} OPENED after {current_failures} failures")
                self._states[key] = CircuitState.OPEN

    def is_available(self, key: str) -> bool:
        """Check if a service is available."""
        return self.get_state(key) != CircuitState.OPEN


def get_circuit_breaker() -> CircuitBreaker:
    return CircuitBreaker.get_instance()
