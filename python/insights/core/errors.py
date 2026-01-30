"""
Custom exceptions for InSights-ai.

All exceptions inherit from InsightsError for consistent error handling.
"""


class InsightsError(Exception):
    """Base exception for all InSights-ai errors."""

    def __init__(self, message: str, code: str | None = None):
        self.message = message
        self.code = code or self.__class__.__name__
        super().__init__(self.message)


# ─────────────────────────────────────────────────────────────
# Infrastructure Errors
# ─────────────────────────────────────────────────────────────

class DatabaseError(InsightsError):
    """Database operation failed."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Failed to connect to database."""
    pass


class DatabaseQueryError(DatabaseError):
    """Query execution failed."""
    pass


# ─────────────────────────────────────────────────────────────
# MCP Errors
# ─────────────────────────────────────────────────────────────

class MCPError(InsightsError):
    """MCP-related error."""
    pass


class MCPConnectionError(MCPError):
    """Failed to connect to MCP server."""
    pass


class MCPToolError(MCPError):
    """MCP tool execution failed."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class MCPTimeoutError(MCPError):
    """MCP operation timed out."""
    pass


# ─────────────────────────────────────────────────────────────
# LLM Errors
# ─────────────────────────────────────────────────────────────

class LLMError(InsightsError):
    """LLM provider error."""
    pass


class LLMProviderError(LLMError):
    """LLM provider returned an error."""

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] {message}")


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""
    pass


class LLMContextLengthError(LLMError):
    """Input exceeds model context length."""
    pass


class CircuitBreakerOpenError(LLMError):
    """Circuit breaker is open, provider unavailable."""
    pass


class ModelNotFoundError(LLMError):
    """Requested model not found."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' not found")


class ProviderNotFoundError(LLMError):
    """Requested provider not found."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"Provider '{provider}' not found")


# ─────────────────────────────────────────────────────────────
# Sentiment Errors
# ─────────────────────────────────────────────────────────────

class SentimentServiceError(InsightsError):
    """FinBERT service error."""
    pass


class SentimentModelError(SentimentServiceError):
    """FinBERT model inference failed."""
    pass


# ─────────────────────────────────────────────────────────────
# Expert Errors
# ─────────────────────────────────────────────────────────────

class ExpertError(InsightsError):
    """Expert-related error."""
    pass


class ExpertNotFoundError(ExpertError):
    """Requested expert not found in registry."""

    def __init__(self, expert_id: str):
        self.expert_id = expert_id
        super().__init__(f"Expert '{expert_id}' not found in registry")


class ExpertExecutionError(ExpertError):
    """Expert failed during execution."""

    def __init__(self, expert_id: str, message: str):
        self.expert_id = expert_id
        super().__init__(f"Expert '{expert_id}' failed: {message}")


# ─────────────────────────────────────────────────────────────
# Job Errors
# ─────────────────────────────────────────────────────────────

class JobError(InsightsError):
    """Job processing error."""
    pass


class JobNotFoundError(JobError):
    """Job not found."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__(f"Job '{job_id}' not found")


class JobAlreadyExistsError(JobError):
    """Job with same parameters already exists."""
    pass


# ─────────────────────────────────────────────────────────────
# Validation Errors
# ─────────────────────────────────────────────────────────────

class ValidationError(InsightsError):
    """Input validation failed."""
    pass


class InvalidTickerError(ValidationError):
    """Invalid stock ticker."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        super().__init__(f"Invalid ticker: '{ticker}'")
