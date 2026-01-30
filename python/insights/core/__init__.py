"""
InSights-ai Core Module.

Provides configuration, types, errors, and utilities.
"""
from insights.core.config import settings
from insights.core.errors import (
    DatabaseError,
    ExpertError,
    InsightsError,
    JobError,
    LLMError,
    MCPError,
    ValidationError,
)
from insights.core.types import (
    AnalyzeRequest,
    AnalyzeResponse,
    Company,
    DriftResult,
    ErrorResponse,
    ExpertResult,
    Filing,
    HeatmapData,
    HeatmapEntry,
    Job,
    JobProgress,
    JobStatus,
    RiskFactor,
    RiskZone,
    SentimentLabel,
    SentimentResult,
    SynthesizedReport,
)

__all__ = [
    # Config
    "settings",
    # Enums
    "JobStatus",
    "RiskZone",
    "SentimentLabel",
    # Models
    "Company",
    "Filing",
    "RiskFactor",
    "ExpertResult",
    "SynthesizedReport",
    "DriftResult",
    "SentimentResult",
    "HeatmapEntry",
    "HeatmapData",
    "Job",
    "JobProgress",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "ErrorResponse",
    # Errors
    "InsightsError",
    "DatabaseError",
    "MCPError",
    "LLMError",
    "ExpertError",
    "JobError",
    "ValidationError",
]
