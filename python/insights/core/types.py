"""
Shared Pydantic models for InSights-ai.

These types are used across the application for consistent data structures.
"""
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    """Status of an analysis job."""
    PENDING = "queued"
    RUNNING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RiskZone(str, Enum):
    """Risk classification zones."""
    CRITICAL_RED = "critical_red"
    WARNING_ORANGE = "warning_orange"
    NEW_BLUE = "new_blue"
    STABLE_GRAY = "stable_gray"


class SentimentLabel(str, Enum):
    """Sentiment classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# ─────────────────────────────────────────────────────────────
# Core Models
# ─────────────────────────────────────────────────────────────

class Company(BaseModel):
    """Company information."""
    id: str | None = None
    ticker: str
    name: str
    cik: str | None = None
    sector: str | None = None
    industry: str | None = None


class Filing(BaseModel):
    """SEC filing metadata."""
    id: str | None = None
    company_id: str
    accession_number: str
    form_type: str
    filing_date: datetime
    fiscal_year: int
    fiscal_period: str | None = None
    url: str | None = None


class RiskFactor(BaseModel):
    """Extracted risk factor from a filing."""
    id: str | None = None
    filing_id: str
    title: str
    content: str
    section_order: int = 0
    word_count: int = 0
    embedding: list[float] | None = None


# ─────────────────────────────────────────────────────────────
# Expert System Models
# ─────────────────────────────────────────────────────────────

class ExpertResult(BaseModel):
    """Result from an expert analysis."""
    expert_id: str
    findings: str
    confidence: float = Field(ge=0, le=1)
    sources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "expert_id": "risk_analyst",
                "findings": "Key risks identified...",
                "confidence": 0.85,
                "sources": ["10-K FY2024", "8-K Q3"],
            }
        }
    )


class SynthesizedReport(BaseModel):
    """Final synthesized report from Orchestrator."""
    company: Company
    summary: str
    expert_findings: list[ExpertResult]
    recommendations: list[str] = Field(default_factory=list)
    risk_level: str = "medium"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ─────────────────────────────────────────────────────────────
# Risk Analysis Models
# ─────────────────────────────────────────────────────────────

class DriftResult(BaseModel):
    """Result of drift calculation between two risk factors."""
    old_title: str
    new_title: str
    similarity: float = Field(ge=0, le=1)
    is_material: bool
    zone: RiskZone
    heat_score: float = Field(ge=0, le=1)
    sentiment_delta: float = Field(ge=-2, le=2)


class SentimentResult(BaseModel):
    """Sentiment analysis result."""
    label: SentimentLabel
    score: float = Field(ge=0, le=1)
    positive: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)


class HeatmapEntry(BaseModel):
    """Single entry in a risk heatmap."""
    title: str
    zone: RiskZone
    heat_score: float
    old_sentiment: SentimentResult | None = None
    new_sentiment: SentimentResult | None = None
    old_snippet: str | None = None
    new_snippet: str | None = None


class HeatmapData(BaseModel):
    """Complete heatmap data for a company."""
    company: Company
    period_old: str  # e.g., "FY2023"
    period_new: str  # e.g., "FY2024"
    entries: list[HeatmapEntry]
    summary: dict[str, int] = Field(default_factory=dict)  # Zone counts


# ─────────────────────────────────────────────────────────────
# Job Models
# ─────────────────────────────────────────────────────────────

class Job(BaseModel):
    """Analysis job."""
    id: str
    ticker: str
    years: list[int]
    status: JobStatus = JobStatus.PENDING
    progress: float = Field(ge=0, le=1, default=0)
    current_step: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class JobProgress(BaseModel):
    """Progress update for a job (sent via SSE)."""
    job_id: str
    status: JobStatus
    progress: float
    step: str | None = None
    message: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ─────────────────────────────────────────────────────────────
# API Models
# ─────────────────────────────────────────────────────────────

class AnalysisOptions(BaseModel):
    """Configuration options for analysis."""
    include_sentiment: bool = True
    generate_heatmap: bool = True
    depth: str = "deep"  # shallow or deep


class AnalyzeRequest(BaseModel):
    """Request to start analysis."""
    ticker: str = Field(..., min_length=1, max_length=10)
    years: list[int] = Field(default_factory=lambda: [2024, 2023])
    analysis_type: str = "risk_drift"
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)
    webhook_url: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ticker": "AAPL",
                "years": [2024, 2023],
                "analysis_type": "risk_drift",
                "options": {
                    "include_sentiment": True,
                    "generate_heatmap": True,
                    "depth": "deep"
                }
            }
        }
    )


class AnalyzeResponse(BaseModel):
    """Response after starting analysis."""
    job_id: str
    status: JobStatus
    message: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    code: str
    detail: str | None = None
