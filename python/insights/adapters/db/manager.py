"""
Database manager for InSights-ai.

Provides async interface to Supabase for all database operations.

Note: Many type: ignore comments are needed due to Supabase's dynamic JSON typing.
"""
from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from supabase import Client, create_client

from insights.core.config import settings
from insights.core.errors import DatabaseConnectionError, DatabaseQueryError

# ─────────────────────────────────────────────────────────────
# Data Models for Database Operations
# ─────────────────────────────────────────────────────────────

class CompanyCreate(BaseModel):
    """Data for creating/upserting a company."""
    ticker: str
    cik: str
    name: str | None = None
    sector: str | None = None
    exchange: str | None = None


class CompanyRecord(BaseModel):
    """Company record from database."""
    id: UUID
    ticker: str
    cik: str
    name: str | None = None
    sector: str | None = None
    exchange: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class FilingCreate(BaseModel):
    """Data for creating a filing."""
    company_id: UUID
    accession_number: str
    form_type: str
    filing_date: date
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    report_url: str | None = None
    raw_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FilingRecord(BaseModel):
    """Filing record from database."""
    id: UUID
    company_id: UUID
    accession_number: str
    form_type: str
    filing_date: date
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    report_url: str | None = None
    raw_text: str | None = None
    created_at: datetime | None = None


class RiskFactorCreate(BaseModel):
    """Data for creating a risk factor."""
    filing_id: UUID
    title: str
    content: str
    rank: int
    embedding: list[float] | None = None
    word_count: int | None = None


class RiskFactorRecord(BaseModel):
    """Risk factor record from database."""
    id: UUID
    filing_id: UUID
    title: str
    content: str
    rank: int
    embedding: list[float] | None = None
    word_count: int | None = None
    created_at: datetime | None = None


class RiskDriftCreate(BaseModel):
    """Data for creating a risk drift record."""
    company_id: UUID
    risk_factor_id: UUID
    prev_factor_id: UUID | None = None
    rank_current: int
    rank_prev: int | None = None
    rank_delta: int | None = None
    semantic_score: float | None = None
    fuzzy_score: int | None = None
    drift_type: str
    zone: str
    heat_score: int = 0
    modality_shift: str = "none"
    analysis: str | None = None
    strategic_recommendation: str | None = None
    original_text_snippet: str | None = None
    new_text_snippet: str | None = None
    materiality: str | None = None
    confidence_score: float = 0.9


class JobCreate(BaseModel):
    """Data for creating a job."""
    job_type: str
    request_payload: dict[str, Any]
    user_id: UUID | None = None
    webhook_url: str | None = None


class JobRecord(BaseModel):
    """Job record from database."""
    id: UUID
    job_type: str
    request_payload: dict[str, Any]
    status: str
    progress: int = 0
    current_step: str | None = None
    result_summary: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class AuditLogCreate(BaseModel):
    """Data for creating an audit log entry."""
    job_id: UUID | None = None
    operation: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    success: bool = True
    error_message: str | None = None


class VectorSearchResult(BaseModel):
    """Result from vector similarity search."""
    id: UUID
    filing_id: UUID
    title: str
    content: str
    rank: int
    similarity: float


# ─────────────────────────────────────────────────────────────
# Database Manager
# ─────────────────────────────────────────────────────────────

class DBManager:
    """
    Database manager for Supabase operations.

    Uses service role key for backend operations (bypasses RLS).
    """

    _instance: "DBManager | None" = None

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
    ):
        """
        Initialize database manager.

        Args:
            supabase_url: Supabase project URL (defaults to settings)
            supabase_key: Supabase service role key (defaults to settings)
        """
        self._url = supabase_url or settings.SUPABASE_URL
        self._key = supabase_key or settings.SUPABASE_SERVICE_ROLE_KEY
        self._client: Client | None = None

    @property
    def client(self) -> Client:
        """Get Supabase client (lazy initialization)."""
        if self._client is None:
            try:
                self._client = create_client(self._url, self._key)
            except Exception as e:
                raise DatabaseConnectionError(f"Failed to connect to Supabase: {e}") from e
        return self._client

    @classmethod
    def get_instance(cls) -> "DBManager":
        """Get singleton instance of DBManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ─────────────────────────────────────────────────────────
    # Company Operations
    # ─────────────────────────────────────────────────────────

    async def get_or_create_company(self, data: CompanyCreate) -> UUID:
        """
        Upsert company by ticker, return UUID.

        Args:
            data: Company data to upsert

        Returns:
            UUID of the company record
        """
        try:
            result = self.client.table("companies").upsert(
                data.model_dump(exclude_none=True),
                on_conflict="ticker"
            ).execute()

            if not result.data:
                raise DatabaseQueryError("No data returned from company upsert")

            return UUID(result.data[0]["id"])
        except Exception as e:
            if "DatabaseQueryError" in str(type(e)):
                raise
            raise DatabaseQueryError(f"Failed to upsert company: {e}") from e

    async def get_company_by_ticker(self, ticker: str) -> CompanyRecord | None:
        """Get company by ticker symbol."""
        try:
            result = self.client.table("companies")\
                .select("*")\
                .eq("ticker", ticker.upper())\
                .limit(1)\
                .execute()

            if result.data:
                return CompanyRecord(**result.data[0])
            return None
        except Exception as e:
            raise DatabaseQueryError(f"Failed to get company: {e}") from e

    async def get_company_by_id(self, company_id: UUID) -> CompanyRecord | None:
        """Get company by ID."""
        try:
            result = self.client.table("companies")\
                .select("*")\
                .eq("id", str(company_id))\
                .limit(1)\
                .execute()

            if result.data:
                return CompanyRecord(**result.data[0])
            return None
        except Exception as e:
            raise DatabaseQueryError(f"Failed to get company: {e}") from e

    # ─────────────────────────────────────────────────────────
    # Filing Operations
    # ─────────────────────────────────────────────────────────

    async def save_filing(self, data: FilingCreate) -> UUID:
        """
        Insert or update a filing.

        Uses upsert on accession_number for idempotency.
        """
        try:
            payload = data.model_dump(mode="json", exclude_none=True)
            payload["company_id"] = str(data.company_id)

            result = self.client.table("filings").upsert(
                payload,
                on_conflict="accession_number"
            ).execute()

            if not result.data:
                raise DatabaseQueryError("No data returned from filing upsert")

            return UUID(result.data[0]["id"])
        except Exception as e:
            if "DatabaseQueryError" in str(type(e)):
                raise
            raise DatabaseQueryError(f"Failed to save filing: {e}") from e

    async def get_filing_by_accession(
        self,
        accession_number: str
    ) -> FilingRecord | None:
        """Get filing by accession number."""
        try:
            result = self.client.table("filings")\
                .select("*")\
                .eq("accession_number", accession_number)\
                .limit(1)\
                .execute()

            if result.data:
                return FilingRecord(**result.data[0])
            return None
        except Exception as e:
            raise DatabaseQueryError(f"Failed to get filing: {e}") from e

    async def get_filings_by_company(
        self,
        company_id: UUID,
        form_type: str | None = None,
        years: list[int] | None = None,
        limit: int = 10,
    ) -> list[FilingRecord]:
        """Get filings for a company with optional filters."""
        try:
            query = self.client.table("filings")\
                .select("*")\
                .eq("company_id", str(company_id))\
                .order("filing_date", desc=True)\
                .limit(limit)

            if form_type:
                query = query.eq("form_type", form_type)

            if years:
                query = query.in_("fiscal_year", years)

            result = query.execute()
            return [FilingRecord(**r) for r in result.data]
        except Exception as e:
            raise DatabaseQueryError(f"Failed to get filings: {e}") from e

    # ─────────────────────────────────────────────────────────
    # Risk Factor Operations
    # ─────────────────────────────────────────────────────────

    async def save_risk_factors(
        self,
        factors: list[RiskFactorCreate],
    ) -> list[UUID]:
        """
        Batch insert risk factors.

        Args:
            factors: List of risk factors to insert

        Returns:
            List of created UUIDs
        """
        if not factors:
            return []

        try:
            records = []
            for f in factors:
                record = f.model_dump(mode="json", exclude_none=True)
                record["filing_id"] = str(f.filing_id)
                records.append(record)

            result = self.client.table("risk_factors")\
                .insert(records)\
                .execute()

            return [UUID(r["id"]) for r in result.data]
        except Exception as e:
            raise DatabaseQueryError(f"Failed to save risk factors: {e}") from e

    async def get_risk_factors_by_filing(
        self,
        filing_id: UUID,
    ) -> list[RiskFactorRecord]:
        """Get all risk factors for a filing ordered by rank."""
        try:
            result = self.client.table("risk_factors")\
                .select("*")\
                .eq("filing_id", str(filing_id))\
                .order("rank")\
                .execute()

            return [RiskFactorRecord(**r) for r in result.data]
        except Exception as e:
            raise DatabaseQueryError(f"Failed to get risk factors: {e}") from e

    async def update_risk_factor_embedding(
        self,
        risk_factor_id: UUID,
        embedding: list[float],
    ) -> None:
        """Update embedding for a risk factor."""
        try:
            self.client.table("risk_factors")\
                .update({"embedding": embedding})\
                .eq("id", str(risk_factor_id))\
                .execute()
        except Exception as e:
            raise DatabaseQueryError(f"Failed to update embedding: {e}") from e

    # ─────────────────────────────────────────────────────────
    # Risk Drift Operations
    # ─────────────────────────────────────────────────────────

    async def save_risk_drifts(
        self,
        drifts: list[RiskDriftCreate],
    ) -> list[UUID]:
        """Batch insert risk drift records."""
        if not drifts:
            return []

        try:
            records = []
            for d in drifts:
                record = d.model_dump(mode="json", exclude_none=True)
                record["company_id"] = str(d.company_id)
                record["risk_factor_id"] = str(d.risk_factor_id)
                if d.prev_factor_id:
                    record["prev_factor_id"] = str(d.prev_factor_id)
                records.append(record)

            result = self.client.table("risk_drifts")\
                .insert(records)\
                .execute()

            return [UUID(r["id"]) for r in result.data]
        except Exception as e:
            raise DatabaseQueryError(f"Failed to save risk drifts: {e}") from e

    async def get_risk_drifts_by_company(
        self,
        company_id: UUID,
        zone: str | None = None,
        min_heat_score: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get risk drifts for a company with optional filters."""
        try:
            query = self.client.table("risk_drifts")\
                .select("*, risk_factors!risk_factor_id(*)")\
                .eq("company_id", str(company_id))\
                .gte("heat_score", min_heat_score)\
                .order("heat_score", desc=True)\
                .limit(limit)

            if zone:
                query = query.eq("zone", zone)

            result = query.execute()
            return result.data
        except Exception as e:
            raise DatabaseQueryError(f"Failed to get risk drifts: {e}") from e

    # ─────────────────────────────────────────────────────────
    # Vector Search Operations
    # ─────────────────────────────────────────────────────────

    async def vector_search_risk_factors(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[VectorSearchResult]:
        """
        Semantic search across risk factors using pgvector.

        Args:
            query_embedding: 768-dim embedding vector
            limit: Maximum results to return
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of matching risk factors with similarity scores
        """
        try:
            result = self.client.rpc(
                "match_risk_factors",
                {
                    "query_embedding": query_embedding,
                    "match_count": limit,
                    "similarity_threshold": threshold,
                }
            ).execute()

            return [VectorSearchResult(**r) for r in result.data]
        except Exception as e:
            raise DatabaseQueryError(f"Vector search failed: {e}") from e

    async def vector_search_filing_chunks(
        self,
        query_embedding: list[float],
        filing_id: UUID | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Semantic search across filing chunks."""
        try:
            params: dict[str, Any] = {
                "query_embedding": query_embedding,
                "match_count": limit,
            }
            if filing_id:
                params["p_filing_id"] = str(filing_id)

            result = self.client.rpc(
                "match_filing_chunks",
                params
            ).execute()

            return result.data
        except Exception as e:
            raise DatabaseQueryError(f"Vector search failed: {e}") from e

    # ─────────────────────────────────────────────────────────
    # Filing Embeddings Operations
    # ─────────────────────────────────────────────────────────

    async def save_filing_embeddings(
        self,
        filing_id: UUID,
        chunks: list[dict[str, Any]],
    ) -> list[UUID]:
        """
        Save chunked embeddings for a filing.

        Args:
            filing_id: Filing UUID
            chunks: List of dicts with content, embedding, section_type, chunk_index
        """
        if not chunks:
            return []

        try:
            records = []
            for chunk in chunks:
                records.append({
                    "filing_id": str(filing_id),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "content": chunk.get("content", ""),
                    "section_type": chunk.get("section_type"),
                    "embedding": chunk.get("embedding"),
                    "token_count": chunk.get("token_count"),
                })

            result = self.client.table("filing_embeddings")\
                .upsert(records, on_conflict="filing_id,chunk_index")\
                .execute()

            return [UUID(r["id"]) for r in result.data]
        except Exception as e:
            raise DatabaseQueryError(f"Failed to save embeddings: {e}") from e

    # ─────────────────────────────────────────────────────────
    # Job Queue Operations
    # ─────────────────────────────────────────────────────────

    async def create_job(self, data: JobCreate) -> UUID:
        """Create a new job in the queue."""
        try:
            payload = {
                "job_type": data.job_type,
                "request_payload": data.request_payload,
                "status": "queued",
            }
            if data.user_id:
                payload["user_id"] = str(data.user_id)
            if data.webhook_url:
                payload["webhook_url"] = data.webhook_url

            result = self.client.table("job_queue")\
                .insert(payload)\
                .execute()

            if not result.data:
                raise DatabaseQueryError("No data returned from job insert")

            return UUID(result.data[0]["id"])
        except Exception as e:
            if "DatabaseQueryError" in str(type(e)):
                raise
            raise DatabaseQueryError(f"Failed to create job: {e}") from e

    async def get_job(self, job_id: UUID) -> JobRecord | None:
        """Get job by ID."""
        try:
            result = self.client.table("job_queue")\
                .select("*")\
                .eq("id", str(job_id))\
                .limit(1)\
                .execute()

            if result.data:
                return JobRecord(**result.data[0])
            return None
        except Exception as e:
            raise DatabaseQueryError(f"Failed to get job: {e}") from e

    async def update_job_status(
        self,
        job_id: UUID,
        status: str,
        progress: int | None = None,
        current_step: str | None = None,
        result_summary: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update job status and progress."""
        try:
            update_data: dict[str, Any] = {"status": status}

            if progress is not None:
                update_data["progress"] = progress
            if current_step is not None:
                update_data["current_step"] = current_step
            if result_summary is not None:
                update_data["result_summary"] = result_summary
            if error_message is not None:
                update_data["error_message"] = error_message

            # Set timing fields based on status
            now = datetime.now(tz=None).astimezone().isoformat()
            if status == "processing":
                update_data["started_at"] = now
            elif status in ("completed", "failed", "cancelled"):
                update_data["completed_at"] = now

            self.client.table("job_queue")\
                .update(update_data)\
                .eq("id", str(job_id))\
                .execute()
        except Exception as e:
            raise DatabaseQueryError(f"Failed to update job: {e}") from e

    async def get_pending_jobs(
        self,
        job_type: str | None = None,
        limit: int = 10,
    ) -> list[JobRecord]:
        """Get pending jobs for processing."""
        try:
            query = self.client.table("job_queue")\
                .select("*")\
                .eq("status", "queued")\
                .order("created_at")\
                .limit(limit)

            if job_type:
                query = query.eq("job_type", job_type)

            result = query.execute()
            return [JobRecord(**r) for r in result.data]
        except Exception as e:
            raise DatabaseQueryError(f"Failed to get pending jobs: {e}") from e

    # ─────────────────────────────────────────────────────────
    # Reports Operations
    # ─────────────────────────────────────────────────────────

    async def save_report(
        self,
        company_id: UUID,
        title: str,
        report_type: str,
        markdown_content: str,
        job_id: UUID | None = None,
        parameters: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> UUID:
        """Save a generated report."""
        try:
            payload: dict[str, Any] = {
                "company_id": str(company_id),
                "title": title,
                "report_type": report_type,
                "markdown_content": markdown_content,
            }
            if job_id:
                payload["job_id"] = str(job_id)
            if parameters:
                payload["parameters"] = parameters
            if summary:
                payload["summary"] = summary

            result = self.client.table("reports")\
                .insert(payload)\
                .execute()

            return UUID(result.data[0]["id"])
        except Exception as e:
            raise DatabaseQueryError(f"Failed to save report: {e}") from e

    async def get_reports_by_company(
        self,
        company_id: UUID,
        report_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get reports for a company."""
        try:
            query = self.client.table("reports")\
                .select("*")\
                .eq("company_id", str(company_id))\
                .order("created_at", desc=True)\
                .limit(limit)

            if report_type:
                query = query.eq("report_type", report_type)

            result = query.execute()
            return result.data
        except Exception as e:
            raise DatabaseQueryError(f"Failed to get reports: {e}") from e

    # ─────────────────────────────────────────────────────────
    # Audit Log Operations
    # ─────────────────────────────────────────────────────────

    async def log_llm_usage(self, data: AuditLogCreate) -> UUID:
        """Log LLM usage for auditing and cost tracking."""
        try:
            payload = data.model_dump(exclude_none=True)
            if data.job_id:
                payload["job_id"] = str(data.job_id)

            result = self.client.table("audit_logs")\
                .insert(payload)\
                .execute()

            return UUID(result.data[0]["id"])
        except Exception as e:
            raise DatabaseQueryError(f"Failed to log usage: {e}") from e

    async def get_usage_summary(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """Get usage summary by model and date."""
        try:
            # Use the materialized view for aggregated stats
            # Refresh it first
            self.client.rpc("refresh_materialized_view", {
                "view_name": "daily_usage"
            }).execute()

            query = self.client.table("daily_usage")\
                .select("*")\
                .order("date", desc=True)

            if start_date:
                query = query.gte("date", start_date.isoformat())
            if end_date:
                query = query.lte("date", end_date.isoformat())

            result = query.execute()
            return result.data
        except Exception as e:
            # Fallback to direct query if materialized view fails
            raise DatabaseQueryError(f"Failed to get usage summary: {e}") from e


# ─────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────

def get_db() -> DBManager:
    """Get the singleton database manager instance."""
    return DBManager.get_instance()
