# 02. Database Schema (Supabase)

## Overview

InSights-ai uses Supabase for two distinct purposes:
1. **Relational Data**: Companies, filings, analysis results, job tracking
2. **Vector Store**: Embeddings for semantic search (pgvector)

This schema supports:
- Multi-filing comparison (risk drift analysis)
- Async job processing with status tracking
- LLM usage auditing for cost management
- Version-controlled migrations via Alembic

---

## Entity Relationship Diagram

```
┌─────────────────────┐       ┌─────────────────────┐
│     companies       │       │       filings       │
├─────────────────────┤       ├─────────────────────┤
│ id (PK)             │◄──────│ company_id (FK)     │
│ ticker              │       │ id (PK)             │
│ cik                 │       │ accession_number    │
│ name                │       │ form_type           │
│ sector              │       │ filing_date         │
│ created_at          │       │ fiscal_year         │
└─────────────────────┘       │ raw_text            │
                              │ created_at          │
                              └─────────┬───────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │   risk_factors    │ │ filing_embeddings │ │  analysis_chunks  │
        ├───────────────────┤ ├───────────────────┤ ├───────────────────┤
        │ id (PK)           │ │ id (PK)           │ │ id (PK)           │
        │ filing_id (FK)    │ │ filing_id (FK)    │ │ filing_id (FK)    │
        │ title             │ │ chunk_index       │ │ chunk_index       │
        │ content           │ │ content           │ │ content           │
        │ rank              │ │ embedding (768)   │ │ sentiment_label   │
        │ embedding (768)   │ │ section_type      │ │ sentiment_score   │
        └─────────┬─────────┘ └───────────────────┘ │ embedding (768)   │
                  │                                 └───────────────────┘
                  │
                  ▼
        ┌───────────────────┐       ┌───────────────────┐
        │    risk_drifts    │       │      reports      │
        ├───────────────────┤       ├───────────────────┤
        │ id (PK)           │       │ id (PK)           │
        │ company_id (FK)   │       │ company_id (FK)   │
        │ risk_factor_id(FK)│       │ title             │
        │ prev_factor_id(FK)│       │ markdown_content  │
        │ rank_current      │       │ job_id (FK)       │
        │ rank_prev         │       │ created_at        │
        │ rank_delta        │       └───────────────────┘
        │ semantic_score    │
        │ drift_type        │       ┌───────────────────┐
        │ heat_score        │       │     job_queue     │
        │ zone              │       ├───────────────────┤
        │ analysis          │       │ id (PK)           │
        │ created_at        │       │ status            │
        └───────────────────┘       │ request_payload   │
                                    │ result_summary    │
        ┌───────────────────┐       │ error_message     │
        │    audit_logs     │       │ webhook_url       │
        ├───────────────────┤       │ created_at        │
        │ id (PK)           │       │ started_at        │
        │ job_id (FK)       │       │ completed_at      │
        │ model_id          │       └───────────────────┘
        │ provider          │
        │ input_tokens      │
        │ output_tokens     │
        │ cost_usd          │
        │ latency_ms        │
        │ created_at        │
        └───────────────────┘
```

---

## SQL Schema

Execute the following in the Supabase SQL Editor or via Alembic migration.

```sql
-- ============================================================
-- Enable Extensions
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search optimization

-- ============================================================
-- 1. Companies Reference
-- ============================================================
CREATE TABLE companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker TEXT UNIQUE NOT NULL,
    cik TEXT NOT NULL,
    name TEXT,
    sector TEXT,
    exchange TEXT,  -- NYSE, NASDAQ, etc.
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_companies_ticker ON companies(ticker);
CREATE INDEX idx_companies_cik ON companies(cik);

-- ============================================================
-- 2. Filings (Source of Truth)
-- ============================================================
CREATE TABLE filings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    accession_number TEXT UNIQUE NOT NULL,
    form_type TEXT NOT NULL,  -- '10-K', '10-Q', '8-K'
    filing_date DATE NOT NULL,
    fiscal_year INTEGER,
    fiscal_period TEXT,  -- 'FY', 'Q1', 'Q2', 'Q3', 'Q4'
    report_url TEXT,
    raw_text TEXT,  -- Full extracted text from MCP
    metadata JSONB DEFAULT '{}',  -- Additional SEC metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_filings_company ON filings(company_id);
CREATE INDEX idx_filings_date ON filings(filing_date DESC);
CREATE INDEX idx_filings_type ON filings(form_type);
CREATE UNIQUE INDEX idx_filings_company_year ON filings(company_id, form_type, fiscal_year);

-- ============================================================
-- 3. Risk Factors (Extracted from Filings)
-- ============================================================
CREATE TABLE risk_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_id UUID NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    rank INTEGER NOT NULL,  -- Order in the filing (1-indexed)
    embedding VECTOR(768),  -- Google text-embedding-004 dimensions
    word_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_risk_factors_filing ON risk_factors(filing_id);
CREATE INDEX idx_risk_factors_rank ON risk_factors(filing_id, rank);

-- Vector similarity search index
CREATE INDEX idx_risk_factors_embedding ON risk_factors 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);

-- ============================================================
-- 4. Risk Drifts (Comparison Results)
-- ============================================================
CREATE TABLE risk_drifts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    
    -- Current filing risk
    risk_factor_id UUID NOT NULL REFERENCES risk_factors(id) ON DELETE CASCADE,
    
    -- Previous filing risk (NULL if new)
    prev_factor_id UUID REFERENCES risk_factors(id) ON DELETE SET NULL,
    
    -- Rank analysis
    rank_current INTEGER NOT NULL,
    rank_prev INTEGER,
    rank_delta INTEGER,  -- prev_rank - current_rank (positive = climbed up)
    
    -- Semantic analysis
    semantic_score FLOAT,  -- Cosine similarity (0.0 to 1.0)
    fuzzy_score INTEGER,   -- Title match score (0-100)
    
    -- Classification
    drift_type TEXT NOT NULL,  -- 'structural', 'semantic', 'new', 'removed', 'stable'
    zone TEXT NOT NULL,        -- 'critical_red', 'warning_orange', 'new_blue', 'stable_gray'
    heat_score INTEGER NOT NULL DEFAULT 0,  -- 0-100
    
    -- Text analysis
    modality_shift TEXT DEFAULT 'none',  -- 'probabilistic_to_deterministic', etc.
    
    -- LLM-generated content
    analysis TEXT,
    strategic_recommendation TEXT,
    
    -- Snippets for UI display
    original_text_snippet TEXT,
    new_text_snippet TEXT,
    
    -- Materiality flags
    materiality TEXT,
    confidence_score FLOAT DEFAULT 0.9,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_risk_drifts_company ON risk_drifts(company_id);
CREATE INDEX idx_risk_drifts_zone ON risk_drifts(zone);
CREATE INDEX idx_risk_drifts_heat ON risk_drifts(heat_score DESC);

-- ============================================================
-- 5. Filing Embeddings (Chunked for RAG)
-- ============================================================
CREATE TABLE filing_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_id UUID NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    section_type TEXT,  -- 'risk_factors', 'business', 'mda', etc.
    embedding VECTOR(768),
    token_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(filing_id, chunk_index)
);

CREATE INDEX idx_filing_embeddings_filing ON filing_embeddings(filing_id);
CREATE INDEX idx_filing_embeddings_section ON filing_embeddings(section_type);

-- Vector similarity search index
CREATE INDEX idx_filing_embeddings_vector ON filing_embeddings 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);

-- ============================================================
-- 6. Analysis Chunks (With FinBERT Sentiment)
-- ============================================================
CREATE TABLE analysis_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_id UUID NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    
    -- FinBERT Analysis Results
    sentiment_label TEXT,  -- 'positive', 'negative', 'neutral'
    sentiment_score FLOAT,  -- 0.0 to 1.0 (confidence)
    sentiment_logits JSONB,  -- Raw logits for debugging
    
    -- Semantic Search
    embedding VECTOR(768),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(filing_id, chunk_index)
);

CREATE INDEX idx_analysis_chunks_filing ON analysis_chunks(filing_id);
CREATE INDEX idx_analysis_chunks_sentiment ON analysis_chunks(sentiment_label);

-- Vector index
CREATE INDEX idx_analysis_chunks_vector ON analysis_chunks 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);

-- ============================================================
-- 7. Job Queue (Async Processing)
-- ============================================================
CREATE TYPE job_status AS ENUM ('queued', 'processing', 'completed', 'failed', 'cancelled');

CREATE TABLE job_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Job definition
    job_type TEXT NOT NULL,  -- 'risk_analysis', 'sentiment_batch', etc.
    request_payload JSONB NOT NULL,
    
    -- Status tracking
    status job_status NOT NULL DEFAULT 'queued',
    progress INTEGER DEFAULT 0,  -- 0-100 percentage
    current_step TEXT,  -- Human-readable current step
    
    -- Results
    result_summary JSONB,
    error_message TEXT,
    error_details JSONB,
    
    -- Webhook notification
    webhook_url TEXT,
    webhook_sent BOOLEAN DEFAULT FALSE,
    
    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    -- User tracking
    user_id UUID,  -- From Supabase Auth
    
    -- Retry handling
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3
);

CREATE INDEX idx_job_queue_status ON job_queue(status);
CREATE INDEX idx_job_queue_user ON job_queue(user_id);
CREATE INDEX idx_job_queue_created ON job_queue(created_at DESC);

-- ============================================================
-- 8. Generated Reports
-- ============================================================
CREATE TABLE reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    job_id UUID REFERENCES job_queue(id) ON DELETE SET NULL,
    
    title TEXT NOT NULL,
    report_type TEXT NOT NULL,  -- 'risk_drift', 'sentiment_analysis', 'full_research'
    markdown_content TEXT NOT NULL,
    
    -- Report metadata
    parameters JSONB,  -- Input parameters used
    summary JSONB,     -- Key findings
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_reports_company ON reports(company_id);
CREATE INDEX idx_reports_job ON reports(job_id);
CREATE INDEX idx_reports_type ON reports(report_type);

-- ============================================================
-- 9. Audit Logs (LLM Usage Tracking)
-- ============================================================
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Context
    job_id UUID REFERENCES job_queue(id) ON DELETE SET NULL,
    operation TEXT NOT NULL,  -- 'extraction', 'analysis', 'synthesis'
    
    -- Model details
    model_id TEXT NOT NULL,  -- 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet'
    provider TEXT NOT NULL,  -- 'openrouter', 'direct'
    
    -- Token usage
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER GENERATED ALWAYS AS (input_tokens + output_tokens) STORED,
    
    -- Cost (calculated from model rates)
    cost_usd NUMERIC(10, 6),
    
    -- Performance
    latency_ms INTEGER,
    
    -- Status
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_job ON audit_logs(job_id);
CREATE INDEX idx_audit_logs_model ON audit_logs(model_id);
CREATE INDEX idx_audit_logs_date ON audit_logs(created_at DESC);

-- Materialized view for cost aggregation
CREATE MATERIALIZED VIEW daily_usage AS
SELECT 
    DATE(created_at) AS date,
    model_id,
    provider,
    COUNT(*) AS request_count,
    SUM(input_tokens) AS total_input_tokens,
    SUM(output_tokens) AS total_output_tokens,
    SUM(cost_usd) AS total_cost_usd
FROM audit_logs
WHERE success = TRUE
GROUP BY DATE(created_at), model_id, provider;

CREATE UNIQUE INDEX idx_daily_usage ON daily_usage(date, model_id, provider);

-- ============================================================
-- 10. Vector Search Functions
-- ============================================================

-- Semantic search across risk factors
CREATE OR REPLACE FUNCTION match_risk_factors(
    query_embedding VECTOR(768),
    match_count INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    id UUID,
    filing_id UUID,
    title TEXT,
    content TEXT,
    rank INTEGER,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rf.id,
        rf.filing_id,
        rf.title,
        rf.content,
        rf.rank,
        1 - (rf.embedding <=> query_embedding) AS similarity
    FROM risk_factors rf
    WHERE 1 - (rf.embedding <=> query_embedding) > similarity_threshold
    ORDER BY rf.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Semantic search across filing chunks
CREATE OR REPLACE FUNCTION match_filing_chunks(
    query_embedding VECTOR(768),
    p_filing_id UUID DEFAULT NULL,
    match_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    filing_id UUID,
    content TEXT,
    section_type TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fe.id,
        fe.filing_id,
        fe.content,
        fe.section_type,
        1 - (fe.embedding <=> query_embedding) AS similarity
    FROM filing_embeddings fe
    WHERE (p_filing_id IS NULL OR fe.filing_id = p_filing_id)
    ORDER BY fe.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================
-- 11. Row Level Security (RLS)
-- ============================================================

-- Enable RLS on all tables
ALTER TABLE companies ENABLE ROW LEVEL SECURITY;
ALTER TABLE filings ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_drifts ENABLE ROW LEVEL SECURITY;
ALTER TABLE job_queue ENABLE ROW LEVEL SECURITY;
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Public read for companies (reference data)
CREATE POLICY "Companies are viewable by everyone" 
    ON companies FOR SELECT 
    USING (TRUE);

-- Authenticated users can read filings
CREATE POLICY "Filings are viewable by authenticated users" 
    ON filings FOR SELECT 
    TO authenticated
    USING (TRUE);

-- Users can only see their own jobs
CREATE POLICY "Users can view their own jobs" 
    ON job_queue FOR SELECT 
    TO authenticated
    USING (user_id = auth.uid());

CREATE POLICY "Users can create their own jobs" 
    ON job_queue FOR INSERT 
    TO authenticated
    WITH CHECK (user_id = auth.uid());

-- Service role bypass for backend operations
CREATE POLICY "Service role has full access to job_queue" 
    ON job_queue FOR ALL 
    TO service_role
    USING (TRUE);

CREATE POLICY "Service role has full access to filings" 
    ON filings FOR ALL 
    TO service_role
    USING (TRUE);

CREATE POLICY "Service role has full access to risk_factors" 
    ON risk_factors FOR ALL 
    TO service_role
    USING (TRUE);

CREATE POLICY "Service role has full access to risk_drifts" 
    ON risk_drifts FOR ALL 
    TO service_role
    USING (TRUE);

CREATE POLICY "Service role has full access to reports" 
    ON reports FOR ALL 
    TO service_role
    USING (TRUE);

CREATE POLICY "Service role has full access to audit_logs" 
    ON audit_logs FOR ALL 
    TO service_role
    USING (TRUE);
```

---

## Python Adapter Interface

Implement in `insights/adapters/db/manager.py`:

```python
from typing import Optional, List
from uuid import UUID
from datetime import date
from supabase import create_client, Client
from pydantic import BaseModel

class CompanyCreate(BaseModel):
    ticker: str
    cik: str
    name: Optional[str] = None
    sector: Optional[str] = None

class FilingCreate(BaseModel):
    company_id: UUID
    accession_number: str
    form_type: str
    filing_date: date
    fiscal_year: Optional[int] = None
    raw_text: Optional[str] = None

class RiskFactorCreate(BaseModel):
    filing_id: UUID
    title: str
    content: str
    rank: int
    embedding: Optional[List[float]] = None

class DBManager:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
    
    async def get_or_create_company(self, data: CompanyCreate) -> UUID:
        """Upsert company by ticker, return UUID."""
        result = self.client.table("companies").upsert(
            data.model_dump(),
            on_conflict="ticker"
        ).execute()
        return UUID(result.data[0]["id"])
    
    async def save_filing(self, data: FilingCreate) -> UUID:
        """Insert filing, return UUID."""
        result = self.client.table("filings").insert(
            data.model_dump(mode="json")
        ).execute()
        return UUID(result.data[0]["id"])
    
    async def save_risk_factors(
        self, 
        filing_id: UUID, 
        factors: List[RiskFactorCreate]
    ) -> List[UUID]:
        """Batch insert risk factors."""
        records = [
            {**f.model_dump(mode="json"), "filing_id": str(filing_id)}
            for f in factors
        ]
        result = self.client.table("risk_factors").insert(records).execute()
        return [UUID(r["id"]) for r in result.data]
    
    async def vector_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[dict]:
        """Semantic search using pgvector."""
        result = self.client.rpc(
            "match_risk_factors",
            {
                "query_embedding": query_embedding,
                "match_count": limit,
                "similarity_threshold": threshold
            }
        ).execute()
        return result.data
    
    async def create_job(
        self, 
        job_type: str, 
        payload: dict, 
        user_id: Optional[UUID] = None,
        webhook_url: Optional[str] = None
    ) -> UUID:
        """Create a new job in the queue."""
        result = self.client.table("job_queue").insert({
            "job_type": job_type,
            "request_payload": payload,
            "user_id": str(user_id) if user_id else None,
            "webhook_url": webhook_url
        }).execute()
        return UUID(result.data[0]["id"])
    
    async def update_job_status(
        self,
        job_id: UUID,
        status: str,
        progress: Optional[int] = None,
        current_step: Optional[str] = None,
        result_summary: Optional[dict] = None,
        error_message: Optional[str] = None
    ):
        """Update job status."""
        update_data = {"status": status}
        if progress is not None:
            update_data["progress"] = progress
        if current_step:
            update_data["current_step"] = current_step
        if result_summary:
            update_data["result_summary"] = result_summary
        if error_message:
            update_data["error_message"] = error_message
        if status == "processing":
            update_data["started_at"] = "now()"
        if status in ("completed", "failed"):
            update_data["completed_at"] = "now()"
        
        self.client.table("job_queue").update(update_data)\
            .eq("id", str(job_id)).execute()
    
    async def log_llm_usage(
        self,
        job_id: Optional[UUID],
        operation: str,
        model_id: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: int,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log LLM usage for auditing."""
        self.client.table("audit_logs").insert({
            "job_id": str(job_id) if job_id else None,
            "operation": operation,
            "model_id": model_id,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "success": success,
            "error_message": error_message
        }).execute()
```

---

## Alembic Migration Setup

```bash
# Install Alembic
uv add alembic asyncpg

# Initialize Alembic
cd python/insights
alembic init migrations

# Update alembic.ini
# sqlalchemy.url = postgresql+asyncpg://${SUPABASE_DB_URL}
```

Example migration file:

```python
# migrations/versions/001_initial_schema.py
"""Initial schema

Revision ID: 001
Create Date: 2024-01-29
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = '001'
down_revision = None

def upgrade():
    # Enable extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create companies table
    op.create_table(
        'companies',
        sa.Column('id', sa.UUID(), primary_key=True),
        sa.Column('ticker', sa.Text(), unique=True, nullable=False),
        sa.Column('cik', sa.Text(), nullable=False),
        sa.Column('name', sa.Text()),
        sa.Column('sector', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )
    # ... continue with other tables

def downgrade():
    op.drop_table('companies')
    # ... continue with other tables
```