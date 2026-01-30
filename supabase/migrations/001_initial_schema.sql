-- ============================================================
-- InSights-ai Database Schema for Supabase
-- Version: 1.0.0
-- 
-- Execute this in Supabase SQL Editor or via migration tool
-- ============================================================

-- ============================================================
-- Enable Extensions
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search optimization

-- ============================================================
-- 1. Companies Reference
-- ============================================================
CREATE TABLE IF NOT EXISTS companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker TEXT UNIQUE NOT NULL,
    cik TEXT NOT NULL,
    name TEXT,
    sector TEXT,
    exchange TEXT,  -- NYSE, NASDAQ, etc.
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker);
CREATE INDEX IF NOT EXISTS idx_companies_cik ON companies(cik);

-- ============================================================
-- 2. Filings (Source of Truth)
-- ============================================================
CREATE TABLE IF NOT EXISTS filings (
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

CREATE INDEX IF NOT EXISTS idx_filings_company ON filings(company_id);
CREATE INDEX IF NOT EXISTS idx_filings_date ON filings(filing_date DESC);
CREATE INDEX IF NOT EXISTS idx_filings_type ON filings(form_type);
CREATE UNIQUE INDEX IF NOT EXISTS idx_filings_company_year 
    ON filings(company_id, form_type, fiscal_year);

-- ============================================================
-- 3. Risk Factors (Extracted from Filings)
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_factors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_id UUID NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    rank INTEGER NOT NULL,  -- Order in the filing (1-indexed)
    embedding VECTOR(768),  -- Google text-embedding-004 dimensions
    word_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_factors_filing ON risk_factors(filing_id);
CREATE INDEX IF NOT EXISTS idx_risk_factors_rank ON risk_factors(filing_id, rank);

-- Vector similarity search index (requires ~100 records before creating)
-- CREATE INDEX idx_risk_factors_embedding ON risk_factors 
--     USING ivfflat (embedding vector_cosine_ops) 
--     WITH (lists = 100);

-- ============================================================
-- 4. Risk Drifts (Comparison Results)
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_drifts (
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

CREATE INDEX IF NOT EXISTS idx_risk_drifts_company ON risk_drifts(company_id);
CREATE INDEX IF NOT EXISTS idx_risk_drifts_zone ON risk_drifts(zone);
CREATE INDEX IF NOT EXISTS idx_risk_drifts_heat ON risk_drifts(heat_score DESC);

-- ============================================================
-- 5. Filing Embeddings (Chunked for RAG)
-- ============================================================
CREATE TABLE IF NOT EXISTS filing_embeddings (
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

CREATE INDEX IF NOT EXISTS idx_filing_embeddings_filing ON filing_embeddings(filing_id);
CREATE INDEX IF NOT EXISTS idx_filing_embeddings_section ON filing_embeddings(section_type);

-- ============================================================
-- 6. Analysis Chunks (With FinBERT Sentiment)
-- ============================================================
CREATE TABLE IF NOT EXISTS analysis_chunks (
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

CREATE INDEX IF NOT EXISTS idx_analysis_chunks_filing ON analysis_chunks(filing_id);
CREATE INDEX IF NOT EXISTS idx_analysis_chunks_sentiment ON analysis_chunks(sentiment_label);

-- ============================================================
-- 7. Job Queue (Async Processing)
-- ============================================================
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'job_status') THEN
        CREATE TYPE job_status AS ENUM ('queued', 'processing', 'completed', 'failed', 'cancelled');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS job_queue (
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

CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status);
CREATE INDEX IF NOT EXISTS idx_job_queue_user ON job_queue(user_id);
CREATE INDEX IF NOT EXISTS idx_job_queue_created ON job_queue(created_at DESC);

-- ============================================================
-- 8. Generated Reports
-- ============================================================
CREATE TABLE IF NOT EXISTS reports (
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

CREATE INDEX IF NOT EXISTS idx_reports_company ON reports(company_id);
CREATE INDEX IF NOT EXISTS idx_reports_job ON reports(job_id);
CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type);

-- ============================================================
-- 9. Audit Logs (LLM Usage Tracking)
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Context
    job_id UUID REFERENCES job_queue(id) ON DELETE SET NULL,
    operation TEXT NOT NULL,  -- 'extraction', 'analysis', 'synthesis'
    
    -- Model details
    model_id TEXT NOT NULL,  -- 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet'
    provider TEXT NOT NULL,  -- 'openrouter', 'siliconflow', 'direct'
    
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

CREATE INDEX IF NOT EXISTS idx_audit_logs_job ON audit_logs(job_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_model ON audit_logs(model_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_date ON audit_logs(created_at DESC);

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
        (1 - (rf.embedding <=> query_embedding))::FLOAT AS similarity
    FROM risk_factors rf
    WHERE rf.embedding IS NOT NULL
        AND (1 - (rf.embedding <=> query_embedding)) > similarity_threshold
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
        (1 - (fe.embedding <=> query_embedding))::FLOAT AS similarity
    FROM filing_embeddings fe
    WHERE fe.embedding IS NOT NULL
        AND (p_filing_id IS NULL OR fe.filing_id = p_filing_id)
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
ALTER TABLE filing_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE job_queue ENABLE ROW LEVEL SECURITY;
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Public read for companies (reference data)
DROP POLICY IF EXISTS "Companies are viewable by everyone" ON companies;
CREATE POLICY "Companies are viewable by everyone" 
    ON companies FOR SELECT 
    USING (TRUE);

-- Authenticated users can read filings
DROP POLICY IF EXISTS "Filings are viewable by authenticated users" ON filings;
CREATE POLICY "Filings are viewable by authenticated users" 
    ON filings FOR SELECT 
    TO authenticated
    USING (TRUE);

DROP POLICY IF EXISTS "Risk factors are viewable by authenticated users" ON risk_factors;
CREATE POLICY "Risk factors are viewable by authenticated users" 
    ON risk_factors FOR SELECT 
    TO authenticated
    USING (TRUE);

DROP POLICY IF EXISTS "Risk drifts are viewable by authenticated users" ON risk_drifts;
CREATE POLICY "Risk drifts are viewable by authenticated users" 
    ON risk_drifts FOR SELECT 
    TO authenticated
    USING (TRUE);

-- Users can only see their own jobs
DROP POLICY IF EXISTS "Users can view their own jobs" ON job_queue;
CREATE POLICY "Users can view their own jobs" 
    ON job_queue FOR SELECT 
    TO authenticated
    USING (user_id = auth.uid());

DROP POLICY IF EXISTS "Users can create their own jobs" ON job_queue;
CREATE POLICY "Users can create their own jobs" 
    ON job_queue FOR INSERT 
    TO authenticated
    WITH CHECK (user_id = auth.uid());

-- Service role bypass for backend operations
DROP POLICY IF EXISTS "Service role has full access to companies" ON companies;
CREATE POLICY "Service role has full access to companies" 
    ON companies FOR ALL 
    TO service_role
    USING (TRUE)
    WITH CHECK (TRUE);

DROP POLICY IF EXISTS "Service role has full access to filings" ON filings;
CREATE POLICY "Service role has full access to filings" 
    ON filings FOR ALL 
    TO service_role
    USING (TRUE)
    WITH CHECK (TRUE);

DROP POLICY IF EXISTS "Service role has full access to risk_factors" ON risk_factors;
CREATE POLICY "Service role has full access to risk_factors" 
    ON risk_factors FOR ALL 
    TO service_role
    USING (TRUE)
    WITH CHECK (TRUE);

DROP POLICY IF EXISTS "Service role has full access to risk_drifts" ON risk_drifts;
CREATE POLICY "Service role has full access to risk_drifts" 
    ON risk_drifts FOR ALL 
    TO service_role
    USING (TRUE)
    WITH CHECK (TRUE);

DROP POLICY IF EXISTS "Service role has full access to filing_embeddings" ON filing_embeddings;
CREATE POLICY "Service role has full access to filing_embeddings" 
    ON filing_embeddings FOR ALL 
    TO service_role
    USING (TRUE)
    WITH CHECK (TRUE);

DROP POLICY IF EXISTS "Service role has full access to analysis_chunks" ON analysis_chunks;
CREATE POLICY "Service role has full access to analysis_chunks" 
    ON analysis_chunks FOR ALL 
    TO service_role
    USING (TRUE)
    WITH CHECK (TRUE);

DROP POLICY IF EXISTS "Service role has full access to job_queue" ON job_queue;
CREATE POLICY "Service role has full access to job_queue" 
    ON job_queue FOR ALL 
    TO service_role
    USING (TRUE)
    WITH CHECK (TRUE);

DROP POLICY IF EXISTS "Service role has full access to reports" ON reports;
CREATE POLICY "Service role has full access to reports" 
    ON reports FOR ALL 
    TO service_role
    USING (TRUE)
    WITH CHECK (TRUE);

DROP POLICY IF EXISTS "Service role has full access to audit_logs" ON audit_logs;
CREATE POLICY "Service role has full access to audit_logs" 
    ON audit_logs FOR ALL 
    TO service_role
    USING (TRUE)
    WITH CHECK (TRUE);

-- ============================================================
-- 12. Materialized View for Usage Stats
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS daily_usage;
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

CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_usage ON daily_usage(date, model_id, provider);

-- Helper function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_materialized_view(view_name TEXT)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    EXECUTE 'REFRESH MATERIALIZED VIEW CONCURRENTLY ' || quote_ident(view_name);
EXCEPTION
    WHEN OTHERS THEN
        -- If CONCURRENTLY fails (no unique index), try regular refresh
        EXECUTE 'REFRESH MATERIALIZED VIEW ' || quote_ident(view_name);
END;
$$;

-- ============================================================
-- Done!
-- ============================================================
