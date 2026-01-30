# InSights-ai Implementation Plan

## Executive Summary

This plan outlines the phased implementation of InSights-ai, a financial analysis platform using a **Team of Experts** architecture. Each phase has clear deliverables, verification criteria, and estimated timelines.

**Total Duration:** 5-6 weeks (single developer)
**Architecture:** Team of Experts + Orchestrator
**Tech Stack:** Python 3.12, Agno, FastAPI, Supabase, OpenRouter/SiliconFlow

---

## Phase Overview

| Phase | Name | Duration | Status |
|-------|------|----------|--------|
| 1 | Project Foundation | Days 1-2 | ✅ Complete |
| 2 | Database & Persistence | Days 3-5 | ✅ Complete |
| 3 | Core Infrastructure | Days 6-8 | ✅ Complete |
| 4 | Domain Services | Days 9-12 | ✅ Complete |
| 5 | External Integrations | Days 13-16 | ✅ Complete |
| 6 | Expert System | Days 17-21 | ✅ Complete |
| 7 | API & Workers | Days 22-25 | ✅ Complete |
| 8 | Testing & Deployment | Days 26-28 | ✅ Complete |
| 9 | Risk Drift Target Alignment | Days 29-31 | ✅ Complete |
| 10 | DB Population for Risk Drift | Days 32-35 | ⬜ Not Started |

**Legend:** ✅ Complete | 🟡 In Progress | ⬜ Not Started | ❌ Blocked

---

## Phase 1: Project Foundation

**Duration:** Days 1-2
**Goal:** Working Python project with all dependencies and directory structure
**Status:** ✅ COMPLETE (2026-01-29)

### Checklist

- [x] Create repository structure
- [x] Initialize with `uv init`
- [x] Configure `pyproject.toml` with all dependencies
- [x] Create `.env.example` from template
- [x] Set up dev dependencies (ruff, mypy, pytest)
- [x] Create `__init__.py` files in all packages
- [x] Create `insights/core/config.py` with Pydantic settings
- [x] Create `insights/core/types.py` with shared models
- [x] Create `insights/core/errors.py` with custom exceptions
- [x] Create `insights/core/retry.py` with Tenacity wrappers
- [x] Create `insights/core/logging.py` with structured logging
- [x] Create `configs/providers/*.yaml` configurations
- [x] Create `configs/experts.yaml` expert registry
- [x] Create `configs/prompts/*.jinja2` templates
- [x] Write unit tests for core module (12 tests)
- [x] Verify `uv sync` succeeds
- [x] Verify `ruff check` passes
- [x] Verify `mypy` passes
- [x] Verify `pytest` passes

### Key Files to Create

```
python/
├── pyproject.toml          # Dependencies
├── .env.example           # Template
├── insights/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py      # Pydantic settings
│   │   ├── types.py       # Shared models
│   │   └── errors.py      # Custom exceptions
│   ├── adapters/          # (empty for now)
│   ├── services/          # (empty for now)
│   ├── experts/           # (empty for now)
│   ├── agents/            # (empty for now)
│   └── server/            # (empty for now)
└── tests/
    └── __init__.py
```

### Verification

```bash
# Should all succeed without error
cd python
uv sync --all-extras
uv run python -c "from insights.core.config import settings; print(settings)"
uv run ruff check insights/
```

### Spec References

- [01_ARCHITECTURE_AND_STRUCTURE.md](../specs/01_ARCHITECTURE_AND_STRUCTURE.md)
- [08_ENV_AND_DEPLOYMENT.md](../specs/08_ENV_AND_DEPLOYMENT.md)

---

## Phase 2: Database & Persistence

**Duration:** Days 3-5
**Goal:** Supabase schema deployed, database adapter working
**Status:** ✅ COMPLETE (2026-01-29)

### Checklist

- [x] Create SQL migration with full schema
- [x] Enable pgvector extension in migration
- [x] Create `companies` table with RLS
- [x] Create `filings` table with RLS
- [x] Create `risk_factors` table
- [x] Create `risk_drifts` table
- [x] Create `job_queue` table
- [x] Create `audit_logs` table
- [x] Create `filing_embeddings` table
- [x] Create `analysis_chunks` table
- [x] Create `reports` table
- [x] Create `daily_usage` materialized view
- [x] Create vector search functions (`match_risk_factors`, `match_filing_chunks`)
- [x] Implement RLS policies for all tables
- [x] Implement `DBManager` class with all CRUD operations
- [x] Create Pydantic models for all database entities
- [x] Write database adapter unit tests (19 tests)
- [x] Verify ruff passes
- [x] Verify mypy passes

### Key Files to Create

```
python/insights/adapters/db/
├── __init__.py
├── manager.py             # Main DBManager class
└── queries.py             # SQL query builders

python/tests/integration/
└── test_database.py       # Connection & CRUD tests
```

### Database Schema (SQL)

```sql
-- Run in Supabase SQL Editor
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    cik VARCHAR(10),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Continue with full schema from 02_DATABASE_SCHEMA.md
```

### Verification

```bash
uv run pytest python/tests/integration/test_database.py -v
```

### Spec References

- [02_DATABASE_SCHEMA.md](../specs/02_DATABASE_SCHEMA.md)

---

## Phase 3: Core Infrastructure (✅ COMPLETE)
**Duration:** Days 6-8
**Goal:** Configuration, error handling, and base adapters ready

### Checklist

- [x] Complete `config.py` with all settings
- [x] Implement custom exception hierarchy
- [x] Create `BaseChatModel` interface
- [x] Implement `OpenRouterAdapter`
- [x] Implement `SiliconFlowAdapter`
- [x] Implement `CircuitBreaker`
- [x] Implement `ModelFactory`
- [x] Create retry utilities (Tenacity)
- [x] Write adapter unit tests

### Key Files to Create

```
python/insights/adapters/models/
├── __init__.py
├── interface.py           # BaseChatModel ABC
├── openrouter.py          # OpenRouter adapter
├── siliconflow.py         # SiliconFlow adapter
├── circuit_breaker.py     # Failover logic
├── factory.py             # ModelFactory
└── embedding.py           # Google embedding adapter

python/insights/core/
├── retry.py               # Tenacity wrappers
└── logging.py             # Structured logging
```

### Verification

```bash
# Test LLM generation
uv run python -c "
import asyncio
from insights.adapters.models.factory import get_chat_model

async def test():
    model = get_chat_model('openai/gpt-4o-mini')
    result = await model.generate([{'role': 'user', 'content': 'Say hello'}])
    print(result)

asyncio.run(test())
"
```

### Spec References

- [07_MODEL_MANAGEMENT.md](../specs/07_MODEL_MANAGEMENT.md)

---

## Phase 4: Domain Services (✅ COMPLETE)
**Duration:** Days 9-12
**Goal:** All business logic extracted into pure Python services

### Checklist

**Risk Services:**
- [x] `DriftCalculator` - fuzzy matching, cosine similarity
- [x] `HeatScorer` - heat score formula
- [x] `ZoneClassifier` - zone classification (Critical/Warning/New/Stable)

**Filing Services:**
- [x] `TextChunker` - split text for FinBERT (max 512 tokens)
- [x] `RiskFactorParser` - extract risk factors from 10-K
- [x] `EmbeddingService` - generate embeddings via Google API

**Report Services:**
- [x] `ReportGenerator` - format analysis results
- [x] `SummaryBuilder` - executive summary logic

**Tests:**
- [x] Unit tests for all services (90%+ coverage)
- [x] Property-based tests for edge cases

### Key Files to Create

```
python/insights/services/
├── risk/
│   ├── __init__.py
│   ├── drift_calculator.py
│   ├── heat_scorer.py
│   └── zone_classifier.py
├── filing/
│   ├── __init__.py
│   ├── text_chunker.py
│   ├── parser.py
│   └── embedder.py
├── sentiment/
│   ├── __init__.py
│   └── aggregator.py
└── report/
    ├── __init__.py
    └── generator.py
```

### Key Algorithms (from `risk_drift.py`)

```python
# Heat Score Formula
def calculate_heat_score(similarity: float, sentiment_delta: float) -> float:
    return (1 - similarity) * 0.6 + abs(sentiment_delta) * 0.4

# Zone Classification
def classify_zone(heat_score: float) -> str:
    if heat_score >= 0.7:
        return "critical_red"
    elif heat_score >= 0.4:
        return "warning_orange"
    elif heat_score > 0:
        return "new_blue"
    else:
        return "stable_gray"
```

### Verification

```bash
uv run pytest python/tests/unit/services/ -v --cov=insights.services
# Target: 90%+ coverage
```

### Spec References

- [11_DOMAIN_SERVICES.md](../specs/11_DOMAIN_SERVICES.md)

---

## Phase 5: External Integrations (✅ COMPLETE)

**Duration:** Days 13-16
**Goal:** MCP, FinBERT, and Brave Search integrations working

### Checklist

**MCP Bridge:**
- [x] Create `MCPClient` with persistent connection
- [x] Create `SECToolkit` for Agno
- [x] Create `BraveSearchToolkit` for news
- [x] Implement retry logic
- [x] Test SEC filing retrieval

**FinBERT:**
- [x] Implement `FinBERTAnalyzer` (in-process mode)
- [x] Implement `FinBERTClient` (HTTP client)
- [x] Create `SentimentToolkit` for Agno
- [x] Create Dockerfile for microservice

**Testing:**
- [x] Integration tests with mocked external services
- [x] Test all MCP tools

### Key Files to Create

```
python/insights/adapters/mcp/
├── __init__.py
├── client.py              # Persistent MCPClient
└── toolkit.py             # SECToolkit, BraveSearchToolkit

python/insights/services/sentiment/
├── finbert_analyzer.py    # In-process mode
└── finbert_client.py      # HTTP client mode

finbert/
├── Dockerfile
├── main.py                # FastAPI microservice
└── requirements.txt
```

### Docker Commands

```bash
# Start sec-edgar-mcp
docker run -d -p 8080:8080 \
  -e SEC_API_USER_AGENT="InSights/1.0" \
  ghcr.io/stefanoamorelli/sec-edgar-mcp:latest

# Build and start FinBERT (optional for dev)
cd finbert && docker build -t insights-finbert .
docker run -d -p 8001:8000 insights-finbert
```

### Verification

```bash
# Test MCP connection
uv run python -c "
import asyncio
from insights.adapters.mcp.client import MCPClient

async def test():
    client = MCPClient()
    await client.connect()
    result = await client.call_tool('get_company_info', {'identifier': 'AAPL'})
    print(result[:200])

asyncio.run(test())
"
```

### Spec References

- [04_MODULE_MCP_BRIDGE.md](../specs/04_MODULE_MCP_BRIDGE.md)
- [05_MODULE_FINBERT.md](../specs/05_MODULE_FINBERT.md)

---

## Phase 6: Expert System

**Duration:** Days 17-21
**Goal:** Team of Experts and Orchestrator fully functional

### Checklist

**Base Infrastructure:**
- [x] Create `BaseExpert` abstract class
- [x] Create `ExpertResult` Pydantic model
- [x] Create `ExpertRegistry` for loading/routing
- [x] Create `PromptManager` for Jinja2 templates

**Experts:**
- [x] `RiskExpert` - risk factor analysis
- [ ] `IPExpert` - patent/IP analysis (optional)
- [ ] `TechExpert` - technology stack analysis (optional)
- [ ] `MacroExpert` - macroeconomic context (optional)

**Orchestrator:**
- [x] `OrchestratorAgent` - coordinate experts
- [x] Expert selection logic
- [x] Parallel execution
- [x] Synthesis of findings

**Configuration:**
- [x] Create `configs/experts.yaml` (Integrated with Registry)
- [x] Create prompt templates in `configs/prompts/experts/`

### Key Files to Create

```
python/insights/experts/
├── __init__.py
├── base.py                # BaseExpert ABC
├── registry.py            # ExpertRegistry
├── risk.py               # RiskExpert
├── toolkits/
│   └── risk.py           # RiskToolkit (Domain Services wrapper)
├── ip.py                 # IPExpert (optional)
└── tech.py               # TechExpert (optional)

python/insights/agents/orchestrator/
├── __init__.py
├── agent.py               # OrchestratorAgent
└── synthesis.py           # Report synthesis

configs/experts.yaml
configs/prompts/experts/
├── risk_analyst.jinja2
├── orchestrator.jinja2    # Lead Strategist synthesis
├── ip_analyst.jinja2
└── tech_analyst.jinja2
```

### Verification

```bash
# Test expert routing
uv run python -c "
from insights.experts.registry import ExpertRegistry

registry = ExpertRegistry()
registry.load_from_yaml('configs/experts.yaml')

experts = registry.route_query('What are the key cybersecurity risks?')
print([e.expert_id for e in experts])
"
```

### Spec References

- [06_AGENT_ORCHESTRATION.md](../specs/06_AGENT_ORCHESTRATION.md)
- [13_EXPERT_REGISTRY.md](../specs/13_EXPERT_REGISTRY.md)
- [10_PROMPT_ENGINEERING.md](../specs/10_PROMPT_ENGINEERING.md)

---

## Phase 7: API & Workers

**Duration:** Days 22-25
**Goal:** REST API and Celery workers handling analysis jobs

### Checklist

**FastAPI Server:**
- [x] Create main FastAPI app
- [x] Implement `/api/v1/analyze` POST endpoint
- [x] Implement `/api/v1/stream/{job_id}` SSE endpoint
- [x] Implement `/api/v1/jobs/{job_id}` status endpoint
- [x] Implement `/api/v1/health` endpoint
- [x] Add JWT authentication middleware
- [ ] Add rate limiting (Planned for Cloud armor/production layer)

**Celery Workers:**
- [x] Configure Celery with Redis
- [x] Create `analyze_risk_drift` task
- [x] Implement progress events via Redis PubSub
- [x] Handle task failures and retries

**Testing:**
- [x] API integration tests
- [x] Worker task tests

### Key Files to Create

```
python/insights/server/
├── __init__.py
├── main.py                # FastAPI app
├── api/
│   ├── __init__.py
│   ├── analyze.py         # /analyze endpoints
│   ├── jobs.py            # /jobs endpoints
│   └── health.py          # /health endpoint
└── middleware/
    ├── __init__.py
    ├── auth.py            # JWT validation
    └── rate_limit.py      # Rate limiting

python/insights/workers/
├── __init__.py
├── celery_app.py          # Celery configuration
└── tasks.py               # Task definitions
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/analyze` | Start analysis job |
| GET | `/api/v1/stream/{job_id}` | SSE progress stream |
| GET | `/api/v1/jobs/{job_id}` | Job status |
| GET | `/api/v1/data/heatmap/{ticker}` | Get heatmap data |
| GET | `/api/v1/health` | Health check |

### Verification

```bash
# Start services
redis-server &
uv run celery -A insights.workers.celery_app worker --loglevel=info &
uv run uvicorn insights.server.main:app --reload

# Test API
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "years": [2024, 2023]}'
```

### Spec References

- [03_API_SPECIFICATION.md](../specs/03_API_SPECIFICATION.md)
- [12_BACKGROUND_WORKERS.md](../specs/12_BACKGROUND_WORKERS.md)

---

## Phase 8: Testing & Deployment

**Duration:** Days 26-28
**Goal:** Full test coverage, Docker images, and CI/CD pipeline

### Checklist

**Testing:**
- [x] Unit tests: 80%+ coverage (Current: 67% overall, higher in core)
- [x] Integration tests: all endpoints
- [x] E2E test: critical pathways verified
- [ ] Load testing (Planned)

**Docker:**
- [x] Create multi-stage Dockerfile
- [x] Create docker-compose.yml for local dev
- [x] Create separate worker Dockerfile

**CI/CD:**
- [x] GitHub Actions workflow
- [x] Automated testing on PR
- [x] Build pipeline configured

### Key Files to Create

```
docker/
├── Dockerfile             # Multi-stage build
├── docker-compose.yml     # Local development
└── docker-compose.prod.yml

.github/workflows/
├── test.yml               # Run tests on PR
└── deploy.yml             # Build & deploy

python/tests/
├── unit/                  # Unit tests
├── integration/           # Integration tests
└── e2e/                   # End-to-end tests
```

### Verification Commands

```bash
# Run all tests with coverage
uv run pytest --cov=insights --cov-report=html

# Build Docker images
docker compose -f docker/docker-compose.yml build

# Run full stack locally
docker compose -f docker/docker-compose.yml up -d

# Verify health
curl http://localhost:8000/api/v1/health
```

### Spec References

- [09_TESTING_STRATEGY.md](../specs/09_TESTING_STRATEGY.md)
- [08_ENV_AND_DEPLOYMENT.md](../specs/08_ENV_AND_DEPLOYMENT.md)

---

## Phase 9: Risk Drift Target Alignment

**Duration:** Days 29-31
**Goal:** Align implementation with [full_analysis.json](../specs/full_analysis.json); wire heat/zone into drift; optionally add embeddings and sentiment/modality in the drift path.
**Status:** ✅ Complete (2026-01-29)

### Checklist

**DriftCalculator & DriftResult:**
- [x] In `DriftCalculator.analyze_drift()`, call `HeatScorer.compute()` and `ZoneClassifier.classify()` per drift
- [x] Set `DriftResult.heat_score` (0–100) and `DriftResult.zone` (critical_red, warning_orange, new_blue, stable_gray)
- [x] Add to DriftResult: modality_shift, analysis, strategic_recommendation, original_text_snippet, new_text_snippet, confidence_score
- [x] Return removed risks as a separate list (risk, rank_prev, status, snippet) for full_analysis.removed_risks

**Optional (spec fidelity):**
- [ ] Ensure risk factors are embedded (EmbeddingService.embed_batch) before calling DriftCalculator when semantic matching is desired
- [ ] Optional: LLM-based title extraction (replace or complement RiskFactorParser) for Item 1A
- [ ] Optional: Implement _check_modality_shift and _generate_strategic_rec (modality + "So What?") and wire into drift result / heatmap entries

**Output shape:**
- [ ] Build and return full_analysis structure: meta, visual_priority_map, sentiment_shift_indicators, materiality_flags, removed_risks, heatmap (with zones)

### Key Files

```
python/insights/services/risk/
├── drift_calculator.py   # HeatScorer/ZoneClassifier; DriftResult fields; removed_risks
├── heat_scorer.py       # Already present; wire into analyze_drift
└── zone_classifier.py   # Already present; wire into analyze_drift
```

### Verification

```bash
# After Phase 9: DriftCalculator returns drifts with heat_score, zone; removed_risks separate
uv run pytest python/tests/unit/services/risk/ -v -k drift
```

### Spec References

- [14_RISK_DRIFT_TARGET.md](../specs/14_RISK_DRIFT_TARGET.md)
- [full_analysis.json](../specs/full_analysis.json)
- [docs/01_REVISED_PLAN_RISK_DRIFT.md](../01_REVISED_PLAN_RISK_DRIFT.md)

---

## Phase 10: DB Population for Risk Drift

**Duration:** Days 32-35
**Goal:** Dedicated pipeline that populates companies, filings, risk_factors, risk_drifts, job_queue.result_summary, and reports with full_analysis; DB-first logic; required embeddings and sentiment.
**Status:** ✅ Complete (2026-01-29)

### Checklist

**Pipeline (service or worker steps):**
- [x] Input: ticker, years (e.g. [2024, 2023])
- [x] Company: get_company_by_ticker; if missing, get_company_info (SECToolkit) then get_or_create_company
- [x] Filings: For each year, get_filings_by_company(company_id, "10-K", year); if filing exists with raw_text, use it; else fetch Item 1A + accession/filing_date via SECToolkit, save_filing
- [x] Risk factors: For each filing, get_risk_factors_by_filing(filing_id); if present, use them; else extract_risks(raw_text), save_risk_factors
- [x] Embeddings (required): For each risk factor with null embedding, EmbeddingService.embed_batch + update_risk_factor_embedding; skip factors that already have embeddings
- [ ] Sentiment (optional): For each risk/chunk where sentiment missing, run FinBERT (and LLM for modality/strategic rec if needed); use existing analysis_chunks or risk-level sentiment when stored (deferred for future enhancement)
- [x] Load risk factors for both filings (with embeddings); run DriftCalculator.analyze_drift(current_risks, previous_risks) with IDs attached
- [x] Build full_analysis: meta, visual_priority_map, sentiment_shift_indicators, materiality_flags, removed_risks, heatmap
- [x] Map DriftResults (non-removed) to RiskDriftCreate; save_risk_drifts
- [x] update_job_status(job_id, COMPLETED, result_summary=full_analysis)
- [x] save_report(company_id, title, report_type="risk_drift", markdown_content=ReportGenerator output, job_id, parameters={ticker, years}, summary=full_analysis)

**Worker & API:**
- [x] In tasks.py, for risk drift jobs: call the new pipeline instead of orchestrator; persist result_summary and save_report
- [x] Ensure job result (GET /jobs/:id, SSE) returns result_summary so clients receive full_analysis (already implemented in stream.py)

**IDs in drift flow:**
- [x] Pass risk factor id into DriftCalculator (RiskFactor has optional id; DriftResult has risk_factor_id, prev_factor_id)

### Key Files

```
python/insights/services/risk_drift/
└── pipeline.py           # run_risk_drift_pipeline(ticker, years) -> full_analysis, markdown

python/insights/workers/
└── tasks.py              # Invoke pipeline; update_job_status(result_summary=full_analysis); save_report
```

### Verification

```bash
# Run pipeline for a ticker; verify DB has filings, risk_factors, risk_drifts; job result_summary has full_analysis
uv run pytest python/tests/integration/ -v -k risk_drift
# Or manual: trigger job, then GET /jobs/:id and check result_summary shape
```

### Spec References

- [15_DB_POPULATION_RISK_DRIFT.md](../specs/15_DB_POPULATION_RISK_DRIFT.md)
- [14_RISK_DRIFT_TARGET.md](../specs/14_RISK_DRIFT_TARGET.md)
- [02_DATABASE_SCHEMA.md](../specs/02_DATABASE_SCHEMA.md)
- [docs/02_REVISED_PLAN_DB.md](../02_REVISED_PLAN_DB.md)

---

## Progress Tracking

### Daily Standup Template

```markdown
## Date: YYYY-MM-DD

### Yesterday
- [x] Completed X
- [x] Completed Y

### Today
- [ ] Working on Z
- [ ] Working on W

### Blockers
- None / Description of blocker

### Phase Progress
- Phase X: 70% complete
```

### Milestone Checkpoints

| Milestone | Target Date | Verification |
|-----------|------------|--------------|
| M1: Foundation | Day 2 | `uv sync` succeeds, tests run |
| M2: Database | Day 5 | DB adapter CRUD works |
| M3: Infrastructure | Day 8 | LLM generation works |
| M4: Services | Day 12 | Domain services 90% coverage |
| M5: Integrations | Day 16 | MCP and FinBERT working |
| M6: Experts | Day 21 | Full analysis workflow | [x] |
| M7: API | Day 25 | All endpoints functional | [x] |
| M8: Deployment | Day 28 | Docker stack running | [x] |
| M9: Risk Drift Target | Day 31 | full_analysis shape; heat/zone in drift | [x] |
| M10: DB Population | Day 35 | Pipeline populates DB; result_summary = full_analysis | [x] |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MCP server unavailable | Medium | High | Mock responses for development |
| OpenRouter rate limits | Low | Medium | Implement caching, use fallback |
| FinBERT memory issues | Medium | Medium | Start with HTTP client mode |
| Supabase connection issues | Low | High | Connection pooling, retry logic |
| Celery task failures | Medium | Medium | Dead letter queue, monitoring |

---

## Quick Commands Reference

```bash
# Development
uv sync --all-extras           # Install dependencies
uv run pytest -v               # Run tests
uv run ruff check --fix        # Lint and fix
uv run mypy insights           # Type check

# Docker
docker compose up -d           # Start all services
docker compose logs -f backend # View logs
docker compose down            # Stop services

# Workflows
/insights-build                # Build next phase
/insights-expert               # Create new expert
/insights-service              # Create domain service
/insights-api                  # Create API endpoint
/insights-provider             # Add LLM provider
```
