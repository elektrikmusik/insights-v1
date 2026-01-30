# Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the InSights-ai backend from scratch. Follow the phases in order—each builds on the previous.

**Estimated Time:** 3-4 weeks for a single developer

---

## Prerequisites

### Required Accounts & Services

- [ ] **Supabase Project** - Create at [supabase.com](https://supabase.com)
- [ ] **OpenRouter API Key** - Get at [openrouter.ai](https://openrouter.ai)
- [ ] **Google AI API Key** - For embeddings at [aistudio.google.com](https://aistudio.google.com)
- [ ] **GitHub Repository** - For CI/CD

### Local Development Environment

```bash
# Install Python 3.12+
brew install python@3.12

# Install uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Docker Desktop
brew install --cask docker

# Install Redis (for local development)
brew install redis
```

---

## Phase 1: Project Setup (Day 1)

### 1.1 Initialize Repository

```bash
# Clone or create project
mkdir insights-ai && cd insights-ai

# Initialize with uv
uv init --python 3.12

# Create directory structure
mkdir -p {python/insights/{adapters,services,agents,core,server,workers},configs/{agents,providers,prompts},migrations/versions,docker,finbert,tests/{unit,integration,fixtures}}
```

### 1.2 Configure Dependencies

Create `pyproject.toml`:

```toml
[project]
name = "insights-ai"
version = "0.1.0"
requires-python = ">=3.12"

dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "agno>=1.0.0",
    "httpx>=0.26.0",
    "supabase>=2.0.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "celery[redis]>=5.3.0",
    "redis>=5.0.0",
    "tenacity>=8.2.0",
    "tiktoken>=0.5.2",
    "rapidfuzz>=3.6.0",
    "numpy>=1.26.0",
    "jinja2>=3.1.2",
    "python-dotenv>=1.0.0",
    "alembic>=1.13.0",
    "asyncpg>=0.29.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.4.0", "pytest-asyncio>=0.23.0", "pytest-cov>=4.1.0", "ruff>=0.1.0"]
```

Install dependencies:

```bash
uv sync --all-extras
```

### 1.3 Environment Configuration

Create `.env` from template:

```bash
cp docs/specs/08_ENV_AND_DEPLOYMENT.md  # Reference for variables
touch .env

# Populate .env with your credentials
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJhbG...
OPENROUTER_API_KEY=sk-or-v1-...
GOOGLE_API_KEY=AIza...
REDIS_URL=redis://localhost:6379/0
```

---

## Phase 2: Database Layer (Days 2-3)

### 2.1 Run Schema Migrations

Reference: [02_DATABASE_SCHEMA.md](../specs/02_DATABASE_SCHEMA.md)

```bash
# Navigate to Supabase SQL Editor and run:
# 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

# 2. Run the schema SQL from 02_DATABASE_SCHEMA.md
# - companies table
# - filings table
# - risk_factors table
# - risk_drifts table
# - job_queue table
# - audit_logs table
# - filing_embeddings table
```

### 2.2 Implement Database Adapter

Create `python/insights/adapters/db/manager.py`:

```python
"""Database manager for Supabase operations."""
from supabase import create_client, Client
from insights.core.config import settings


class DBManager:
    def __init__(self):
        self.client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY
        )
    
    async def get_job(self, job_id: str) -> dict | None:
        result = self.client.table("job_queue").select("*").eq("id", job_id).execute()
        return result.data[0] if result.data else None
    
    async def update_job_status(self, job_id: str, status: str, **kwargs):
        self.client.table("job_queue").update({"status": status, **kwargs}).eq("id", job_id).execute()
    
    # Add more methods following 02_DATABASE_SCHEMA.md
```

### 2.3 Verify Database Connection

```python
# tests/integration/test_database.py
import pytest
from insights.adapters.db.manager import DBManager

@pytest.mark.asyncio
async def test_database_connection():
    db = DBManager()
    # Should not raise
    result = db.client.table("companies").select("*").limit(1).execute()
    assert result is not None
```

---

## Phase 3: Core Infrastructure (Days 4-6)

### 3.1 Configuration System

Create `python/insights/core/config.py`:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str = "development"
    DEBUG: bool = True
    
    # Supabase
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    
    # LLM
    OPENROUTER_API_KEY: str
    DEFAULT_MODEL: str = "openai/gpt-4o"
    
    # Embeddings
    GOOGLE_API_KEY: str
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # MCP
    MCP_SERVER_URL: str = "http://localhost:8080/sse"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 3.2 Error Handling

Create `python/insights/core/errors.py`:

```python
"""Custom exceptions for InSights-ai."""

class InsightsError(Exception):
    """Base exception."""
    pass

class MCPConnectionError(InsightsError):
    """MCP server connection failed."""
    pass

class MCPToolError(InsightsError):
    """MCP tool execution failed."""
    pass

class LLMProviderError(InsightsError):
    """LLM provider error."""
    pass

class SentimentServiceError(InsightsError):
    """FinBERT service error."""
    pass
```

### 3.3 Model Factory

Reference: [07_MODEL_MANAGEMENT.md](../specs/07_MODEL_MANAGEMENT.md)

Implement in order:
1. `python/insights/adapters/models/interface.py` - Abstract base
2. `python/insights/adapters/models/openrouter.py` - OpenRouter adapter
3. `python/insights/adapters/models/circuit_breaker.py` - Failover logic
4. `python/insights/adapters/models/factory.py` - Factory pattern

---

## Phase 4: Domain Services (Days 7-10)

### 4.1 Risk Services

Reference: [11_DOMAIN_SERVICES.md](../specs/11_DOMAIN_SERVICES.md)

Implement pure business logic (no external dependencies):

```bash
# Create service files
touch python/insights/services/risk/{__init__,drift_calculator,heat_scorer,zone_classifier}.py
```

**Key Implementation Order:**
1. `drift_calculator.py` - Fuzzy matching, cosine similarity
2. `heat_scorer.py` - Heat score formula
3. `zone_classifier.py` - Zone classification rules

### 4.2 Filing Services

```bash
touch python/insights/services/filing/{__init__,text_chunker,parser,embedder}.py
```

Implement:
1. `text_chunker.py` - Split text for FinBERT
2. `parser.py` - Extract risk factors from 10-K
3. `embedder.py` - Generate embeddings via Google API

### 4.3 Unit Tests

```bash
# Run unit tests for domain services
uv run pytest tests/unit/services/ -v
```

**Coverage Target:** 90% for domain services

---

## Phase 5: External Integrations (Days 11-14)

### 5.1 MCP Bridge

Reference: [04_MODULE_MCP_BRIDGE.md](../specs/04_MODULE_MCP_BRIDGE.md)

1. Start sec-edgar-mcp locally:

```bash
docker run -d -p 8080:8080 \
  -e SEC_API_USER_AGENT="InSights/1.0" \
  ghcr.io/stefanoamorelli/sec-edgar-mcp:latest
```

2. Implement MCP client and toolkits:

```bash
touch python/insights/adapters/mcp/{__init__,client,toolkit}.py
```

3. Test MCP connection:

```python
# tests/integration/test_mcp.py
async def test_mcp_fetch_filing():
    from insights.adapters.mcp.client import get_mcp_client
    client = get_mcp_client()
    await client.connect()
    result = await client.call_tool("get_company_info", {"identifier": "AAPL"})
    assert "Apple" in result
```

### 5.2 FinBERT Service

Reference: [05_MODULE_FINBERT.md](../specs/05_MODULE_FINBERT.md)

For development, use in-process mode:

```python
# python/insights/services/sentiment/finbert_analyzer.py
# Implement the FinBERTAnalyzer class
```

For production, build the microservice:

```bash
cd finbert
docker build -t insights-finbert .
docker run -p 8001:8000 insights-finbert
```

---

## Phase 6: Agent Layer (Days 15-17)

### 6.1 Prompt Templates

Reference: [10_PROMPT_ENGINEERING.md](../specs/10_PROMPT_ENGINEERING.md)

Create Jinja2 templates:

```bash
touch configs/prompts/{research_agent_system,risk_extractor,report_generator}.jinja2
```

### 6.2 Toolkit Wrappers

Create thin toolkit wrappers that delegate to services:

```bash
touch python/insights/agents/research/{__init__,agent,toolkits,prompts}.py
```

### 6.3 Research Agent

Reference: [06_AGENT_ORCHESTRATION.md](../specs/06_AGENT_ORCHESTRATION.md)

```python
# python/insights/agents/research/agent.py
from agno.agent import Agent
# Implement get_research_agent()
```

---

## Phase 7: API & Workers (Days 18-21)

### 7.1 FastAPI Application

Reference: [03_API_SPECIFICATION.md](../specs/03_API_SPECIFICATION.md)

```bash
touch python/insights/server/{main,api/{research,data,health}}.py
```

Start the server:

```bash
uv run uvicorn insights.server.main:app --reload
```

### 7.2 Celery Workers

Reference: [12_BACKGROUND_WORKERS.md](../specs/12_BACKGROUND_WORKERS.md)

```bash
touch python/insights/workers/{celery_app,tasks}.py
```

Start worker:

```bash
uv run celery -A insights.workers.celery_app worker --loglevel=info
```

---

## Phase 8: Testing & Deployment (Days 22-28)

### 8.1 Integration Tests

Reference: [09_TESTING_STRATEGY.md](../specs/09_TESTING_STRATEGY.md)

```bash
# Run all tests
uv run pytest --cov=insights --cov-report=html
```

### 8.2 Docker Build

Reference: [08_ENV_AND_DEPLOYMENT.md](../specs/08_ENV_AND_DEPLOYMENT.md)

```bash
# Build images
docker compose -f docker/docker-compose.yml build

# Start all services
docker compose -f docker/docker-compose.yml up -d
```

### 8.3 CI/CD

Set up GitHub Actions workflow at `.github/workflows/deploy.yml`

---

## Verification Checklist

### Core Functionality

- [x] Database connection works
- [x] LLM generation works (OpenRouter)
- [x] Embeddings generate correctly
- [x] MCP fetches SEC data
- [x] FinBERT returns sentiment scores

### API Endpoints

- [x] `POST /analyze` enqueues job
- [x] `GET /stream/{job_id}` streams progress
- [x] `GET /health` returns healthy

### Integration

- [x] Full workflow completes: POST → Worker → SSE → Complete
- [x] Results saved to database
- [x] Webhook fires on completion

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| MCP connection refused | Ensure `sec-edgar-mcp` container is running |
| LLM timeout | Check OpenRouter API key, try smaller model |
| FinBERT OOM | Reduce batch size or add GPU |
| Redis connection | Ensure Redis is running: `redis-cli ping` |
| Import errors | Run `uv sync` to install dependencies |

### Debug Mode

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG
uv run uvicorn insights.server.main:app --reload
```
