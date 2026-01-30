# 08. Environment & Deployment

## Overview

This document covers:
- Environment variable configuration
- Docker Compose for local development
- Google Cloud Run production deployment
- CI/CD with GitHub Actions
- Secrets management

---

## Environment Variables

### `.env.example`

```bash
# ==============================================================================
# InSights-ai Backend Environment Configuration
# ==============================================================================

# Application
ENV=development  # development | production | test
DEBUG=true
LOG_LEVEL=INFO
APP_URL=http://localhost:8000

# ==============================================================================
# Database (Supabase)
# ==============================================================================
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJhbG...
SUPABASE_SERVICE_ROLE_KEY=eyJhbG...
DATABASE_URL=postgresql://postgres:password@db.your-project.supabase.co:5432/postgres

# ==============================================================================
# LLM Providers
# ==============================================================================
# OpenRouter (Primary - aggregates multiple providers)
OPENROUTER_API_KEY=sk-or-v1-...
DEFAULT_MODEL=openai/gpt-4o

# Direct provider keys (Fallback)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Embeddings (Google)
GOOGLE_API_KEY=AIza...
EMBEDDING_MODEL=text-embedding-004

# ==============================================================================
# MCP Servers
# ==============================================================================
# SEC EDGAR MCP
MCP_SERVER_URL=http://localhost:8080/sse
SEC_USER_AGENT=InSights-ai/1.0 (contact@insights.ai)

# Brave Search MCP (Optional)
BRAVE_MCP_URL=http://localhost:8081/sse
BRAVE_API_KEY=BSA...

# ==============================================================================
# FinBERT Service
# ==============================================================================
FINBERT_MODE=local  # local | service
FINBERT_SERVICE_URL=http://finbert:8000

# ==============================================================================
# Task Queue (Redis + Celery)
# ==============================================================================
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# ==============================================================================
# Security
# ==============================================================================
# JWT validation
JWT_SECRET=your-supabase-jwt-secret
JWT_ALGORITHM=HS256

# API rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Webhook signing
WEBHOOK_SECRET=whsec_...

# ==============================================================================
# Monitoring (Optional)
# ==============================================================================
SENTRY_DSN=https://...@sentry.io/...
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

---

## Directory Structure

```
insights-v1/
├── docker/
│   ├── Dockerfile.backend      # Main API service
│   ├── Dockerfile.worker       # Celery worker
│   ├── Dockerfile.finbert      # FinBERT service
│   └── docker-compose.yml      # Local development
├── configs/
│   ├── providers/
│   │   ├── openrouter.yaml
│   │   └── fallback.yaml
│   ├── mcp/
│   │   └── sec_edgar.yaml
│   ├── prompts/
│   │   └── research_agent_system.jinja2
│   └── logging.yaml
├── pyproject.toml              # uv dependency management
├── uv.lock                     # Locked dependencies
└── .env                        # Local environment (gitignored)
```

---

## Docker Configuration

### `docker/Dockerfile.backend`

```dockerfile
# ============================================================================
# InSights-ai Backend Dockerfile
# Multi-stage build for smaller production image
# ============================================================================

# Stage 1: Build dependencies
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Stage 2: Production image
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker/Dockerfile.worker`

```dockerfile
# ============================================================================
# Celery Worker Dockerfile
# ============================================================================

FROM python:3.12-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libpq5 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

COPY src/ ./src/
COPY configs/ ./configs/

RUN useradd --create-home --shell /bin/bash appuser
RUN chown -R appuser:appuser /app
USER appuser

# Run Celery worker
CMD ["celery", "-A", "src.workers.celery_app", "worker", "--loglevel=info", "--concurrency=4"]
```

### `docker/docker-compose.yml`

```yaml
# ============================================================================
# InSights-ai Local Development Stack
# ============================================================================
version: "3.9"

services:
  # -------------------------------------------------------------------------
  # Main API Service
  # -------------------------------------------------------------------------
  backend:
    build:
      context: ..
      dockerfile: docker/Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - ENV=development
      - DEBUG=true
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_ANON_KEY=${SUPABASE_ANON_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - MCP_SERVER_URL=http://sec-mcp:8080/sse
      - REDIS_URL=redis://redis:6379/0
      - FINBERT_MODE=service
      - FINBERT_SERVICE_URL=http://finbert:8000
    depends_on:
      redis:
        condition: service_healthy
      sec-mcp:
        condition: service_healthy
    volumes:
      - ../src:/app/src:ro  # Hot reload in dev
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # -------------------------------------------------------------------------
  # Celery Worker
  # -------------------------------------------------------------------------
  worker:
    build:
      context: ..
      dockerfile: docker/Dockerfile.worker
    environment:
      - ENV=development
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_SERVICE_ROLE_KEY=${SUPABASE_SERVICE_ROLE_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - MCP_SERVER_URL=http://sec-mcp:8080/sse
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
      - FINBERT_SERVICE_URL=http://finbert:8000
    depends_on:
      - redis
      - sec-mcp
    volumes:
      - ../src:/app/src:ro

  # -------------------------------------------------------------------------
  # Redis (Task Queue)
  # -------------------------------------------------------------------------
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  # -------------------------------------------------------------------------
  # SEC EDGAR MCP Server
  # -------------------------------------------------------------------------
  sec-mcp:
    image: ghcr.io/stefanoamorelli/sec-edgar-mcp:latest
    environment:
      - SEC_API_USER_AGENT=${SEC_USER_AGENT:-InSights/1.0}
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # -------------------------------------------------------------------------
  # FinBERT Service (GPU optional for local dev)
  # -------------------------------------------------------------------------
  finbert:
    build:
      context: ../finbert
      dockerfile: Dockerfile
    ports:
      - "8001:8000"
    # Uncomment for GPU support
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s  # Model loading time

volumes:
  redis_data:

networks:
  default:
    name: insights-network
```

---

## Google Cloud Run Deployment

### `cloudbuild.yaml`

```yaml
# ============================================================================
# Google Cloud Build Configuration
# ============================================================================
steps:
  # Build backend image
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/insights-backend:$SHORT_SHA'
      - '-f'
      - 'docker/Dockerfile.backend'
      - '.'

  # Build worker image
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/insights-worker:$SHORT_SHA'
      - '-f'
      - 'docker/Dockerfile.worker'
      - '.'

  # Push images
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/insights-backend:$SHORT_SHA']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/insights-worker:$SHORT_SHA']

  # Deploy backend to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'insights-backend'
      - '--image=gcr.io/$PROJECT_ID/insights-backend:$SHORT_SHA'
      - '--region=us-central1'
      - '--platform=managed'
      - '--allow-unauthenticated'
      - '--memory=2Gi'
      - '--cpu=2'
      - '--min-instances=1'
      - '--max-instances=10'
      - '--set-secrets=SUPABASE_URL=supabase-url:latest,SUPABASE_ANON_KEY=supabase-anon-key:latest,OPENROUTER_API_KEY=openrouter-key:latest,GOOGLE_API_KEY=google-key:latest'

  # Deploy worker to Cloud Run Jobs (or use GKE for persistent workers)
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'jobs'
      - 'update'
      - 'insights-worker'
      - '--image=gcr.io/$PROJECT_ID/insights-worker:$SHORT_SHA'
      - '--region=us-central1'

images:
  - 'gcr.io/$PROJECT_ID/insights-backend:$SHORT_SHA'
  - 'gcr.io/$PROJECT_ID/insights-worker:$SHORT_SHA'

options:
  logging: CLOUD_LOGGING_ONLY
```

### Production Architecture

```
                    ┌─────────────────────────────────────────┐
                    │  Google Cloud Platform                  │
                    │                                         │
                    │  ┌─────────────────────────────────┐   │
                    │  │  Cloud Run (Backend)             │   │
                    │  │  - Auto-scaling (1-10 instances)│   │
                    │  │  - 2GB RAM, 2 vCPU              │   │
                    │  └──────────────┬──────────────────┘   │
                    │                 │                       │
Internet ──────────►│  Cloud Load Balancer                   │
                    │                 │                       │
                    │  ┌──────────────┴──────────────────┐   │
                    │  │                                  │   │
                    │  ▼                                  ▼   │
                    │  ┌─────────┐          ┌─────────────┐  │
                    │  │  Redis  │          │  Cloud Run  │  │
                    │  │Memorystr│◄─────────│   Worker    │  │
                    │  └─────────┘          └─────────────┘  │
                    │                                         │
                    │  ┌─────────────────────────────────┐   │
                    │  │  Cloud Run (FinBERT) - GPU      │   │
                    │  │  or Vertex AI Endpoint          │   │
                    │  └─────────────────────────────────┘   │
                    │                                         │
                    │  ┌─────────────────────────────────┐   │
                    │  │  Cloud Run (SEC MCP)            │   │
                    │  └─────────────────────────────────┘   │
                    │                                         │
                    │  ┌─────────────────────────────────┐   │
                    │  │  Secret Manager                  │   │
                    │  │  - API Keys                      │   │
                    │  │  - Database credentials          │   │
                    │  └─────────────────────────────────┘   │
                    └─────────────────────────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │  Supabase (External)                    │
                    │  - PostgreSQL + pgvector                │
                    │  - Auth (JWT)                           │
                    │  - Realtime (optional)                  │
                    └─────────────────────────────────────────┘
```

---

## GitHub Actions CI/CD

### `.github/workflows/deploy.yml`

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: us-central1

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Install dependencies
        run: uv sync --frozen
      
      - name: Run tests
        run: uv run pytest tests/ -v --cov=src --cov-report=xml
        env:
          ENV: test
          SUPABASE_URL: ${{ secrets.SUPABASE_URL_TEST }}
          SUPABASE_ANON_KEY: ${{ secrets.SUPABASE_ANON_KEY_TEST }}
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
        with:
          project_id: ${{ env.PROJECT_ID }}
      
      - name: Configure Docker for GCR
        run: gcloud auth configure-docker
      
      - name: Build and push backend
        run: |
          docker build -t gcr.io/$PROJECT_ID/insights-backend:${{ github.sha }} -f docker/Dockerfile.backend .
          docker push gcr.io/$PROJECT_ID/insights-backend:${{ github.sha }}
      
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy insights-backend \
            --image gcr.io/$PROJECT_ID/insights-backend:${{ github.sha }} \
            --region $REGION \
            --platform managed \
            --allow-unauthenticated \
            --memory 2Gi \
            --cpu 2 \
            --min-instances 1 \
            --max-instances 10
```

---

## Secrets Management

### Google Secret Manager

```bash
# Create secrets
gcloud secrets create supabase-url --replication-policy="automatic"
gcloud secrets versions add supabase-url --data-file=- <<< "https://your-project.supabase.co"

gcloud secrets create openrouter-key --replication-policy="automatic"
gcloud secrets versions add openrouter-key --data-file=- <<< "sk-or-v1-..."

# Grant Cloud Run access
gcloud secrets add-iam-policy-binding supabase-url \
    --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### Local Development

Use `.env` file (never commit):

```bash
# Copy example and fill in values
cp .env.example .env
```

---

## Dependency Management with uv

### `pyproject.toml`

```toml
[project]
name = "insights-ai"
version = "1.0.0"
description = "AI-powered SEC filing analysis platform"
readme = "README.md"
requires-python = ">=3.12"

dependencies = [
    # Web framework
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    
    # AI/ML
    "agno>=1.0.0",
    "openai>=1.10.0",
    "tiktoken>=0.5.2",
    
    # Database
    "supabase>=2.0.0",
    "asyncpg>=0.29.0",
    "alembic>=1.13.0",
    
    # Task Queue
    "celery[redis]>=5.3.0",
    "redis>=5.0.0",
    
    # HTTP
    "httpx>=0.26.0",
    "tenacity>=8.2.0",
    
    # Utilities
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "pyyaml>=6.0.1",
    "jinja2>=3.1.2",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "httpx>=0.26.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]

finbert = [
    "torch>=2.1.0",
    "transformers>=4.37.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.mypy]
python_version = "3.12"
strict = true
```

### Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --all-extras

# Add a dependency
uv add httpx

# Run tests
uv run pytest

# Run the app
uv run uvicorn src.main:app --reload
```