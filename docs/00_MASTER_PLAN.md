# 00. Master Plan & Tech Stack

## Project Overview

**InSights-ai** is a Deep Research & Analysis Platform built as a **"Team of Experts managed by an Orchestrator."** It coordinates specialized domain experts—each with their own tools, models, and prompts—to deliver comprehensive financial analysis.

> **Constraint:** Strictly NO trading execution capabilities. This is an analysis-only platform.

---

## Architecture Philosophy

### Team of Experts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR LAYER                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Orchestrator Agent                                                  │   │
│  │  • Routes queries to appropriate Experts                            │   │
│  │  • Coordinates parallel expert execution                            │   │
│  │  • Synthesizes multi-expert findings                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────┬─────────────────┬─────────────────┬──────────────────────┘
                   │                 │                 │
         ┌─────────▼─────────┐ ┌─────▼─────┐ ┌─────────▼─────────┐
         │  Risk Expert      │ │  IP Expert │ │  Tech Expert      │
         │  ├─ FinBERT      │ │  ├─ Gemini  │ │  ├─ GPT-4o       │
         │  ├─ SEC MCP      │ │  ├─ Patents │ │  ├─ GitHub API   │
         │  └─ Drift Tools  │ │  └─ Legal   │ │  └─ StackShare   │
         └───────────────────┘ └────────────┘ └───────────────────┘
                   │                 │                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DOMAIN SERVICES LAYER (Business Logic)                                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│  │ RiskService    │  │ FilingService  │  │ SentimentSvc   │                │
│  │ • Drift calc   │  │ • Text chunk   │  │ • FinBERT call │                │
│  │ • Heat scores  │  │ • SEC parsing  │  │ • Score agg    │                │
│  │ • Zone class   │  │ • Embeddings   │  │ • Batch proc   │                │
│  └────────────────┘  └────────────────┘  └────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ADAPTERS LAYER (Infrastructure)                                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│  │ MCP Client     │  │ Supabase DB    │  │ Model Factory  │                │
│  │ (sec-edgar)    │  │ (pgvector)     │  │ (OpenRouter)   │                │
│  └────────────────┘  └────────────────┘  └────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Team of Experts**: Specialized domain experts (Risk, IP, Tech, Macro) managed by an Orchestrator.
2. **Modular Expert System**: Add new domains without changing core architecture via Expert Registry.
3. **Agents Reason, Services Execute**: Experts decide workflow; Python services perform computation.
4. **Multi-Provider LLM**: Each expert can use optimal model (FinBERT, Gemini, GPT-4o, Claude).
5. **Structured Outputs**: All expert responses use Pydantic models for type-safe synthesis.
6. **Async-First**: Experts execute in parallel. FinBERT runs as a separate microservice in production.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Language** | Python 3.12+ | Async/await, type hints, dataclasses |
| **Framework** | Agno + FastAPI | Agent orchestration + REST API |
| **Database** | Supabase (PostgreSQL + pgvector) | Relational data + vector search |
| **ORM/Data** | Pydantic v2, supabase-py | Validation, serialization |
| **LLM Providers** | OpenRouter, SiliconFlow | Multi-provider: GPT-4o, Claude, DeepSeek, Qwen |
| **Embeddings** | Google text-embedding-004 | 768-dim vectors for semantic search |
| **Sentiment** | FinBERT (ProsusAI/finbert) | Financial sentiment classification |
| **Data Source** | sec-edgar-mcp (MCP Server) | SEC filings via Model Context Protocol |
| **News Search** | Brave Search MCP | Real-time sentiment context |
| **Task Queue** | Redis + Celery | Async job processing |
| **Dependency Mgmt** | uv | Fast Python package installer |

---

## Directory Structure

```text
insights/
├── configs/
│   ├── agents/                 # Agent YAML definitions
│   │   └── research_agent.yaml
│   ├── providers/              # LLM provider configs
│   │   ├── openrouter.yaml
│   │   └── fallback_chain.yaml
│   ├── prompts/                # Jinja2 prompt templates
│   │   ├── risk_analysis.jinja2
│   │   └── report_synthesis.jinja2
│   └── database.yaml           # Supabase credentials (via env vars)
│
├── python/
│   ├── insights/
│   │   ├── adapters/           # Infrastructure Layer
│   │   │   ├── mcp/            # MCP Client implementation
│   │   │   │   ├── client.py
│   │   │   │   └── toolkit.py  # Agno Toolkit wrapper
│   │   │   ├── db/             # Supabase interactions
│   │   │   │   └── manager.py
│   │   │   └── models/         # LLM Factory
│   │   │       ├── interface.py     # Abstract base
│   │   │       ├── factory.py       # Provider resolver
│   │   │       ├── openrouter.py    # OpenRouter adapter
│   │   │       └── circuit_breaker.py
│   │   │
│   │   ├── services/           # Domain Services Layer
│   │   │   ├── risk/
│   │   │   │   ├── drift_calculator.py   # Math: cosine sim, fuzzy match
│   │   │   │   ├── heat_scorer.py        # Heat score formula
│   │   │   │   └── zone_classifier.py    # Zone classification logic
│   │   │   ├── filing/
│   │   │   │   ├── text_chunker.py       # Text splitting
│   │   │   │   ├── parser.py             # SEC-specific parsing
│   │   │   │   └── embedder.py           # Embedding generation
│   │   │   ├── sentiment/
│   │   │   │   ├── finbert_client.py     # HTTP client to FinBERT service
│   │   │   │   └── aggregator.py         # Score aggregation
│   │   │   └── report/
│   │   │       └── formatter.py          # Markdown report generation
│   │   │
│   │   ├── experts/            # Expert Registry (Team of Experts)
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # BaseExpert abstract class
│   │   │   ├── registry.py           # ExpertRegistry loader
│   │   │   ├── risk.py               # RiskExpert (FinBERT, SEC)
│   │   │   ├── ip.py                 # PatentExpert (Gemini, USPTO)
│   │   │   └── tech.py               # TechExpert (GPT-4o, GitHub)
│   │   │
│   │   ├── agents/             # Agent Layer
│   │   │   ├── orchestrator/         # Coordinates experts
│   │   │   │   ├── __init__.py
│   │   │   │   └── agent.py          # OrchestratorAgent
│   │   │   └── research/             # Legacy single-agent (deprecated)
│   │   │       ├── __init__.py
│   │   │       ├── agent.py
│   │   │       └── toolkits.py
│   │   │
│   │   ├── core/               # Shared Utilities
│   │   │   ├── config.py       # Configuration loader
│   │   │   ├── types.py        # Pydantic models
│   │   │   ├── errors.py       # Custom exceptions
│   │   │   └── retry.py        # Tenacity decorators
│   │   │
│   │   ├── server/             # API Layer
│   │   │   ├── api/
│   │   │   │   ├── research.py # /api/v1/research endpoints
│   │   │   │   ├── data.py     # /api/v1/data endpoints
│   │   │   │   └── health.py   # Health checks
│   │   │   ├── middleware/
│   │   │   │   ├── auth.py     # Supabase JWT validation
│   │   │   │   └── errors.py   # Global error handlers
│   │   │   └── main.py         # FastAPI entry point
│   │   │
│   │   └── workers/            # Background Workers
│   │       └── analysis_worker.py  # Celery task consumer
│   │
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── conftest.py
│   │
│   └── pyproject.toml
│
├── finbert/                    # Separate FinBERT Microservice
│   ├── Dockerfile
│   ├── main.py                 # FastAPI service
│   └── requirements.txt
│
├── docker/
│   ├── Dockerfile              # Main backend
│   └── docker-compose.yml
│
├── migrations/                 # Alembic migrations
│   ├── alembic.ini
│   └── versions/
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Build Order

Implementation follows this dependency chain:

| Phase | Module | Reference | Dependencies |
|-------|--------|-----------|--------------|
| 1 | Database Layer | `02_DATABASE_SCHEMA.md` | Supabase project |
| 2 | Core Architecture | `01_ARCHITECTURE.md` | Config, Types, Errors |
| 3 | Model Factory | `07_MODEL_MANAGEMENT.md` | OpenRouter credentials |
| 4 | Domain Services | `11_DOMAIN_SERVICES.md` | Core types |
| 5 | MCP Bridge | `04_MODULE_MCP_BRIDGE.md` | sec-edgar-mcp running |
| 6 | FinBERT Service | `05_MODULE_FINBERT.md` | Separate container |
| 7 | Prompt Engineering | `10_PROMPT_ENGINEERING.md` | Template files |
| 8 | Agent Orchestration | `06_AGENT_ORCHESTRATION.md` | All services, toolkits |
| 9 | API Layer | `03_API_SPECIFICATION.md` | Agent, Auth |
| 10 | Workers | `12_BACKGROUND_WORKERS.md` | Redis, Celery |
| 11 | Environment & Deploy | `08_ENV_AND_DEPLOYMENT.md` | Docker, Cloud Run |
| 12 | Testing | `09_TESTING_STRATEGY.md` | Mocks, fixtures |

---

## Documentation Reference

All specifications are located in `docs/specs/`. Below is a complete index with descriptions.

### Core Architecture

| Document | Description | Key Topics |
|----------|-------------|------------|
| [01_ARCHITECTURE_AND_STRUCTURE.md](specs/01_ARCHITECTURE_AND_STRUCTURE.md) | System architecture and layered design | Layer responsibilities, separation of concerns, data flow diagrams, error boundaries |
| [02_DATABASE_SCHEMA.md](specs/02_DATABASE_SCHEMA.md) | Supabase/PostgreSQL schema design | ERD, SQL schema, pgvector setup, RLS policies, Alembic migrations |
| [03_API_SPECIFICATION.md](specs/03_API_SPECIFICATION.md) | FastAPI REST endpoints | Authentication, /analyze, /stream SSE, webhooks, rate limiting |

### Modules & Services

| Document | Description | Key Topics |
|----------|-------------|------------|
| [04_MODULE_MCP_BRIDGE.md](specs/04_MODULE_MCP_BRIDGE.md) | SEC data integration via MCP | MCPClient, SECToolkit, retry logic, sec-edgar-mcp tools reference |
| [05_MODULE_FINBERT.md](specs/05_MODULE_FINBERT.md) | FinBERT sentiment analysis | Hybrid architecture, FinBERTAnalyzer, HTTP client, microservice Dockerfile |
| [11_DOMAIN_SERVICES.md](specs/11_DOMAIN_SERVICES.md) | Pure business logic services | DriftCalculator, HeatScorer, ZoneClassifier, TextChunker, formulas |

### Agent & LLM

| Document | Description | Key Topics |
|----------|-------------|------------|
| [06_AGENT_ORCHESTRATION.md](specs/06_AGENT_ORCHESTRATION.md) | Agno agent implementation | Thin Agent Pattern, Research Agent, toolkit registration, workflow diagrams |
| [07_MODEL_MANAGEMENT.md](specs/07_MODEL_MANAGEMENT.md) | LLM provider abstraction | OpenRouter adapter, Circuit Breaker, Model Factory, structured outputs |
| [10_PROMPT_ENGINEERING.md](specs/10_PROMPT_ENGINEERING.md) | Prompt template management | Jinja2 templates, PromptManager, research agent system prompt |
| [13_EXPERT_REGISTRY.md](specs/13_EXPERT_REGISTRY.md) | **Team of Experts architecture** | BaseExpert, ExpertRegistry, Orchestrator, domain experts (Risk, IP, Tech) |

### Infrastructure & Operations

| Document | Description | Key Topics |
|----------|-------------|------------|
| [08_ENV_AND_DEPLOYMENT.md](specs/08_ENV_AND_DEPLOYMENT.md) | Environment and deployment | .env.example, Docker multi-stage builds, Cloud Run, GitHub Actions CI/CD |
| [09_TESTING_STRATEGY.md](specs/09_TESTING_STRATEGY.md) | Testing approach | Test pyramid, fixtures, mocks (FinBERT, LLM), coverage requirements |
| [12_BACKGROUND_WORKERS.md](specs/12_BACKGROUND_WORKERS.md) | Async task processing | Celery configuration, Redis PubSub, SSE streaming, Flower monitoring |

### Guides

| Document | Description |
|----------|-------------|
| [IMPLEMENTATION_GUIDE.md](guides/IMPLEMENTATION_GUIDE.md) | Step-by-step instructions for building from scratch (8 phases, ~4 weeks) |

### Quick Reference

```
docs/
├── 00_MASTER_PLAN.md           ← You are here
├── risk_drift.py               ← Reference algorithm for domain services
├── guides/
│   └── IMPLEMENTATION_GUIDE.md
└── specs/
    ├── 01_ARCHITECTURE_AND_STRUCTURE.md
    ├── 02_DATABASE_SCHEMA.md
    ├── 03_API_SPECIFICATION.md
    ├── 04_MODULE_MCP_BRIDGE.md
    ├── 05_MODULE_FINBERT.md
    ├── 06_AGENT_ORCHESTRATION.md
    ├── 07_MODEL_MANAGEMENT.md
    ├── 08_ENV_AND_DEPLOYMENT.md
    ├── 09_TESTING_STRATEGY.md
    ├── 10_PROMPT_ENGINEERING.md
    ├── 11_DOMAIN_SERVICES.md
    ├── 12_BACKGROUND_WORKERS.md
    └── 13_EXPERT_REGISTRY.md     ← NEW: Team of Experts
```

---

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Expert Architecture** | Team of Experts + Orchestrator | Modular domain specialization; each expert uses optimal model |
| LLM Provider | OpenRouter | Single SDK for GPT-4o, Claude, DeepSeek |
| Fallback Strategy | Circuit Breaker | Auto-switch after 3 failures for 5 minutes |
| Output Format | Pydantic Structured | Type-safe Supabase inserts |
| FinBERT Arch | Hybrid (in-proc dev, service prod) | Avoid blocking FastAPI event loop |
| Error Handling | Tenacity decorators | Configurable retry policies |
| Frontend Errors | SSE Error Events | Non-fatal error propagation |
| Migrations | Alembic | Version-controlled schema changes |
| Auth | Supabase JWT | Unified auth with frontend |
| Scaling | Redis + Celery | Queue-based job processing |
| Deployment | Docker Compose → Cloud Run | Scale-to-zero, cost-efficient |