# 🌌 InSights-ai

**Deep Research & Financial Analysis Platform**

InSights-ai is a cutting-edge financial analysis engine built as a **"Team of Experts managed by an Orchestrator."** It leverages specialized AI agents to analyze SEC filings, sentiment, and market data, providing deep insights through a modular, agentic architecture.

> [!IMPORTANT]
> **Constraint:** This is an analysis-only platform. Strictly NO trading execution capabilities are implemented.

---

## 🚀 Key Features

- **🛡️ Team of Experts Architecture**: A modular system where an Orchestrator routes complex queries to specialized Domain Experts (Risk, IP, Tech, Macro).
- **📉 Risk Drift Analysis**: Advanced comparison of SEC filings (10-K/Q) over time with heat score calculation and zone classification (Critical, Warning, New, Stable).
- **🎭 Multi-Model Intelligence**: Each expert uses the optimal model for its task (e.g., GPT-4o for Risk, Claude for Tech, FinBERT for Sentiment).
- **🔌 SEC EDGAR Integration**: Real-time data fetching via [sec-edgar-mcp](https://github.com/stefanoamorelli/sec-edgar-mcp) using the Model Context Protocol.
- **📊 Financial Sentiment**: Integrated FinBERT microservice for high-precision financial context sentiment analysis.
- **⚡ Async & Streaming**: Parallel expert execution with SSE (Server-Sent Events) for real-time response streaming.
- **🗄️ Vector Intelligence**: Supabase (pgvector) integration for semantic search and embedding storage.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Core** | Python 3.12+, FastAPI, [Agno](https://agno.com/) |
| **Database** | Supabase (PostgreSQL + pgvector) |
| **LLMs** | OpenRouter (GPT-4o, Claude 3.5, Gemini 1.5, DeepSeek) |
| **Sentiment** | FinBERT (Transformers + PyTorch) |
| **Data** | SEC EDGAR MCP, Brave Search |
| **Infra** | Docker, Redis, Celery |
| **Build** | `uv` (Package management) |

---

## 📂 Project Structure

```text
.
├── configs/                # YAML/Jinja2 configuration for agents & experts
├── docker/                 # Service orchestration & Dockerfiles
├── docs/                   # Full technical specifications & research
├── finbert/                # Specialized sentiment analysis microservice
├── python/                 # Core backend logic (FastAPI + Agno Agents)
│   ├── insights/           # Main package
│   │   ├── adapters/       # Infrastructure (DB, MCP, LLM Factory)
│   │   ├── agents/         # Orchestrator & Research Logic
│   │   ├── core/           # Shared utilities & config
│   │   ├── experts/        # Domain Expert Registry
│   │   ├── server/         # FastAPI Routes & API
│   │   └── services/       # Domain Business Logic
│   └── tests/              # Comprehensive test suite
└── supabase/               # Database migrations & configuration
```

---

## 🚦 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) & Docker Compose
- [uv](https://github.com/astral-sh/uv) (for local development)
- Supabase Account
- OpenRouter API Key

### Configuration

1. **Clone the repository**
2. **Setup environment variables**:
   Create a `.env` file in the root directory (refer to `python/.env.example`):
   ```bash
   SUPABASE_URL=your_url
   SUPABASE_ANON_KEY=your_key
   OPENROUTER_API_KEY=your_key
   GOOGLE_API_KEY=your_key
   ```

### Running with Docker (Recommended)

The easiest way to start the entire ecosystem is via Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up -d
```

This will spin up:
- **Backend API** (port 8000)
- **FinBERT Service** (port 8001)
- **SEC MCP Server** (port 8080)
- **Redis** & **Celery Worker**

### Local Development

If you prefer running the backend manually:

```bash
cd python
uv venv
source .venv/bin/activate
uv pip install -e .
python main.py
```

---

## 🧪 Testing

The project uses `pytest` for rigorous testing of agents and services.

```bash
cd python
pytest
```

---

## 📖 Documentation

For detailed technical specifications, refer to the `docs/` folder:
- [Master Plan](docs/00_MASTER_PLAN.md)
- [Architecture & Structure](docs/specs/01_ARCHITECTURE_AND_STRUCTURE.md)
- [Expert Registry System](docs/specs/13_EXPERT_REGISTRY.md)
- [Database Schema](docs/specs/02_DATABASE_SCHEMA.md)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
