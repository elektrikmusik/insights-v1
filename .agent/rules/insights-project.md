# InSights-ai Project Rules

## Project Architecture

This project follows a **Team of Experts** architecture where:
- An **Orchestrator Agent** coordinates specialized domain experts
- Each **Expert** (Risk, IP, Tech, Macro) has its own model, tools, and prompts
- **Domain Services** contain pure business logic (no LLM calls)
- **Adapters** wrap external infrastructure (Supabase, MCP, LLM providers)

## Documentation Reference

Before implementing any feature, consult the specs in `docs/specs/`:
1. `01_ARCHITECTURE_AND_STRUCTURE.md` - Layer responsibilities
2. `02_DATABASE_SCHEMA.md` - Supabase tables and RLS
3. `06_AGENT_ORCHESTRATION.md` - Thin Agent Pattern
4. `07_MODEL_MANAGEMENT.md` - Multi-provider LLM (OpenRouter, SiliconFlow)
5. `11_DOMAIN_SERVICES.md` - Business logic patterns
6. `13_EXPERT_REGISTRY.md` - Expert system design

## Code Organization

```
python/insights/
├── adapters/       # Infrastructure (DB, MCP, LLM)
├── services/       # Domain logic (risk/, filing/, sentiment/)
├── experts/        # Expert implementations
├── agents/         # Orchestrator only
├── core/           # Shared utilities
└── server/         # FastAPI endpoints
```

## Key Constraints

1. **NO trading execution** - Analysis only platform
2. **Thin Agents** - Agents reason, services execute
3. **Structured Outputs** - All LLM responses use Pydantic models
4. **Async-First** - All I/O is async
5. **Config-Driven** - Experts defined in YAML, prompts in Jinja2

## LLM Provider Rules

- Use **OpenRouter** for GPT-4o, Claude, via `openai/` or `anthropic/` prefix
- Use **SiliconFlow** for DeepSeek, Qwen via `deepseek-ai/` or `Qwen/` prefix
- Each expert can use a different model based on cost/capability
- Circuit breakers are per-provider

## Testing Requirements

- Unit tests for all domain services (80%+ coverage)
- Mock LLM responses in tests
- Mock MCP calls in tests
- Integration tests for API endpoints
