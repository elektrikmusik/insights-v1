---
description: Build InSights-ai backend following the 10-phase implementation plan
---

# InSights-ai Build Workflow

Reference: `docs/guides/IMPLEMENTATION_GUIDE.md`, `docs/plans/IMPLEMENTATION_PLAN.md`

## Phase Selection

Ask the user which phase to work on:

1. **Phase 1: Project Setup** - uv, directory structure, configs
2. **Phase 2: Database** - Supabase schema, Alembic migrations
3. **Phase 3: Core Infrastructure** - config loader, error types, retry
4. **Phase 4: Domain Services** - risk/, filing/, sentiment/ services
5. **Phase 5: External Integrations** - MCP client, FinBERT, LLM adapters
6. **Phase 6: Agent Layer** - Experts, Registry, Orchestrator
7. **Phase 7: API & Workers** - FastAPI endpoints, Celery tasks
8. **Phase 8: Testing & Deployment** - pytest, Docker, CI/CD
9. **Phase 9: Risk Drift Target Alignment** - full_analysis shape, heat/zone in DriftCalculator, optional embeddings/modality
10. **Phase 10: DB Population for Risk Drift** - dedicated pipeline, DB-first, result_summary = full_analysis, save_report

## Workflow Steps

// turbo
1. Read the relevant spec files for the selected phase
2. Check existing code in `python/insights/` for conflicts
3. Create directory structure if missing
4. Implement in order: types → adapters → services → agents → API
5. Write tests alongside implementation
6. Run `uv run pytest` to verify
7. Update `docs/plans/IMPLEMENTATION_PLAN.md` with completion status

## Phase-Specific Specs

| Phase | Primary Specs |
|-------|---------------|
| 1 | 01_ARCHITECTURE, 08_ENV_AND_DEPLOYMENT |
| 2 | 02_DATABASE_SCHEMA |
| 3 | 01_ARCHITECTURE (core/ section) |
| 4 | 11_DOMAIN_SERVICES |
| 5 | 04_MODULE_MCP_BRIDGE, 05_MODULE_FINBERT, 07_MODEL_MANAGEMENT |
| 6 | 06_AGENT_ORCHESTRATION, 13_EXPERT_REGISTRY |
| 7 | 03_API_SPECIFICATION, 12_BACKGROUND_WORKERS |
| 8 | 09_TESTING_STRATEGY, 08_ENV_AND_DEPLOYMENT |
| 9 | 14_RISK_DRIFT_TARGET, full_analysis.json, 01_REVISED_PLAN_RISK_DRIFT |
| 10 | 15_DB_POPULATION_RISK_DRIFT, 14_RISK_DRIFT_TARGET, 02_DATABASE_SCHEMA, 02_REVISED_PLAN_DB |

## Verification

After each phase:
// turbo
- Run `uv run ruff check python/`
- Run `uv run mypy python/insights`
- Run `uv run pytest python/tests`
