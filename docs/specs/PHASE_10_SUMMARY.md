# Phase 10: DB Population for Risk Drift - Implementation Summary

**Status:** ✅ Complete  
**Date Completed:** 2026-01-29  
**Reference:** [15_DB_POPULATION_RISK_DRIFT.md](15_DB_POPULATION_RISK_DRIFT.md)

---

## Overview

Phase 10 implemented a comprehensive DB-first risk drift pipeline that:
- Avoids redundant fetching and computation by checking the database first
- Populates all required tables (companies, filings, risk_factors, risk_drifts, reports)
- Builds the complete `full_analysis.json` structure
- Generates markdown reports with heat scores and zones

---

## Key Deliverables

### 1. Risk Drift Pipeline Service

**File:** `python/insights/services/risk_drift/pipeline.py`

A complete pipeline orchestrator with the following capabilities:

#### Core Features:
- **DB-First Architecture:** Checks database before fetching from SEC or computing embeddings
- **Company Management:** `_get_or_create_company()` - fetches from SEC if not in DB
- **Filing Management:** `_get_or_fetch_filing()` - reuses cached filings with raw_text
- **Risk Factor Extraction:** `_get_or_extract_risk_factors()` - reuses cached risk factors
- **Embedding Management:** `_ensure_embeddings()` - only computes where `embedding IS NULL`
- **Drift Calculation:** Uses Phase 9's enhanced DriftCalculator with heat/zone
- **Full Analysis Builder:** `_build_full_analysis()` - constructs complete JSON structure
- **Database Persistence:** `_save_risk_drifts()` - saves non-removed drifts to DB

#### Data Flow:
```
Input: ticker, years
  ↓
Get/Create Company (DB-first)
  ↓
Get/Fetch Filings (DB-first, reuse raw_text)
  ↓
Get/Extract Risk Factors (DB-first)
  ↓
Ensure Embeddings (only where NULL)
  ↓
Run Drift Calculator (with IDs)
  ↓
Build full_analysis JSON
  ↓
Save risk_drifts (non-removed)
  ↓
Generate Markdown Report
  ↓
Return: full_analysis + markdown
```

### 2. Worker Integration

**File:** `python/insights/workers/tasks.py`

Updated `_analyze_risk_drift_async()` to:
- Use `RiskDriftPipeline` instead of orchestrator
- Set `result_summary = full_analysis` (not wrapped in `{"content": ...}`)
- Call `db.save_report()` with:
  - `markdown_content` from pipeline
  - `summary = full_analysis` for JSONB storage
  - Full parameters (ticker, years)

**Benefits:**
- Job queue now contains complete `full_analysis` in `result_summary`
- Reports table stores both markdown and structured JSON
- Clients receive full analysis via GET /jobs/:id or SSE

### 3. Full Analysis Structure

The `full_analysis` JSON matches [full_analysis.json](full_analysis.json) specification:

```json
{
  "meta": {
    "total_risks_current": 17,
    "total_risks_prev": 18,
    "drift_count": 4,
    "filing_date": "2024-02-21"
  },
  "visual_priority_map": [
    {
      "risk": "Risk title",
      "rank": 1,
      "previous_rank": 2,
      "change": 1,
      "status": "Climbed"
    }
  ],
  "sentiment_shift_indicators": [
    {
      "risk": "Risk title",
      "score": 0.0,
      "confidence": 0.9,
      "shift_detected": true,
      "shift_type": "rank_change",
      "analysis": "Analysis text",
      "strategic_recommendation": "Recommendation",
      "original_snippet": "...",
      "new_snippet": "..."
    }
  ],
  "materiality_flags": [
    {
      "risk": "Risk title",
      "alert": "Material Drift Detected",
      "details": "Details",
      "recommendation": "Action",
      "intensity": "HIGH"
    }
  ],
  "removed_risks": [
    {
      "risk": "Old risk",
      "rank_prev": 3,
      "status": "Removed",
      "snippet": "..."
    }
  ],
  "heatmap": {
    "heatmap_title": "Risk Profile Evolution",
    "generated_at": "2026-01-28T08:32:37.884321+00:00",
    "zones": {
      "critical_red": [],
      "warning_orange": [],
      "new_blue": [...],
      "stable_gray": [...]
    }
  }
}
```

### 4. Database Population

The pipeline populates:

| Table | What's Saved | How |
|-------|-------------|-----|
| **companies** | Ticker, CIK, name, sector | `get_or_create_company()` |
| **filings** | Accession, date, fiscal_year, raw_text (Item 1A) | `save_filing()` |
| **risk_factors** | Title, content, rank, embedding, word_count | `save_risk_factors()` + `update_risk_factor_embedding()` |
| **risk_drifts** | All DriftResult fields (non-removed) | `save_risk_drifts()` |
| **job_queue** | `result_summary = full_analysis` | `update_job_status()` |
| **reports** | Markdown + `summary = full_analysis` | `save_report()` |

**Note:** Removed risks are stored only in JSON (not as risk_drifts rows) because schema requires `risk_factor_id NOT NULL`.

### 5. Testing

#### Unit Tests
**File:** `python/tests/unit/test_risk_drift_pipeline.py`

Tests for:
- `_build_full_analysis()` structure
- Status label generation
- Risk ID generation
- Heatmap zone distribution

#### Integration Tests
**File:** `python/tests/integration/test_risk_drift_pipeline.py`

Tests for:
- End-to-end pipeline execution
- Database population verification
- DB-first caching behavior
- Drift count consistency
- Removed risks capture
- Markdown report generation

---

## DB-First Logic (Cost Savings)

The pipeline implements aggressive caching to minimize:
- SEC MCP calls
- LLM calls (embeddings, sentiment)
- Duplicate extraction work

### Caching Strategy:

1. **Company:** Check `get_company_by_ticker()` → use if exists, else fetch from SEC
2. **Filings:** Check `get_filings_by_company()` → if `raw_text` exists, use it (don't fetch from SEC)
3. **Risk Factors:** Check `get_risk_factors_by_filing()` → if present, use (don't re-extract)
4. **Embeddings:** Only call `embed_batch()` for factors where `embedding IS NULL`
5. **Sentiment:** (Optional future) Only compute where missing

**Result:** Subsequent runs for the same ticker/years are nearly instant and free.

---

## API Integration

### Job Result Format

GET `/api/v1/jobs/{job_id}` now returns:

```json
{
  "id": "uuid",
  "status": "completed",
  "result_summary": {
    "meta": {...},
    "visual_priority_map": [...],
    "sentiment_shift_indicators": [...],
    "materiality_flags": [...],
    "removed_risks": [...],
    "heatmap": {...}
  }
}
```

### SSE Stream

`/api/v1/stream/jobs/{job_id}` also exposes `result_summary` in completion event.

---

## Verification Commands

```bash
# Import verification
cd python && uv run python -c "
from insights.services.risk_drift import RiskDriftPipeline
pipeline = RiskDriftPipeline()
print('✓ Pipeline ready')
"

# Unit tests
uv run pytest tests/unit/test_risk_drift_pipeline.py -v

# Integration tests (requires test DB + MCP)
uv run pytest tests/integration/test_risk_drift_pipeline.py -v -m integration

# Full suite
uv run pytest tests/ -v
```

---

## Known Limitations & Future Enhancements

### Current Limitations:
1. **Sentiment Analysis:** Currently using placeholder values (score=0.0). Future: integrate FinBERT for actual sentiment
2. **Modality Shift:** Detecting via LLM not yet implemented (shows "none")
3. **Strategic Recommendations:** Currently brief; future: enhance with LLM-generated strategic insights

### Future Enhancements (Post-Phase 10):
1. **FinBERT Integration:** Compute actual sentiment scores for `sentiment_shift_indicators`
2. **Modality Detection:** LLM-based detection of probabilistic → deterministic shifts
3. **Enhanced Strategic Rec:** Deeper "So What?" analysis via LLM
4. **Caching Layer:** Redis cache for embeddings/sentiment to avoid DB lookups
5. **Batch Processing:** Multi-company batch analysis
6. **Historical Trends:** Multi-period drift tracking beyond 2 years

---

## Files Modified/Created

### Created:
- `python/insights/services/risk_drift/__init__.py`
- `python/insights/services/risk_drift/pipeline.py` (650+ lines)
- `python/tests/unit/test_risk_drift_pipeline.py`
- `python/tests/integration/test_risk_drift_pipeline.py`
- `docs/specs/PHASE_10_SUMMARY.md` (this file)

### Modified:
- `python/insights/workers/tasks.py` (updated `_analyze_risk_drift_async`)
- `docs/plans/IMPLEMENTATION_PLAN.md` (marked Phase 10 complete)

---

## Success Metrics

✅ **All Phase 10 Checklist Items Complete:**
- [x] DB-first company/filing/risk factor pipeline
- [x] Embeddings computed only where missing
- [x] DriftCalculator integration with IDs
- [x] full_analysis JSON builder
- [x] risk_drifts persistence
- [x] result_summary = full_analysis
- [x] save_report with summary
- [x] Worker integration
- [x] Unit tests
- [x] Integration tests

✅ **Verification:**
- Imports work
- No linter errors
- Pipeline instantiates correctly
- Database schema aligned

---

## Next Steps (Post-Phase 10)

1. **Production Testing:** Run pipeline against live data in staging environment
2. **Performance Optimization:** Profile DB queries, add indices as needed
3. **Monitoring:** Add observability for pipeline execution times
4. **Documentation:** Update API docs with new result_summary structure
5. **Sentiment Integration:** Begin Phase 11 (if planned) for FinBERT integration

---

**Phase 10 Status:** ✅ **COMPLETE**  
**All 10 Phases of Initial Implementation Plan:** ✅ **COMPLETE**
