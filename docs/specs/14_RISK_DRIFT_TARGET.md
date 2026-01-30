# 14. Risk Drift: Target Output vs Implementation

## Overview

The canonical result of a risk drift analysis is defined in [full_analysis.json](full_analysis.json) (NVDA example). The API/job result for risk drift **must** return this shape. This spec documents the target structure, current implementation gaps, and required changes to align code with the spec.

**Reference:** [docs/01_REVISED_PLAN_RISK_DRIFT.md](../01_REVISED_PLAN_RISK_DRIFT.md)

---

## Target Output: full_analysis.json

| Section | Description |
|--------|-------------|
| **meta** | total_risks_current, total_risks_prev, drift_count, filing_date |
| **visual_priority_map** | Ordered list: risk (title), rank, previous_rank, change (delta), status ("Stable", "New", "Fell", "Climbed", "Removed") |
| **sentiment_shift_indicators** | Per-risk: score, confidence, shift_detected, shift_type, analysis, strategic_recommendation, original_snippet, new_snippet |
| **materiality_flags** | risk, alert, details, recommendation, intensity ("LOW" / "MEDIUM" / "HIGH") |
| **removed_risks** | risk, rank_prev, status, snippet |
| **heatmap** | heatmap_title, generated_at, zones (critical_red, warning_orange, new_blue, stable_gray) — each zone holds entries with risk_id, title, current_rank, rank_delta (number or "NEW"), modality_shift, heat_score (0–100), summary |

---

## Current vs Target

| What | Current | Target |
|------|---------|--------|
| Job/API result | SynthesizedReport (company, summary text, expert_findings, risk_level, recommendations). Job stores result_summary.content. | Single JSON: meta, visual_priority_map, sentiment_shift_indicators, materiality_flags, removed_risks, heatmap. |
| Structured drift | Not produced. Expert tools return drift-like dicts; nothing assembles the full structure. | Explicit sections for priority map, sentiment shifts, materiality, removed risks, zoned heatmap. |
| Heatmap zones | DriftCalculator does not set heat_score or zone; ReportGenerator not fed by pipeline. | heatmap.zones with critical_red, warning_orange, new_blue, stable_gray and per-entry fields. |
| Snippets / recommendations | Not in drift path. | sentiment_shift_indicators and materiality_flags include snippets and strategic recommendations. |

---

## Data Models (Spec vs Implementation)

### RiskFactor

| Aspect | Spec (docs/risk_drift.py) | Implementation |
|--------|---------------------------|-----------------|
| Fields | title, content, rank, embedding, filing_date | drift_calculator.RiskFactor: rank, title, content, embedding, sentiment_score. No filing_date. core.types.RiskFactor has section_order, word_count, no rank. |

### DriftResult / RiskDrift

| Aspect | Spec | Implementation |
|--------|------|----------------|
| Fields | risk_title, rank_current, rank_prev, rank_delta, semantic_score, drift_type, modality_shift, analysis, strategic_recommendation, semantic_intensity, heat_score (0–100), original_text_snippet, new_text_snippet, confidence_score | DriftResult: risk_title, rank_current, rank_previous, rank_delta, semantic_score, drift_type, heat_score (float 0–1), zone, analysis. Missing: modality_shift, strategic_recommendation, semantic_intensity, text snippets, confidence_score. |

### Heatmap Output

| Aspect | Spec | Implementation |
|--------|------|----------------|
| Structure | heatmap_zones: critical_red, warning_orange, new_blue, stable_gray with entries (risk_id, title, current_rank, rank_delta, modality_shift, heat_score, summary); heatmap_title, generated_at | ReportGenerator consumes flat list of drift dicts; no zones structure. core.types has HeatmapEntry/HeatmapData with zones but no code builds this from the drift pipeline. |

---

## Risk Factor Extraction

| Aspect | Spec | Implementation |
|--------|------|----------------|
| Method | LLM (Agno Agent + Gemini) extracts titles from Item 1A; text split by titles for content. Retries, TOC/summary detection. | RiskFactorParser: regex-only (bold/plain header patterns). No LLM. |
| Location | RiskDriftAnalyzer.extract_risk_factors() (async) | RiskFactorParser.extract_risks() (sync) in filing service. |
| Gap | Spec uses LLM for robustness; implementation uses heuristics only. Parser not used in main analysis pipeline. |

---

## Embeddings

| Aspect | Spec | Implementation |
|--------|------|----------------|
| Computation | compute_embeddings() on RiskFactor list via Google text-embedding-004, batched (chunk size 5). | EmbeddingService + GoogleEmbeddingAdapter exist; no path embeds risk factors and passes them into DriftCalculator. |
| Usage | Cosine similarity between current/previous embeddings when comparing risks. | DriftCalculator supports embedding and _cosine_similarity when title match is weak; pipeline does not fill embeddings before drift. |

---

## Drift Analysis Logic

| Aspect | Spec | Implementation |
|--------|------|----------------|
| Matching | Fuzzy title (rapidfuzz fuzz.ratio), threshold 85; then rank delta and optional semantic/sentiment. | DriftCalculator: fuzz.token_set_ratio (0–1), FUZZY_THRESHOLD 0.75; SEMANTIC_THRESHOLD 0.80 when embeddings present. |
| Drift types | structural, semantic, new, removed, stable | rank_change, stable, new, removed (no structural/semantic labels). |
| Removed risks | Separate list removed_risks (risk, rank_prev, status, snippet). | Removed as DriftResult with drift_type="removed" in same list. |
| Heat score | Formula: (max(0, climb_magnitude) * 2) + sentiment_impact, capped 100; per drift. | DriftCalculator.analyze_drift() never sets heat_score or zone (default 0.0, "stable_gray"). HeatScorer and ZoneClassifier exist but not called in calculator. |
| Zone thresholds | climb > 5 → critical_red; >= 1 → warning_orange; new → new_blue. | ZoneClassifier: abs_delta >= 4 → critical_red; >= 2 → warning_orange; new → new_blue. Not used in DriftCalculator. |

---

## Sentiment and Modality

| Aspect | Spec | Implementation |
|--------|------|----------------|
| FinBERT | _get_finbert_score(text) via HuggingFace (ProsusAI/finbert), 0–1 negative intensity; used for modality and strategic rec. | SentimentToolkit + FinBERT available as agent tool; not invoked in drift pipeline; no sentiment passed into DriftCalculator or heat formula. |
| Modality shift | _check_modality_shift(old, new, sentiment) via LLM: probabilistic vs deterministic language. | Not implemented in drift path. |
| Strategic recommendation | _generate_strategic_rec(title, content, analysis, sentiment) ("So What?"). | Not implemented in drift path. |

---

## Pipeline and Orchestration

| Aspect | Spec | Implementation |
|--------|------|----------------|
| Flow | Single RiskDriftAnalyzer: extract_risk_factors (LLM) ×2 → compute_embeddings → analyze_drift → heatmap zones. | No extract→embed→drift→heatmap pipeline. Orchestrator runs experts; RiskExpert has tools; no automatic fetch of two filings → parse → embed → drift → heatmap. |
| Output | One class returns { drifts, removed_risks, heatmap } with zones. | Orchestrator synthesizes summary. ReportGenerator expects drift list with heat_score/zone but is not called with such data; API returns SynthesizedReport, not full_analysis. |

---

## Gaps to Address (Implementation Checklist)

1. **Wire heat and zone into drift:** In `DriftCalculator.analyze_drift()`, call `HeatScorer.compute()` and `ZoneClassifier.classify()` per drift; set `DriftResult.heat_score` and `DriftResult.zone` (and optionally sentiment when available).

2. **Optional: dedicated drift pipeline:** Add a path (service or task) that: fetches Item 1A for two years → extracts risks (parser or future LLM) → optionally runs embeddings → runs DriftCalculator → builds zone-based heatmap dict → passes result to ReportGenerator or stores in job result.

3. **Optional: LLM-based extraction:** Replace or complement RiskFactorParser with LLM-based title extraction and keep content splitting from spec.

4. **Optional: embeddings in pipeline:** Ensure risk factors are embedded (e.g. `EmbeddingService.embed_batch`) before calling DriftCalculator when semantic matching is desired.

5. **Optional: modality and "So What?":** Implement _check_modality_shift and _generate_strategic_rec (or equivalents); add modality_shift and strategic_recommendation to DriftResult and heatmap entries.

---

## Files Reference

| Area | Path |
|------|------|
| **Target schema** | [full_analysis.json](full_analysis.json) |
| **Original spec** | docs/risk_drift.py — RiskFactor, RiskDrift, RiskDriftAnalyzer |
| **Extraction** | python/insights/services/filing/parser.py — RiskFactorParser.extract_risks() |
| **Drift** | python/insights/services/risk/drift_calculator.py — DriftCalculator, DriftResult |
| **Heat/zone** | python/insights/services/risk/heat_scorer.py, zone_classifier.py |
| **Tools** | python/insights/experts/toolkits/risk.py — analyze_risk_drift, calculate_heat_score |
| **Orchestration** | python/insights/agents/orchestrator/agent.py, python/insights/workers/tasks.py |
| **Report** | python/insights/services/report/generator.py |
