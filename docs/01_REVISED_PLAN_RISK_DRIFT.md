# Risk Drift: Target Output vs Implementation

**Target:** `full_analysis.json`

The canonical result of a risk drift analysis is defined in `docs/specs/full_analysis.json` (NVDA example). The API/job result for risk drift must return this shape:

- **meta:** total_risks_current, total_risks_prev, drift_count, filing_date
- **visual_priority_map:** Ordered list of risks with risk (title), rank, previous_rank, change (delta), status ("Stable", "New", "Fell", "Climbed", "Removed")
- **sentiment_shift_indicators:** Per-risk entries with score, confidence, shift_detected, shift_type, analysis, strategic_recommendation, original_snippet, new_snippet
- **materiality_flags:** risk, alert, details, recommendation, intensity ("LOW" / "MEDIUM" / "HIGH")
- **removed_risks:** risk, rank_prev, status, snippet
- **heatmap:** heatmap_title, generated_at, zones (critical_red, warning_orange, new_blue, stable_gray) — each zone holds entries with risk_id, title, current_rank, rank_delta (number or "NEW"), modality_shift, heat_score (0–100), summary

---

## Current vs Target

| What | Current implementation | Target (full_analysis.json) |
|------|------------------------|-----------------------------|
| Job/API result | SynthesizedReport (company, summary text, expert_findings, risk_level, recommendations). Job stores result_summary.content. | Single JSON object with meta, visual_priority_map, sentiment_shift_indicators, materiality_flags, removed_risks, heatmap. |
| Structured drift | Not produced. Expert tools return drift-like dicts; nothing assembles the full structure. | Explicit sections for priority map, sentiment shifts, materiality, removed risks, zoned heatmap. |
| Heatmap zones | DriftCalculator does not set heat_score or zone; ReportGenerator not fed by pipeline. | heatmap.zones with critical_red, warning_orange, new_blue, stable_gray and per-entry fields. |
| Snippets / recommendations | Not in drift path. | sentiment_shift_indicators and materiality_flags include snippets and strategic recommendations. |

The implementation is far from the target: it never builds or returns the full_analysis structure.

---

## Summary (original spec vs codebase)

The original `docs/risk_drift.py` describes a single, end-to-end **RiskDriftAnalyzer** that: extracts risk factors (LLM + parsing), computes embeddings, runs drift analysis with fuzzy + semantic matching, assigns heat scores and zones, and returns drifts + removed risks + a structured heatmap.

The codebase implements the same concepts in a **split architecture**: extraction is regex-based (no LLM), drift/heat/zone live in separate services, and heat/zone are never applied inside the drift pipeline. The orchestration flow does not run a dedicated "extract → drift → heatmap" pipeline; it runs experts (which can call tools) and synthesizes text, so the spec's heatmap output is not produced.

---

## 1. Data Models

| Aspect | Spec (docs/risk_drift.py) | Implementation |
|--------|---------------------------|----------------|
| RiskFactor | title, content, rank, embedding, filing_date | drift_calculator.RiskFactor: rank, title, content, embedding, sentiment_score. No filing_date. core.types.RiskFactor (Pydantic) has section_order, word_count, no rank. |
| RiskDrift / DriftResult | risk_title, rank_current, rank_prev, rank_delta, semantic_score, drift_type, modality_shift, analysis, strategic_recommendation, semantic_intensity, heat_score (0–100), original_text_snippet, new_text_snippet, confidence_score | DriftResult: risk_title, rank_current, rank_previous, rank_delta, semantic_score, drift_type, heat_score (float 0–1), zone, analysis. Missing: modality_shift, strategic_recommendation, semantic_intensity, text snippets, confidence_score. |
| Heatmap output | heatmap_zones: critical_red, warning_orange, new_blue, stable_gray with entries (risk_id, title, current_rank, rank_delta, modality_shift, heat_score, summary); plus heatmap_title, generated_at | ReportGenerator consumes a flat list of drift dicts and sorts by heat_score; it does not consume a zones structure. core.types has HeatmapEntry / HeatmapData with zones and sentiment, but no code builds this from the drift pipeline. |

---

## 2. Risk Factor Extraction

| Aspect | Spec | Implementation |
|--------|------|-----------------|
| Method | LLM (Agno Agent + Gemini) extracts titles only from Item 1A; then text is split by those titles to get content. Retries, TOC/summary detection, and content cleanup. | RiskFactorParser: regex-only (bold/plain header patterns, line-by-line). No LLM. |
| Location | RiskDriftAnalyzer.extract_risk_factors() (async) | RiskFactorParser.extract_risks() (sync) in filing service. |
| Gap | Spec uses LLM for robustness and TOC handling; implementation uses heuristics only. Parser is not used in the main analysis pipeline (orchestrator runs experts, which use SECToolkit; no call to RiskFactorParser in the flow). |

---

## 3. Embeddings

| Aspect | Spec | Implementation |
|--------|------|-----------------|
| Computation | compute_embeddings() on list of RiskFactor using Google GenAI text-embedding-004, batched (chunk size 5). | EmbeddingService + GoogleEmbeddingAdapter exist for generic text; no code path embeds risk factors and passes them into the drift calculator. |
| Usage in drift | Cosine similarity between current/previous embeddings when comparing risks. | DriftCalculator supports embedding on RiskFactor and uses _cosine_similarity when title match is weak; but nothing in the pipeline fills embeddings for risks before drift. |

---

## 4. Drift Analysis Logic

| Aspect | Spec | Implementation |
|--------|------|-----------------|
| Matching | Fuzzy title match (rapidfuzz fuzz.ratio), threshold 85; then rank delta and optional semantic/sentiment. | DriftCalculator: fuzz.token_set_ratio (normalized to 0–1), FUZZY_THRESHOLD = 0.75; if title weak and embeddings present, uses SEMANTIC_THRESHOLD = 0.80. |
| Drift types | structural, semantic, new, removed, stable | rank_change, stable, new, removed (no "structural"/"semantic" labels). |
| Removed risks | Returned as separate list removed_risks (risk, rank_prev, status, snippet). | Removed risks are appended as DriftResult with drift_type="removed" (same list as current-risk drifts). |
| Heat score in drift | Formula: (max(0, climb_magnitude) * 2) + sentiment_impact, capped 100; stored per drift. | DriftCalculator.analyze_drift() never sets heat_score or zone; they remain default 0.0 and "stable_gray". HeatScorer and ZoneClassifier exist but are not called inside the calculator. |
| Zone thresholds | Spec: climb > 5 → critical_red; >= 1 → warning_orange; new → new_blue. | ZoneClassifier: abs_delta >= 4 → critical_red; >= 2 → warning_orange; new → new_blue; removed → warning_orange. Not used in DriftCalculator. |

---

## 5. Sentiment and Modality

| Aspect | Spec | Implementation |
|--------|------|-----------------|
| FinBERT | _get_finbert_score(text) via HuggingFace Inference API (ProsusAI/finbert), 0–1 negative intensity; used for modality and strategic rec. | SentimentToolkit + services: FinBERT available as agent tool (e.g. analyze_sentiment, get_sentiment_scores). Not invoked inside the drift pipeline; no sentiment passed into DriftCalculator or heat formula. |
| Modality shift | _check_modality_shift(old, new, sentiment) via LLM: probabilistic vs deterministic language. | Not implemented in drift path. |
| Strategic recommendation | _generate_strategic_rec(title, content, analysis, sentiment) ("So What?"). | Not implemented in drift path. |

---

## 6. Pipeline and Orchestration

| | Spec | Implementation |
|---|-----|-----------------|
| **Flow** | Single RiskDriftAnalyzer: extract_risk_factors (LLM) ×2 → compute_embeddings → analyze_drift → heatmap zones | No extract→embed→drift→heatmap pipeline. Orchestrator runs experts. RiskExpert has tools (SECToolkit, RiskToolkit, Sentiment). Agent may call analyze_risk_drift / calculate_heat_score. |
| **Output** | One class drives: extract (LLM) for current and previous text → embed → analyze_drift() → returns { drifts, removed_risks, heatmap } with zones. | Orchestrator gets company info, runs experts in parallel with a generic query, synthesizes summary. RiskExpert exposes tools; no automatic fetch of two filings → parse risks → embed → drift → heatmap. ReportGenerator.generate_markdown_report expects a list of drift dicts (with heat_score, zone) but is not called by the orchestrator with such data; the API returns SynthesizedReport (summary + expert findings), not the spec's heatmap structure. |

---

## 7. Gaps to Address (if aligning with spec)

1. **Wire heat and zone into drift:** In `DriftCalculator.analyze_drift()`, call `HeatScorer.compute()` and `ZoneClassifier.classify()` for each drift and set `DriftResult.heat_score` and `DriftResult.zone` (and optionally pass sentiment if available).

2. **Optional: dedicated drift pipeline:** Add a path (e.g. in a service or task) that: fetches Item 1A for two years → extracts risks (parser or future LLM) → optionally runs embeddings → runs DriftCalculator → builds zone-based heatmap dict → passes result to ReportGenerator or stores in job result.

3. **Optional: LLM-based extraction:** If you want spec fidelity, replace or complement RiskFactorParser with LLM-based title extraction (e.g. in a dedicated service) and keep content splitting logic from the spec.

4. **Optional: embeddings in pipeline:** Ensure risk factors are embedded (e.g. via `EmbeddingService.embed_batch`) before calling the drift calculator when semantic matching is desired.

5. **Optional: modality and "So What?":** Implement `_check_modality_shift` and `_generate_strategic_rec` (or equivalents) and add modality_shift and strategic_recommendation to the drift result model and heatmap entries if needed for the product.

---

## 8. Files Reference

| Area | Path |
|------|------|
| **Spec** | `docs/risk_drift.py` — RiskFactor, RiskDrift, RiskDriftAnalyzer (extract, embed, analyze_drift, FinBERT, modality, strategic rec). |
| **Extraction** | `python/insights/services/filing/parser.py` — RiskFactorParser.extract_risks(). |
| **Drift** | `python/insights/services/risk/drift_calculator.py` — DriftCalculator, RiskFactor, DriftResult (heat/zone not set). |
| **Heat/zone** | `python/insights/services/risk/heat_scorer.py`, `python/insights/services/risk/zone_classifier.py` — present but unused in drift. |
| **Tools** | `python/insights/experts/toolkits/risk.py` — analyze_risk_drift, calculate_heat_score (agent tools). |
| **Orchestration** | `python/insights/agents/orchestrator/agent.py`, `python/insights/workers/tasks.py`. |
| **Report** | `python/insights/services/report/generator.py` — expects drift list; not fed by current pipeline. |
