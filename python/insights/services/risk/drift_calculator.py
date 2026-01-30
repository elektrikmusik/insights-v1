"""
Risk Drift Calculator - Pure business logic for comparing risk factors.

Extracted algorithms:
- Fuzzy title matching (Levenshtein-based)
- Cosine similarity for semantic comparison
- Rank delta calculation
- Heat score and zone classification (full_analysis.json target)
"""
import re
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz

from insights.services.risk.heat_scorer import HeatScorer
from insights.services.risk.zone_classifier import ZoneClassifier

# Snippet length for original/new text in drift results
SNIPPET_MAX_LEN = 200


@dataclass
class RiskFactor:
    """A single risk factor from a 10-K filing."""
    rank: int
    title: str
    content: str
    embedding: list[float] | None = None
    sentiment_score: float | None = None
    id: str | None = None  # Optional DB id for pipeline mapping


def _snippet(text: str, max_len: int = SNIPPET_MAX_LEN) -> str:
    """First max_len chars of text, stripped."""
    if not text:
        return ""
    s = text.strip()
    return s[:max_len] + ("..." if len(s) > max_len else "")


@dataclass
class RemovedRisk:
    """A risk that appeared in the previous period but not the current (full_analysis.removed_risks)."""
    risk: str
    rank_prev: int
    status: str = "Removed"
    snippet: str = ""


@dataclass
class DriftResult:
    """Result of comparing a risk between two years (current-period risks only; removed are in RemovedRisk)."""
    risk_title: str
    drift_type: str  # 'new', 'rank_change', 'stable'
    rank_current: int
    rank_previous: int | None
    rank_delta: int
    semantic_score: float  # Cosine similarity 0-1
    heat_score: float = 0.0  # 0-100 per full_analysis.json
    zone: str = "stable_gray"
    analysis: str = ""
    # full_analysis.json / Phase 9 fields
    modality_shift: str = "none"
    strategic_recommendation: str = ""
    original_text_snippet: str | None = None
    new_text_snippet: str | None = None
    confidence_score: float = 0.0
    risk_factor_id: str | None = None  # Optional for DB mapping
    prev_factor_id: str | None = None


@dataclass
class DriftAnalysisResult:
    """Result of analyze_drift: drifts (current risks only) and removed_risks separately."""
    drifts: list[DriftResult] = field(default_factory=list)
    removed_risks: list[RemovedRisk] = field(default_factory=list)


class DriftCalculator:
    """
    Calculates risk drift between two filing years.

    Algorithm:
    1. Match risks by title (fuzzy) and content (semantic)
    2. Calculate rank changes for matched risks; assign heat_score and zone
    3. Identify new risks (in current) and removed risks (previous only)
    4. Return DriftAnalysisResult with drifts (current risks) and removed_risks separately
    """

    FUZZY_THRESHOLD = 0.75  # Minimum title similarity for match
    SEMANTIC_THRESHOLD = 0.80  # Minimum embedding similarity

    def __init__(
        self,
        heat_scorer: HeatScorer | None = None,
        zone_classifier: ZoneClassifier | None = None,
    ):
        self._heat_scorer = heat_scorer or HeatScorer()
        self._zone_classifier = zone_classifier or ZoneClassifier()

    def analyze_drift(
        self,
        current_risks: list[RiskFactor],
        previous_risks: list[RiskFactor]
    ) -> DriftAnalysisResult:
        """
        Compare risk factors between two years.

        Args:
            current_risks: Risks from current year (newer)
            previous_risks: Risks from previous year (older)

        Returns:
            DriftAnalysisResult with drifts (current risks only) and removed_risks
        """
        drifts: list[DriftResult] = []
        removed_risks: list[RemovedRisk] = []
        matched_previous = set()

        # Match current risks to previous
        for curr in current_risks:
            best_match, match_score = self._find_best_match(curr, previous_risks)

            if best_match and best_match.rank not in matched_previous:
                matched_previous.add(best_match.rank)
                rank_delta = best_match.rank - curr.rank  # Positive = moved up
                drift_type = "rank_change" if abs(rank_delta) >= 3 else "stable"

                heat_01 = self._heat_scorer.compute(
                    rank_delta=rank_delta,
                    semantic_score=match_score,
                    is_new=False,
                )
                zone = self._zone_classifier.classify(drift_type, rank_delta)
                analysis = self._analysis_text(drift_type, rank_delta, best_match.rank, curr.rank)

                drifts.append(DriftResult(
                    risk_title=curr.title,
                    drift_type=drift_type,
                    rank_current=curr.rank,
                    rank_previous=best_match.rank,
                    rank_delta=rank_delta,
                    semantic_score=match_score,
                    heat_score=round(heat_01 * 100.0, 1),  # 0-100 per full_analysis.json
                    zone=zone,
                    analysis=analysis,
                    original_text_snippet=_snippet(best_match.content) or None,
                    new_text_snippet=_snippet(curr.content) or None,
                    confidence_score=round(match_score, 2),
                    risk_factor_id=curr.id,
                    prev_factor_id=best_match.id,
                ))
            else:
                # New risk (no match found)
                heat_01 = self._heat_scorer.compute(
                    rank_delta=0,
                    semantic_score=0.0,
                    is_new=True,
                )
                zone = self._zone_classifier.classify("new", 0)
                analysis = "New risk factor identified."

                drifts.append(DriftResult(
                    risk_title=curr.title,
                    drift_type="new",
                    rank_current=curr.rank,
                    rank_previous=None,
                    rank_delta=0,
                    semantic_score=0.0,
                    heat_score=round(heat_01 * 100.0, 1),
                    zone=zone,
                    analysis=analysis,
                    new_text_snippet=_snippet(curr.content) or None,
                    confidence_score=0.0,
                    risk_factor_id=curr.id,
                ))

        # Removed risks: previous risks not matched
        for prev in previous_risks:
            if prev.rank not in matched_previous:
                removed_risks.append(RemovedRisk(
                    risk=prev.title,
                    rank_prev=prev.rank,
                    status="Removed",
                    snippet=_snippet(prev.content),
                ))

        return DriftAnalysisResult(drifts=drifts, removed_risks=removed_risks)

    def _analysis_text(self, drift_type: str, rank_delta: int, prev_rank: int, curr_rank: int) -> str:
        """Short analysis text for the drift."""
        if drift_type == "stable":
            return "Risk position updated."
        if rank_delta > 0:
            return f"Moved up from rank {prev_rank} to {curr_rank}."
        if rank_delta < 0:
            return f"Moved down from rank {prev_rank} to {curr_rank}."
        return "Risk position updated."

    def _find_best_match(
        self,
        current: RiskFactor,
        previous: list[RiskFactor]
    ) -> tuple[RiskFactor | None, float]:
        """Find best matching risk from previous year."""
        best = None
        best_score = 0.0

        for prev in previous:
            # Combine fuzzy title and semantic similarity
            title_score = self._fuzzy_match(current.title, prev.title)

            # Prioritize title match first
            if title_score > self.FUZZY_THRESHOLD:
                if title_score > best_score:
                    best = prev
                    best_score = title_score
            # If title match is weak, check embeddings if available
            elif current.embedding and prev.embedding:
                semantic = self._cosine_similarity(current.embedding, prev.embedding)
                if semantic > self.SEMANTIC_THRESHOLD and semantic > best_score:
                    best = prev
                    best_score = semantic

        return best, best_score

    def _fuzzy_match(self, title_a: str, title_b: str) -> float:
        """Fuzzy match two titles using token set ratio."""
        a_normalized = self._normalize_title(title_a)
        b_normalized = self._normalize_title(title_b)
        return fuzz.token_set_ratio(a_normalized, b_normalized) / 100.0

    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        # Remove common prefixes
        title = re.sub(r'^(risks?\s+related\s+to|our)\s+', '', title, flags=re.I)
        # Remove punctuation and lowercase
        title = re.sub(r'[^\w\s]', '', title.lower())
        return title.strip()

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))
