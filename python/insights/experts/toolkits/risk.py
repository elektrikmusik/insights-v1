"""
Risk Toolkit - Agno tools for risk analysis domain services.
"""
import logging
from typing import List, Dict, Any, Optional, Union

from agno.tools import Toolkit
from insights.core.events import publish_tool_used
from insights.services.risk.drift_calculator import (
    DriftAnalysisResult,
    DriftCalculator,
    DriftResult,
    RiskFactor,
)
from insights.services.risk.heat_scorer import HeatScorer

logger = logging.getLogger(__name__)


class RiskToolkit(Toolkit):
    """
    Toolkit for risk analysis domain services.
    Provides tools for calculating risk drift and heat scores.
    """

    def __init__(self):
        super().__init__(name="risk_toolkit")
        self.register(self.analyze_risk_drift)
        self.register(self.calculate_heat_score)

    def analyze_risk_drift(
        self, 
        current_risks: List[Union[Dict[str, Any], str]], 
        previous_risks: List[Union[Dict[str, Any], str]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze the drift (changes) between risk factors of two different filing periods.
        
        Args:
            current_risks: List of dicts (rank, title, content) or plain strings (treated as content).
            previous_risks: List of dicts (rank, title, content) or plain strings (treated as content).
            
        Returns:
            List of drift results containing rank_delta, semantic_score, and drift_type.
        """
        publish_tool_used("risk_toolkit", "analyze_risk_drift", message="Comparing risk factors across periods", current_count=len(current_risks), previous_count=len(previous_risks))
        try:
            calc = DriftCalculator()

            def _normalize(item: Any, index: int) -> Dict[str, Any]:
                """Accept dict with title/content or plain string; return dict with rank, title, content."""
                if isinstance(item, dict):
                    return {
                        "rank": item.get("rank", index + 1),
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "embedding": item.get("embedding"),
                        "sentiment_score": item.get("sentiment_score"),
                    }
                if isinstance(item, str):
                    title = item[:80] + "..." if len(item) > 80 else item
                    return {"rank": index + 1, "title": title, "content": item}
                return {"rank": index + 1, "title": "", "content": str(item)}

            def to_factor(d: Dict[str, Any]) -> RiskFactor:
                return RiskFactor(
                    rank=d.get("rank", 0),
                    title=d.get("title", ""),
                    content=d.get("content", ""),
                    embedding=d.get("embedding"),
                    sentiment_score=d.get("sentiment_score")
                )

            curr = [to_factor(_normalize(r, i)) for i, r in enumerate(current_risks)]
            prev = [to_factor(_normalize(r, i)) for i, r in enumerate(previous_risks)]

            result: DriftAnalysisResult = calc.analyze_drift(curr, prev)

            # Convert to dict for agent compatibility (drifts only; removed_risks in result.removed_risks)
            return [
                {
                    "risk_title": r.risk_title,
                    "drift_type": r.drift_type,
                    "rank_current": r.rank_current,
                    "rank_previous": r.rank_previous,
                    "rank_delta": r.rank_delta,
                    "semantic_score": r.semantic_score,
                    "heat_score": r.heat_score,
                    "zone": r.zone,
                    "analysis": r.analysis,
                    "modality_shift": r.modality_shift,
                    "strategic_recommendation": r.strategic_recommendation,
                    "original_text_snippet": r.original_text_snippet,
                    "new_text_snippet": r.new_text_snippet,
                    "confidence_score": r.confidence_score,
                }
                for r in result.drifts
            ]
        except Exception as e:
            logger.error(f"Error in analyze_risk_drift: {e}")
            return [{"error": str(e)}]

    def calculate_heat_score(
        self, 
        rank_delta: int, 
        semantic_score: float, 
        is_new: bool
    ) -> float:
        """
        Calculate a heat score (0.0 to 1.0) for a risk factor based on its change intensity.
        
        Args:
            rank_delta: Change in rank (positive = moved up).
            semantic_score: Content similarity score (0.0 to 1.0).
            is_new: Whether this is a new risk factor.
            
        Returns:
            Heat score (0.0 = cold/stable, 1.0 = hot/volatile).
        """
        publish_tool_used("risk_toolkit", "calculate_heat_score", message="Computing heat score", rank_delta=rank_delta, semantic_score=semantic_score, is_new=is_new)
        try:
            scorer = HeatScorer()
            return scorer.compute(
                rank_delta=rank_delta,
                semantic_score=semantic_score,
                is_new=is_new
            )
        except Exception as e:
            logger.error(f"Error in calculate_heat_score: {e}")
            return 0.0
