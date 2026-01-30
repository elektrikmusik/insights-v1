"""
Heat Score Calculator - Determines visual intensity for heatmap.

Formula:
  heat_score = w1 * rank_factor + w2 * change_factor + w3 * newness_factor
Where:
  - rank_factor: Higher current rank = less heat (1 is hottest)
  - change_factor: Larger rank delta = more heat
  - newness_factor: New risks get bonus heat
"""
from dataclasses import dataclass


@dataclass
class HeatWeights:
    """Weights for heat score calculation."""
    rank_weight: float = 0.3
    change_weight: float = 0.5
    new_bonus: float = 0.2


class HeatScorer:
    """Calculates heat scores for risk factors."""

    def __init__(self, weights: HeatWeights | None = None):
        self.weights = weights or HeatWeights()
        self.max_rank = 20  # Normalize rank to this max

    def compute(
        self,
        rank_delta: int,
        semantic_score: float,
        is_new: bool
    ) -> float:
        """
        Compute heat score for a risk factor.

        Args:
            rank_delta: Position change (positive = moved up)
            semantic_score: Similarity to previous (0-1)
            is_new: Whether this is a new risk

        Returns:
            Heat score 0.0 to 1.0
        """
        # Change factor: larger delta = more heat
        change_factor = min(abs(rank_delta) / 5.0, 1.0)

        # Semantic factor: lower similarity = more change = more heat
        change_factor += (1 - semantic_score) * 0.3

        # New risk bonus
        new_factor = self.weights.new_bonus if is_new else 0.0

        # Combine (clamped to 0-1)
        heat = (
            self.weights.change_weight * change_factor +
            new_factor
        )

        return min(max(heat, 0.0), 1.0)
