"""
Zone Classifier - Categorizes risks for heatmap visualization.

Zones:
- critical_red: Major escalation (rank delta >= 4)
- warning_orange: Moderate change (rank delta 2-3) or removed
- new_blue: Newly identified risks
- stable_gray: Minimal change
"""


class ZoneClassifier:
    """Classifies risks into visualization zones."""

    # Thresholds
    CRITICAL_THRESHOLD = 4
    WARNING_THRESHOLD = 2

    def classify(self, drift_type: str, rank_delta: int) -> str:
        """
        Classify a risk into a zone.

        Args:
            drift_type: 'new', 'removed', 'rank_change', 'stable'
            rank_delta: Position change (positive = moved up)

        Returns:
            Zone name: 'critical_red', 'warning_orange', 'new_blue', 'stable_gray'
        """
        if drift_type == "new":
            return "new_blue"

        if drift_type == "removed":
            return "warning_orange"

        # Rank change classification
        abs_delta = abs(rank_delta)

        if abs_delta >= self.CRITICAL_THRESHOLD:
            return "critical_red"
        elif abs_delta >= self.WARNING_THRESHOLD:
            return "warning_orange"
        else:
            return "stable_gray"
