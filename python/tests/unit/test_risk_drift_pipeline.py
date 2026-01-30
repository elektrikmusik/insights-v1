"""
Unit tests for Risk Drift Pipeline components.

These tests verify individual pipeline methods without requiring
full database or MCP connections.
"""
import pytest
from datetime import date
from uuid import uuid4

from insights.services.risk_drift.pipeline import RiskDriftPipeline
from insights.services.risk.drift_calculator import (
    DriftResult,
    DriftAnalysisResult,
    RemovedRisk,
)
from insights.adapters.db.manager import RiskFactorRecord


class TestRiskDriftPipeline:
    """Unit tests for pipeline helper methods."""

    def test_build_full_analysis_structure(self):
        """Test _build_full_analysis creates correct structure."""
        pipeline = RiskDriftPipeline()
        
        # Mock drift result
        drift_result = DriftAnalysisResult(
            drifts=[
                DriftResult(
                    risk_title="Test Risk",
                    new_title="Test Risk",
                    rank_current=1,
                    rank_prev=2,
                    rank_delta=1,
                    drift_type="rank_change",
                    zone="warning_orange",
                    heat_score=75.0,
                    semantic_score=0.9,
                    fuzzy_score=90,
                    analysis="Risk moved up",
                    risk_factor_id=str(uuid4()),
                    prev_factor_id=str(uuid4()),
                )
            ],
            removed_risks=[
                RemovedRisk(
                    risk="Old Risk",
                    rank_prev=3,
                    status="Removed",
                    snippet="Old risk snippet",
                )
            ],
        )
        
        # Mock risk records
        current_risks = [
            RiskFactorRecord(
                id=uuid4(),
                filing_id=uuid4(),
                title="Test Risk",
                content="Test content",
                rank=1,
                embedding=[0.1] * 768,
            )
        ]
        
        previous_risks = [
            RiskFactorRecord(
                id=uuid4(),
                filing_id=uuid4(),
                title="Test Risk",
                content="Test content old",
                rank=2,
                embedding=[0.1] * 768,
            ),
            RiskFactorRecord(
                id=uuid4(),
                filing_id=uuid4(),
                title="Old Risk",
                content="Old content",
                rank=3,
                embedding=[0.1] * 768,
            ),
        ]
        
        # Build full_analysis
        full_analysis = pipeline._build_full_analysis(
            drift_result=drift_result,
            current_risks=current_risks,
            previous_risks=previous_risks,
            filing_date=date(2024, 2, 1),
        )
        
        # Verify structure
        assert "meta" in full_analysis
        assert "visual_priority_map" in full_analysis
        assert "sentiment_shift_indicators" in full_analysis
        assert "materiality_flags" in full_analysis
        assert "removed_risks" in full_analysis
        assert "heatmap" in full_analysis
        
        # Verify meta
        assert full_analysis["meta"]["total_risks_current"] == 1
        assert full_analysis["meta"]["total_risks_prev"] == 2
        assert full_analysis["meta"]["drift_count"] == 1
        assert full_analysis["meta"]["filing_date"] == "2024-02-01"
        
        # Verify visual_priority_map
        assert len(full_analysis["visual_priority_map"]) >= 1
        assert full_analysis["visual_priority_map"][0]["risk"] == "Test Risk"
        
        # Verify removed_risks
        assert len(full_analysis["removed_risks"]) == 1
        assert full_analysis["removed_risks"][0]["risk"] == "Old Risk"
        
        # Verify heatmap zones
        zones = full_analysis["heatmap"]["zones"]
        assert "warning_orange" in zones
        assert len(zones["warning_orange"]) == 1

    def test_get_status_label(self):
        """Test status label generation."""
        pipeline = RiskDriftPipeline()
        
        assert pipeline._get_status_label("new", None) == "New"
        assert pipeline._get_status_label("stable", 0) == "Stable"
        assert pipeline._get_status_label("rank_change", 3) == "Climbed"
        assert pipeline._get_status_label("rank_change", -2) == "Fell"

    def test_generate_risk_id(self):
        """Test risk ID generation."""
        pipeline = RiskDriftPipeline()
        
        risk_id = pipeline._generate_risk_id("Competition could impact market share")
        assert isinstance(risk_id, str)
        assert len(risk_id) <= 50
        assert " " not in risk_id
        assert risk_id == "competition_could_impact_market_share"

    def test_full_analysis_heatmap_zones(self):
        """Test that all zone types are represented in heatmap."""
        pipeline = RiskDriftPipeline()
        
        # Mock drifts across all zones
        drift_result = DriftAnalysisResult(
            drifts=[
                DriftResult(
                    risk_title="Critical",
                    new_title="Critical",
                    rank_current=1,
                    rank_prev=5,
                    rank_delta=4,
                    drift_type="rank_change",
                    zone="critical_red",
                    heat_score=95.0,
                    semantic_score=0.8,
                    fuzzy_score=80,
                    analysis="Critical change",
                    risk_factor_id=str(uuid4()),
                ),
                DriftResult(
                    risk_title="Warning",
                    new_title="Warning",
                    rank_current=2,
                    rank_prev=None,
                    rank_delta=None,
                    drift_type="new",
                    zone="new_blue",
                    heat_score=80.0,
                    semantic_score=0.0,
                    fuzzy_score=0,
                    analysis="New risk",
                    risk_factor_id=str(uuid4()),
                ),
                DriftResult(
                    risk_title="Stable",
                    new_title="Stable",
                    rank_current=3,
                    rank_prev=3,
                    rank_delta=0,
                    drift_type="stable",
                    zone="stable_gray",
                    heat_score=0.0,
                    semantic_score=1.0,
                    fuzzy_score=100,
                    analysis="No change",
                    risk_factor_id=str(uuid4()),
                ),
            ],
            removed_risks=[],
        )
        
        full_analysis = pipeline._build_full_analysis(
            drift_result=drift_result,
            current_risks=[],
            previous_risks=[],
            filing_date=date(2024, 1, 1),
        )
        
        zones = full_analysis["heatmap"]["zones"]
        
        # Verify all zones present
        assert "critical_red" in zones
        assert "warning_orange" in zones
        assert "new_blue" in zones
        assert "stable_gray" in zones
        
        # Verify correct distribution
        assert len(zones["critical_red"]) == 1
        assert len(zones["new_blue"]) == 1
        assert len(zones["stable_gray"]) == 1
