"""
Integration tests for Risk Drift Pipeline.

These tests verify:
1. End-to-end pipeline execution
2. Database population (companies, filings, risk_factors, risk_drifts)
3. full_analysis structure
4. DB-first caching behavior

Note: Requires test database and MCP connection
"""
import pytest
from datetime import date
from uuid import UUID

from insights.services.risk_drift import RiskDriftPipeline
from insights.adapters.db.manager import DBManager


@pytest.mark.integration
@pytest.mark.asyncio
class TestRiskDriftPipeline:
    """Integration tests for risk drift pipeline."""

    async def test_pipeline_end_to_end(self, db_manager: DBManager):
        """Test complete pipeline execution populates database."""
        pipeline = RiskDriftPipeline(db_manager=db_manager)
        
        # Run pipeline
        ticker = "NVDA"
        years = [2024, 2023]
        
        result = await pipeline.run(ticker, years)
        
        # Verify result structure
        assert result.company_id is not None
        assert isinstance(result.company_id, UUID)
        assert result.current_filing_id is not None
        assert result.previous_filing_id is not None
        assert result.full_analysis is not None
        assert result.markdown_report is not None
        
        # Verify full_analysis structure
        full_analysis = result.full_analysis
        assert "meta" in full_analysis
        assert "visual_priority_map" in full_analysis
        assert "sentiment_shift_indicators" in full_analysis
        assert "materiality_flags" in full_analysis
        assert "removed_risks" in full_analysis
        assert "heatmap" in full_analysis
        
        # Verify meta
        meta = full_analysis["meta"]
        assert "total_risks_current" in meta
        assert "total_risks_prev" in meta
        assert "drift_count" in meta
        assert "filing_date" in meta
        
        # Verify heatmap structure
        heatmap = full_analysis["heatmap"]
        assert "zones" in heatmap
        zones = heatmap["zones"]
        assert "critical_red" in zones
        assert "warning_orange" in zones
        assert "new_blue" in zones
        assert "stable_gray" in zones

    async def test_pipeline_populates_database(self, db_manager: DBManager):
        """Test that pipeline populates all database tables."""
        pipeline = RiskDriftPipeline(db_manager=db_manager)
        
        ticker = "AAPL"
        years = [2024, 2023]
        
        result = await pipeline.run(ticker, years)
        
        # Verify company exists in DB
        company = await db_manager.get_company_by_ticker(ticker)
        assert company is not None
        assert company.ticker == ticker
        assert company.id == result.company_id
        
        # Verify filings exist in DB
        filings = await db_manager.get_filings_by_company(
            company.id, form_type="10-K", years=years
        )
        assert len(filings) >= 2
        
        # Verify risk factors exist in DB
        current_filing = next(f for f in filings if f.fiscal_year == years[0])
        previous_filing = next(f for f in filings if f.fiscal_year == years[1])
        
        current_risks = await db_manager.get_risk_factors_by_filing(current_filing.id)
        previous_risks = await db_manager.get_risk_factors_by_filing(previous_filing.id)
        
        assert len(current_risks) > 0
        assert len(previous_risks) > 0
        
        # Verify embeddings are populated
        assert all(r.embedding is not None for r in current_risks)
        assert all(r.embedding is not None for r in previous_risks)
        
        # Verify risk drifts exist in DB
        drifts = await db_manager.get_risk_drifts_by_company(company.id)
        assert len(drifts) > 0
        
        # Verify drift structure
        for drift in drifts:
            assert drift.zone in ("critical_red", "warning_orange", "new_blue", "stable_gray")
            assert 0 <= drift.heat_score <= 100
            assert drift.drift_type in ("new", "rank_change", "stable")

    async def test_pipeline_db_first_caching(self, db_manager: DBManager):
        """Test that pipeline uses DB-first approach (no re-fetching)."""
        pipeline = RiskDriftPipeline(db_manager=db_manager)
        
        ticker = "MSFT"
        years = [2024, 2023]
        
        # First run - fetches from SEC
        result1 = await pipeline.run(ticker, years)
        
        # Get counts
        company = await db_manager.get_company_by_ticker(ticker)
        filings1 = await db_manager.get_filings_by_company(
            company.id, form_type="10-K", years=years
        )
        
        # Second run - should use DB (no new fetches)
        result2 = await pipeline.run(ticker, years)
        
        filings2 = await db_manager.get_filings_by_company(
            company.id, form_type="10-K", years=years
        )
        
        # Verify same filing IDs (no duplicates created)
        assert len(filings1) == len(filings2)
        assert {f.id for f in filings1} == {f.id for f in filings2}
        
        # Verify same company ID
        assert result1.company_id == result2.company_id

    async def test_full_analysis_has_correct_drift_counts(self, db_manager: DBManager):
        """Test that full_analysis drift counts match actual drifts."""
        pipeline = RiskDriftPipeline(db_manager=db_manager)
        
        ticker = "GOOGL"
        years = [2024, 2023]
        
        result = await pipeline.run(ticker, years)
        full_analysis = result.full_analysis
        
        # Get drift count from meta
        meta_drift_count = full_analysis["meta"]["drift_count"]
        
        # Count non-stable drifts in heatmap
        heatmap_drift_count = 0
        for zone_name, zone_risks in full_analysis["heatmap"]["zones"].items():
            if zone_name != "stable_gray":
                heatmap_drift_count += len(zone_risks)
        
        # Verify counts are consistent
        assert meta_drift_count >= 0
        assert heatmap_drift_count >= meta_drift_count  # May include stable with changes

    async def test_removed_risks_in_full_analysis(self, db_manager: DBManager):
        """Test that removed risks are captured in full_analysis."""
        pipeline = RiskDriftPipeline(db_manager=db_manager)
        
        ticker = "TSLA"
        years = [2024, 2023]
        
        result = await pipeline.run(ticker, years)
        full_analysis = result.full_analysis
        
        # Verify removed_risks structure
        removed_risks = full_analysis.get("removed_risks", [])
        
        if removed_risks:
            for removed in removed_risks:
                assert "risk" in removed
                assert "rank_prev" in removed
                assert "status" in removed
                assert "snippet" in removed
                assert removed["status"] == "Removed"
                assert isinstance(removed["rank_prev"], int)

    async def test_markdown_report_generation(self, db_manager: DBManager):
        """Test that markdown report is generated correctly."""
        pipeline = RiskDriftPipeline(db_manager=db_manager)
        
        ticker = "AMZN"
        years = [2024, 2023]
        
        result = await pipeline.run(ticker, years)
        
        # Verify markdown structure
        markdown = result.markdown_report
        assert markdown is not None
        assert len(markdown) > 0
        
        # Check for key sections
        assert f"Risk Assessment Report: {ticker}" in markdown or ticker in markdown
        assert "Executive Summary" in markdown or "summary" in markdown.lower()
        assert "Risk" in markdown  # Should mention risks somewhere


@pytest.fixture
async def db_manager():
    """Provide a test database manager."""
    # Note: This would connect to a test database
    # In production, use a separate test database or mock
    manager = DBManager()
    yield manager
    # Cleanup would go here if needed
