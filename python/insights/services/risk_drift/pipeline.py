"""
Risk Drift Pipeline - DB-first approach for risk drift analysis.

This pipeline:
1. Gets/creates company record
2. Fetches/uses cached filings (DB-first)
3. Extracts/uses cached risk factors (DB-first)
4. Computes embeddings only where missing (required)
5. Runs drift calculation with heat/zone
6. Builds full_analysis.json structure
7. Saves risk_drifts to database
8. Generates markdown report

Reference: docs/specs/15_DB_POPULATION_RISK_DRIFT.md
"""
import json
import logging
from datetime import UTC, datetime, date
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from insights.adapters.db.manager import (
    CompanyCreate,
    CompanyRecord,
    DBManager,
    FilingCreate,
    RiskFactorCreate,
    RiskFactorRecord,
    RiskDriftCreate,
)
from insights.adapters.mcp.toolkit import SECToolkit
from insights.core.types import Company
from insights.services.filing.embedder import EmbeddingService
from insights.services.filing.parser import RiskFactorParser
from insights.services.report.generator import ReportGenerator
from insights.services.risk.drift_calculator import (
    DriftCalculator,
    DriftResult,
    RiskFactor,
    DriftAnalysisResult,
    RemovedRisk,
)

logger = logging.getLogger(__name__)


class RiskDriftPipelineResult(BaseModel):
    """Result from risk drift pipeline execution."""

    full_analysis: dict[str, Any]
    markdown_report: str
    company_id: UUID
    current_filing_id: UUID
    previous_filing_id: UUID


class RiskDriftPipeline:
    """
    DB-first risk drift analysis pipeline.
    
    Avoids re-fetching and re-computing by checking database first.
    """

    def __init__(
        self,
        db_manager: DBManager | None = None,
        sec_toolkit: SECToolkit | None = None,
        embedding_service: EmbeddingService | None = None,
        drift_calculator: DriftCalculator | None = None,
        report_generator: ReportGenerator | None = None,
    ):
        """Initialize pipeline with dependencies."""
        self.db = db_manager or DBManager()
        self.sec = sec_toolkit or SECToolkit()
        self.embedder = embedding_service or EmbeddingService()
        self.calculator = drift_calculator or DriftCalculator()
        self.reporter = report_generator or ReportGenerator()
        self.parser = RiskFactorParser()

    async def run(
        self, ticker: str, years: list[int]
    ) -> RiskDriftPipelineResult:
        """
        Execute complete risk drift pipeline.

        Args:
            ticker: Company ticker symbol
            years: List of years to compare [current, previous]

        Returns:
            RiskDriftPipelineResult with full_analysis and markdown
        """
        if len(years) < 2:
            raise ValueError("Need at least 2 years for comparison")

        current_year, previous_year = years[0], years[1]

        logger.info(f"Pipeline: Starting risk drift for {ticker} ({current_year} vs {previous_year})")

        # Step 1: Get/create company
        company = await self._get_or_create_company(ticker)
        logger.info(f"Pipeline: Company {company.ticker} (ID: {company.id})")

        # Step 2: Get/fetch filings (DB-first)
        current_filing = await self._get_or_fetch_filing(
            company.id, ticker, current_year
        )
        previous_filing = await self._get_or_fetch_filing(
            company.id, ticker, previous_year
        )
        logger.info(
            f"Pipeline: Filings - Current: {current_filing.id}, Previous: {previous_filing.id}"
        )

        # Step 3: Get/extract risk factors (DB-first)
        current_risks = await self._get_or_extract_risk_factors(current_filing)
        previous_risks = await self._get_or_extract_risk_factors(previous_filing)
        logger.info(
            f"Pipeline: Risk factors - Current: {len(current_risks)}, Previous: {len(previous_risks)}"
        )

        # Step 4: Ensure embeddings (required for semantic matching)
        await self._ensure_embeddings(current_risks)
        await self._ensure_embeddings(previous_risks)
        logger.info("Pipeline: Embeddings ensured")

        # Step 5: Run drift calculation
        current_risk_factors = [
            RiskFactor(
                id=str(r.id),
                rank=r.rank,
                title=r.title,
                content=r.content,
                embedding=r.embedding,
            )
            for r in current_risks
        ]
        previous_risk_factors = [
            RiskFactor(
                id=str(r.id),
                rank=r.rank,
                title=r.title,
                content=r.content,
                embedding=r.embedding,
            )
            for r in previous_risks
        ]

        drift_result = self.calculator.analyze_drift(
            current_risk_factors, previous_risk_factors
        )
        logger.info(
            f"Pipeline: Drift calculated - {len(drift_result.drifts)} drifts, {len(drift_result.removed_risks)} removed"
        )

        # Step 6: Build full_analysis structure
        full_analysis = self._build_full_analysis(
            drift_result=drift_result,
            current_risks=current_risks,
            previous_risks=previous_risks,
            filing_date=current_filing.filing_date,
        )

        # Step 7: Save risk_drifts to database
        await self._save_risk_drifts(company.id, drift_result.drifts)
        logger.info("Pipeline: Risk drifts saved to database")

        # Step 8: Generate markdown report
        company_obj = Company(
            ticker=company.ticker,
            name=company.name or company.ticker,
            sector=company.sector,
        )

        drift_dicts = [
            {
                "risk_title": d.risk_title,
                "heat_score": d.heat_score,
                "zone": d.zone,
                "rank_delta": d.rank_delta,
                "drift_type": d.drift_type,
            }
            for d in drift_result.drifts
        ]

        markdown_report = self.reporter.generate_markdown_report(
            company=company_obj,
            results=drift_dicts,
            expert_findings=[],
            summary=full_analysis.get("meta", {}).get("summary", "Risk drift analysis completed."),
        )

        return RiskDriftPipelineResult(
            full_analysis=full_analysis,
            markdown_report=markdown_report,
            company_id=company.id,
            current_filing_id=current_filing.id,
            previous_filing_id=previous_filing.id,
        )

    async def _get_or_create_company(self, ticker: str) -> CompanyRecord:
        """Get company from DB or create from SEC data."""
        # Check DB first
        company = await self.db.get_company_by_ticker(ticker)
        if company:
            logger.debug(f"Company {ticker} found in DB")
            return company

        # Fetch from SEC
        logger.info(f"Fetching company info for {ticker} from SEC")
        company_info_str = await self.sec.get_company_info(ticker)

        try:
            company_info = json.loads(company_info_str)
        except json.JSONDecodeError:
            # Fallback: create minimal record
            company_info = {
                "ticker": ticker,
                "cik": ticker,
                "name": ticker,
            }

        company_data = CompanyCreate(
            ticker=company_info.get("ticker", ticker).upper(),
            cik=company_info.get("cik", ticker),
            name=company_info.get("name", ticker),
            sector=company_info.get("sector"),
            exchange=company_info.get("exchange"),
        )

        company_id = await self.db.get_or_create_company(company_data)
        company = await self.db.get_company_by_id(company_id)
        if not company:
            raise ValueError(f"Failed to create company record for {ticker}")

        return company

    async def _get_or_fetch_filing(
        self, company_id: UUID, ticker: str, year: int
    ) -> Any:
        """Get filing from DB or fetch from SEC (DB-first)."""
        # Check DB first
        filings = await self.db.get_filings_by_company(
            company_id, form_type="10-K", years=[year]
        )

        if filings and filings[0].raw_text:
            logger.debug(f"Filing for {ticker} {year} found in DB")
            return filings[0]

        # Fetch from SEC
        logger.info(f"Fetching 10-K for {ticker} year {year} from SEC")

        # Get accession and filing metadata
        recent_filings_str = await self.sec.search_filings(
            ticker=ticker, form_types=["10-K"], days=365 * 3, limit=20
        )

        try:
            recent_data = json.loads(recent_filings_str)
            filings_list = (
                recent_data if isinstance(recent_data, list) else recent_data.get("filings", [])
            )
        except json.JSONDecodeError:
            filings_list = []

        # Find filing for the target year
        target_filing = None
        for f in filings_list:
            if isinstance(f, dict) and str(f.get("filing_date", "")).startswith(str(year)):
                target_filing = f
                break

        if not target_filing:
            raise ValueError(f"No 10-K filing found for {ticker} in {year}")

        accession = target_filing.get("accession_number")
        filing_date_str = target_filing.get("filing_date")

        # Fetch Item 1A content (MCP uses "risk_factors" key)
        item_1a_content = await self.sec.get_filing_section(
            ticker=ticker, section="risk_factors", form_type="10-K", year=year
        )

        # Save to DB
        filing_data = FilingCreate(
            company_id=company_id,
            accession_number=accession,
            form_type="10-K",
            filing_date=date.fromisoformat(filing_date_str),
            fiscal_year=year,
            fiscal_period="FY",
            report_url=target_filing.get("report_url"),
            raw_text=item_1a_content,
            metadata={"source": "sec_edgar", "section": "Item 1A"},
        )

        filing_id = await self.db.save_filing(filing_data)
        logger.debug(f"Saved filing with ID: {filing_id}, accession: {accession}")
        
        # Query by ID directly (more reliable than accession lookup)
        try:
            result = self.db._client.table("filings")\
                .select("*")\
                .eq("id", str(filing_id))\
                .limit(1)\
                .execute()
            
            if result.data:
                from insights.adapters.db.manager import FilingRecord
                filing_record = FilingRecord(**result.data[0])
                logger.debug(f"Retrieved filing record: {filing_record.id}")
                return filing_record
        except Exception as e:
            logger.warning(f"Failed to retrieve filing by ID: {e}")
        
        # Fallback: Try by accession
        filing_record = await self.db.get_filing_by_accession(accession)
        if filing_record:
            return filing_record
            
        raise ValueError(f"Failed to retrieve filing after save for {ticker} {year} (ID: {filing_id})")

    async def _get_or_extract_risk_factors(
        self, filing: Any
    ) -> list[RiskFactorRecord]:
        """Get risk factors from DB or extract from filing text (DB-first)."""
        # Check DB first
        risk_factors = await self.db.get_risk_factors_by_filing(filing.id)

        if risk_factors:
            logger.debug(f"Risk factors for filing {filing.id} found in DB ({len(risk_factors)} factors)")
            return risk_factors

        # Extract from raw text
        if not filing.raw_text:
            raise ValueError(f"No raw text available for filing {filing.id}")

        logger.info(f"Extracting risk factors for filing {filing.id}")
        parsed_risks = self.parser.extract_risks(filing.raw_text)

        # Save to DB
        risk_factor_creates = [
            RiskFactorCreate(
                filing_id=filing.id,
                title=risk.title,
                content=risk.content,
                rank=risk.rank,
                word_count=len(risk.content.split()) if risk.content else 0,
            )
            for risk in parsed_risks
        ]

        risk_ids = await self.db.save_risk_factors(risk_factor_creates)
        logger.info(f"Saved {len(risk_ids)} risk factors to DB")

        # Fetch back from DB
        risk_factors = await self.db.get_risk_factors_by_filing(filing.id)
        return risk_factors

    async def _ensure_embeddings(self, risk_factors: list[RiskFactorRecord]) -> None:
        """Ensure all risk factors have embeddings (compute only where missing)."""
        missing = [r for r in risk_factors if r.embedding is None]

        if not missing:
            logger.debug("All risk factors have embeddings")
            return

        logger.info(f"Computing embeddings for {len(missing)} risk factors")

        # Batch compute embeddings
        texts = [r.content for r in missing]
        embeddings = await self.embedder.embed_batch(texts)

        # Update DB
        for risk_factor, embedding in zip(missing, embeddings):
            await self.db.update_risk_factor_embedding(risk_factor.id, embedding)

        logger.info(f"Updated {len(missing)} embeddings in DB")

    async def _save_risk_drifts(
        self, company_id: UUID, drifts: list[DriftResult]
    ) -> None:
        """Save drift results to risk_drifts table."""
        drift_creates = []

        for drift in drifts:
            # Skip removed risks (they don't have risk_factor_id)
            if not drift.risk_factor_id:
                continue

            drift_creates.append(
                RiskDriftCreate(
                    company_id=company_id,
                    risk_factor_id=UUID(drift.risk_factor_id),
                    prev_factor_id=UUID(drift.prev_factor_id) if drift.prev_factor_id else None,
                    rank_current=drift.rank_current,
                    rank_prev=drift.rank_previous,
                    rank_delta=drift.rank_delta,
                    semantic_score=drift.semantic_score,
                    fuzzy_score=None,  # fuzzy_score not in DriftResult
                    drift_type=drift.drift_type,
                    zone=drift.zone,
                    heat_score=int(drift.heat_score),
                    modality_shift=drift.modality_shift,
                    analysis=drift.analysis,
                    strategic_recommendation=drift.strategic_recommendation,
                    original_text_snippet=drift.original_text_snippet,
                    new_text_snippet=drift.new_text_snippet,
                    confidence_score=drift.confidence_score,
                )
            )

        if drift_creates:
            await self.db.save_risk_drifts(drift_creates)
            logger.info(f"Saved {len(drift_creates)} risk drifts")

    def _build_full_analysis(
        self,
        drift_result: DriftAnalysisResult,
        current_risks: list[RiskFactorRecord],
        previous_risks: list[RiskFactorRecord],
        filing_date: date,
    ) -> dict[str, Any]:
        """
        Build full_analysis.json structure.

        Reference: docs/specs/full_analysis.json
        """
        # Build visual priority map
        visual_priority_map = []
        
        # Track which risks we've included
        included_risk_ids = set()

        # Add drifts
        for drift in drift_result.drifts:
            if drift.risk_factor_id:
                included_risk_ids.add(drift.risk_factor_id)
            
            visual_priority_map.append({
                "risk": drift.risk_title,
                "rank": drift.rank_current,
                "previous_rank": drift.rank_previous,
                "change": drift.rank_delta if drift.rank_previous else None,
                "status": self._get_status_label(drift.drift_type, drift.rank_delta),
            })

        # Sort by current rank
        visual_priority_map.sort(key=lambda x: x["rank"])

        # Build sentiment shift indicators (for drifts with changes)
        sentiment_shift_indicators = []
        for drift in drift_result.drifts:
            if drift.drift_type == "new" or (drift.rank_delta and abs(drift.rank_delta) > 0):
                sentiment_shift_indicators.append({
                    "risk": drift.risk_title,
                    "score": 0.0,  # Placeholder - would come from FinBERT
                    "confidence": drift.confidence_score,
                    "shift_detected": drift.drift_type != "stable",
                    "shift_type": "new" if drift.drift_type == "new" else "rank_change",
                    "analysis": drift.analysis or f"{drift.drift_type.replace('_', ' ').title()}.",
                    "strategic_recommendation": drift.strategic_recommendation or "Monitor closely.",
                    "original_snippet": drift.original_text_snippet,
                    "new_snippet": drift.new_text_snippet,
                })

        # Build materiality flags (for critical/warning zones)
        materiality_flags = []
        for drift in drift_result.drifts:
            intensity = "HIGH" if drift.zone == "critical_red" else "MEDIUM" if drift.zone == "warning_orange" else "LOW"
            
            if drift.zone in ("critical_red", "warning_orange", "new_blue"):
                materiality_flags.append({
                    "risk": drift.risk_title,
                    "alert": "Material Drift Detected",
                    "details": drift.analysis or f"{drift.drift_type.replace('_', ' ').title()}.",
                    "recommendation": drift.strategic_recommendation or "Assess impact.",
                    "intensity": intensity,
                })

        # Build removed risks list
        removed_risks = [
            {
                "risk": r.risk,
                "rank_prev": r.rank_prev,
                "status": r.status,
                "snippet": r.snippet,
            }
            for r in drift_result.removed_risks
        ]

        # Build heatmap zones
        zones_map: dict[str, list[dict[str, Any]]] = {
            "critical_red": [],
            "warning_orange": [],
            "new_blue": [],
            "stable_gray": [],
        }

        for drift in drift_result.drifts:
            risk_id = self._generate_risk_id(drift.risk_title)
            
            zones_map[drift.zone].append({
                "risk_id": risk_id,
                "title": drift.risk_title,
                "current_rank": drift.rank_current,
                "rank_delta": "NEW" if drift.drift_type == "new" else drift.rank_delta,
                "modality_shift": drift.modality_shift,
                "heat_score": int(drift.heat_score),
                "summary": drift.analysis or "Risk position updated.",
            })

        # Assemble full_analysis
        full_analysis = {
            "meta": {
                "total_risks_current": len(current_risks),
                "total_risks_prev": len(previous_risks),
                "drift_count": len([d for d in drift_result.drifts if d.drift_type != "stable"]),
                "filing_date": filing_date.isoformat(),
            },
            "visual_priority_map": visual_priority_map,
            "sentiment_shift_indicators": sentiment_shift_indicators,
            "materiality_flags": materiality_flags,
            "removed_risks": removed_risks,
            "heatmap": {
                "heatmap_title": "Risk Profile Evolution",
                "generated_at": datetime.now(UTC).isoformat(),
                "zones": zones_map,
            },
        }

        return full_analysis

    def _get_status_label(self, drift_type: str, rank_delta: int | None) -> str:
        """Get human-readable status label."""
        if drift_type == "new":
            return "New"
        if drift_type == "stable":
            return "Stable"
        if rank_delta is None:
            return "Unknown"
        if rank_delta > 0:
            return "Climbed"
        if rank_delta < 0:
            return "Fell"
        return "Stable"

    def _generate_risk_id(self, title: str) -> str:
        """Generate a stable risk ID from title."""
        # Simple slug generation
        import re
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower())
        return slug[:50]  # Limit length
