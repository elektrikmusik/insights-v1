"""
Orchestrator Agent - Coordinates the Team of Experts.
"""
import asyncio
import logging
import json
from datetime import datetime, UTC
from pathlib import Path

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from insights.core.types import SynthesizedReport, Company, ExpertResult
from insights.experts.registry import registry
from insights.adapters.mcp import SECToolkit
from insights.core.config import settings
from insights.core.prompts import prompt_manager

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrates the analysis workflow:
    1. Gather Context (Company Info)
    2. Dispatch to Experts
    3. Synthesize Results using a Lead Strategist Agent
    """
    
    def __init__(self, config_path: str | Path | None = None):
        self.sec_toolkit = SECToolkit()
        
        # Determine config path if not provided
        if config_path is None:
            # Assume we are in python/insights/agents/orchestrator/
            project_root = Path(__file__).parents[4]
            config_path = project_root / "configs" / "experts.yaml"
        
        self.config_path = Path(config_path)
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure experts are loaded from configuration."""
        if not self._initialized:
            if not registry.list_experts():
                logger.info(f"Loading experts from {self.config_path}")
                registry.load_from_yaml(self.config_path)
            
            # Fallback for MVP if loading fails or file empty
            if not registry.list_experts():
                logger.warning("No experts loaded from config. Registering fallback RiskExpert.")
                from insights.experts.risk import RiskExpert
                registry.register(RiskExpert())
            
            self._initialized = True

    async def process_request(
        self, 
        ticker: str, 
        years: list[int],
        options: dict | None = None
    ) -> SynthesizedReport:
        """
        Run the full analysis pipeline.
        """
        await self._ensure_initialized()
        logger.info(f"Starting orchestration for {ticker} over {years}")
        
        # 1. Get Company Info
        try:
            company_info_str = await self.sec_toolkit.get_company_info(ticker)
            name = ticker # Fallback
            cik = None
            try:
                data = json.loads(company_info_str)
                if isinstance(data, dict):
                    name = data.get("name", ticker)
                    cik = data.get("cik")
            except (json.JSONDecodeError, AttributeError):
                pass
        except Exception as e:
            logger.warning(f"Failed to fetch company info: {e}")
            company_info_str = "Not available"
            name = ticker
            cik = None
            
        company = Company(
            ticker=ticker, 
            name=name, 
            cik=cik,
            id=ticker 
        ) 
        
        # 2. Select Experts
        experts = registry.list_experts()
        
        # 3. Parallel Execution
        # Build query with explicit two-filing protocol when comparing years
        if len(years) >= 2:
            query = (
                f"Compare Item 1A Risk Factors for {ticker} between fiscal years {years[0]} and {years[1]}. "
                f"You MUST fetch and use TWO DISTINCT filings (one per year). "
                f"Do not compare a filing to itself - verify that years/accession numbers differ."
            )
        else:
            query = f"Analyze risks and material changes for {ticker} for fiscal year {years[0] if years else 'latest'}."
        
        context = {
            "ticker": ticker, 
            "years": years,
            "company_name": company.name,
            "company_info": company_info_str,
            "options": options or {},
            "two_period_comparison": len(years) >= 2,
            "required_years": years if len(years) >= 2 else None
        }
        
        logger.info(f"Dispatching to {len(experts)} experts: {[e.name for e in experts]}")
        
        # Run experts in parallel
        tasks = [expert.analyze(query, context) for expert in experts]
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results: list[ExpertResult] = []
        for r in results_raw:
            if isinstance(r, ExpertResult):
                valid_results.append(r)
            else:
                logger.error(f"Expert execution failed: {r}")
                
        # 4. Synthesis
        summary = await self._synthesize(valid_results, company, ticker, years)
        
        # 5. Return Report
        report = SynthesizedReport(
            company=company,
            summary=summary,
            expert_findings=valid_results,
            risk_level="high", # TODO: Aggregate from findings metadata
            recommendations=[] # Extracted during synthesis in future
        )
        return report

    async def _synthesize(
        self, 
        findings: list[ExpertResult], 
        company: Company, 
        ticker: str, 
        years: list[int]
    ) -> str:
        """
        Synthesize findings using an LLM-based Summarizer Agent.
        """
        if not findings:
            return "No expert findings were generated for this analysis."

        # Setup lead strategist agent
        orchestrator_agent = Agent(
            model=OpenAIChat(
                id=settings.DEFAULT_MODEL,
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL
            ),
            markdown=True,
        )

        try:
            # Render the synthesis prompt (PromptManager.render takes template_name and context dict)
            system_prompt = prompt_manager.render(
                "experts/lead_strategist.jinja2",
                context={
                    "company_name": company.name,
                    "ticker": ticker,
                    "years": years,
                    "expert_findings": findings,
                },
            )
            
            # Execute synthesis
            # We pass the full rendered template as instructions
            orchestrator_agent.instructions = system_prompt
            
            response = await orchestrator_agent.arun("Synthesize the provided expert reports into a final executive summary.")
            return str(response.content) if response else "Synthesis failed to produce content."
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            return self._fallback_synthesis(findings, company)

    def _fallback_synthesis(self, results: list[ExpertResult], company: Company) -> str:
        """Simple string concatenation fallback."""
        output = [f"## Executive Summary for {company.name} ({company.ticker})\n"]
        for res in results:
            output.append(f"### Findings from {res.expert_id}")
            output.append(res.findings)
            output.append(f"*Confidence: {res.confidence:.2f}*\n")
        return "\n".join(output)
