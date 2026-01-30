"""
Risk Expert implementation.
"""
from typing import Any, Dict
import logging

from insights.experts.base import BaseExpert
from insights.core.prompts import prompt_manager
from insights.adapters.mcp import SECToolkit
from insights.adapters.sentiment import SentimentToolkit
from insights.experts.toolkits.risk import RiskToolkit
from insights.core.types import ExpertResult

logger = logging.getLogger(__name__)


class RiskExpert(BaseExpert):
    """
    Expert specialized in analyzing Risk Factors from SEC filings.
    Uses SECToolkit to fetch data, RiskToolkit for domain logic, and SentimentToolkit for analysis.
    """

    def __init__(self):
        # Render system instructions from template
        try:
            instructions = prompt_manager.render("experts/risk_analyst.jinja2")
        except ValueError:
            logger.warning("Risk analyst template not found, using default instructions.")
            instructions = "You are a risk analyst specializing in SEC filings."

        super().__init__(
            name="Risk Analyst",
            description="Specialist in identifying, comparing, and analyzing financial and operational risks from SEC filings.",
            instructions=instructions,
            tools=[
                SECToolkit(),
                RiskToolkit(),
                SentimentToolkit()
            ]
        )

    async def analyze(self, query: str, context: Dict[str, Any]) -> ExpertResult:
        """
        Analyze risks based on user query and context.
        
        The context is prepended to the user query to provide background
        (e.g. Ticker, Year, previous analysis).
        """
        # Construct contextualized prompt
        # We might format context nicely here
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
        enhanced_query = f"""
Context Information:
{context_str}

Task:
{query}
"""
        logger.info(f"RiskExpert analyzing query: {query[:50]}...")
        
        # Use run_agent to ensure logging
        try:
            response = await self.run_agent(enhanced_query)
            content = response.content if response else "No content generated."
            
            # TODO: Improve source tracking (Agent can return sources if configured)
            # For now, we return the text findings.
            
            return ExpertResult(
                expert_id=self.expert_id,
                findings=str(content),
                confidence=0.85,  # Validated by expert checks in future
                sources=[], 
                metadata={
                    "model": self.agent.model.id,
                    "usage": response.metrics if hasattr(response, "metrics") else {}
                }
            )
        except Exception as e:
            logger.error(f"RiskExpert analysis failed: {e}")
            return ExpertResult(
                expert_id=self.expert_id,
                findings=f"Analysis failed: {str(e)}",
                confidence=0.0,
                sources=[],
                metadata={"error": str(e)}
            )
