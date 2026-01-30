"""
Base Expert class for the Team of Experts system.
Wraps Agno Agents with domain-specific logic.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional
import logging

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit

from insights.core.types import ExpertResult
from insights.core.config import settings

logger = logging.getLogger(__name__)


class BaseExpert(ABC):
    """
    Abstract base class for all Domain Experts.
    Each expert encapsulates an Agno Agent with specific tools and instructions.
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        tools: Optional[List[Toolkit]] = None,
        model_id: Optional[str] = None,
    ):
        """
        Initialize the Expert.

        Args:
            name: Human-readable name of the expert
            description: Description of capabilities (for Orchestrator)
            instructions: System prompt/instructions for the agent
            tools: List of Agno Toolkits to provide to the agent
            model_id: Specific model ID to use (defaults to global setting)
        """
        self.name = name
        self.description = description
        self.expert_id = name.lower().replace(" ", "_")
        
        # Use provided model or default from settings
        target_model_id = model_id or settings.DEFAULT_MODEL
        
        # Configure Agno model
        # We default to OpenRouter configuration as per project settings
        # If needed, we can expand logic here to switch providers based on model_id format
        model = OpenAIChat(
            id=target_model_id,
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
        )
        
        self.agent = Agent(
            name=name,
            description=description,
            instructions=instructions,
            tools=tools or [],
            model=model,
            markdown=True,
        )
        
    async def run_agent(self, query: str) -> Any:
        """
        Run the internal agent and log LLM usage.
        
        Args:
            query: The prompt/query for the agent
            
        Returns:
            Agno Agent RunResponse
        """
        from insights.adapters.db.manager import DBManager
        
        # Execute agent
        response = await self.agent.arun(query)
        
        # Log usage (Agno may expose metrics as dict or as Metrics object)
        try:
            from insights.adapters.db.manager import AuditLogCreate
            db = DBManager()
            usage = getattr(response, "metrics", None)
            if usage is not None:
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
                else:
                    prompt_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0)
                    completion_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0)
                if prompt_tokens or completion_tokens:
                    await db.log_llm_usage(
                        AuditLogCreate(
                            operation="llm_usage",
                            model_id=self.agent.model.id,
                            provider="openrouter",
                            input_tokens=int(prompt_tokens or 0),
                            output_tokens=int(completion_tokens or 0),
                            cost_usd=0.0,
                            latency_ms=0,
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to log LLM usage for {self.name}: {e}")
            
        return response

    @abstractmethod
    async def analyze(self, query: str, context: dict[str, Any]) -> ExpertResult:
        """
        Perform analysis based on query and context.
        
        Args:
            query: The specific question or task for the expert
            context: Shared context (e.g., company info, filings)
            
        Returns:
            ExpertResult containing findings and metadata
        """
        pass
