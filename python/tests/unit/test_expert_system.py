import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from insights.core.types import ExpertResult
from insights.experts.base import BaseExpert
from insights.experts.risk import RiskExpert
from insights.experts.registry import registry

@pytest.fixture
def mock_prompt_manager():
    with patch("insights.experts.risk.prompt_manager") as pm:
        pm.render.return_value = "System Instructions"
        yield pm

@pytest.mark.asyncio
async def test_risk_expert_initialization(mock_prompt_manager):
    # Mock Agno Agent to avoid model calls and tool instantiation side effects
    with patch("insights.experts.base.Agent") as MockAgent:
        # Also mock Toolkits to avoid their initialization side effects (e.g. MCP connection check)
        # But Toolkit init is usually fast. MCPToolkit invokes get_mcp_client -> singletons.
        # We might need to mock get_mcp_client.
        with patch("insights.adapters.mcp.toolkit.get_mcp_client"):
             with patch("insights.adapters.sentiment.get_sentiment_analyzer"):
                expert = RiskExpert()
                assert expert.name == "Risk Analyst"
                
                # Check that Agent was instantiated with tools
                call_args = MockAgent.call_args
                assert call_args is not None
                _, kwargs = call_args
                assert "tools" in kwargs
                # We passed 3 toolkits: SECToolkit, RiskToolkit, SentimentToolkit
                assert len(kwargs["tools"]) == 3

@pytest.mark.asyncio
async def test_risk_expert_analyze(mock_prompt_manager):
    with patch("insights.experts.base.Agent") as MockAgent:
        # Setup mock async run
        mock_agent_instance = MockAgent.return_value
        mock_agent_instance.arun = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Analysis Findings"
        mock_agent_instance.arun.return_value = mock_response
        
        with patch("insights.adapters.mcp.toolkit.get_mcp_client"), \
             patch("insights.adapters.sentiment.get_sentiment_analyzer"):
            
            expert = RiskExpert()
            result = await expert.analyze("Check AAPL", {"ticker": "AAPL"})
            
            assert isinstance(result, ExpertResult)
            assert result.findings == "Analysis Findings"
            assert result.expert_id == "risk_analyst"

def test_registry():
    registry.clear()
    
    # Create dummy expert
    class TestExpert(BaseExpert):
        def __init__(self):
            # Patch Agent to avoid side effects in BaseExpert init
            with patch("insights.experts.base.Agent"):
                super().__init__("Test Expert", "Desc", "Instr")

        async def analyze(self, q, c): 
            return None
            
    expert = TestExpert()
    
    registry.register(expert)
    assert registry.get_expert("test_expert") == expert
    assert len(registry.list_experts()) == 1
