---
name: insights-developer
description: Expert patterns for building InSights-ai financial analysis platform. Use PROACTIVELY when implementing any InSights feature.
---

# InSights-ai Developer Skill

## When to Use

Activate this skill when:
- Implementing any InSights-ai feature
- Creating new Experts or Domain Services
- Working with MCP clients or LLM adapters
- Building API endpoints or background workers

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR LAYER                                              │
│  OrchestratorAgent → routes to Experts → synthesizes findings    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   RiskExpert          IPExpert           TechExpert
   (FinBERT+SEC)       (Gemini+Patents)   (GPT-4o+GitHub)
        │                   │                   │
└───────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  DOMAIN SERVICES (Pure Python, no LLM)                          │
│  DriftCalculator, HeatScorer, ZoneClassifier, TextChunker       │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  ADAPTERS (Infrastructure)                                       │
│  MCPClient, SupabaseDB, ModelFactory (OpenRouter/SiliconFlow)    │
└─────────────────────────────────────────────────────────────────┘
```

## Core Patterns

### 1. Thin Agent Pattern

```python
# ❌ WRONG: Business logic in agent
class Agent:
    async def analyze(self, filing):
        chunks = self._chunk_text(filing)  # Logic in agent
        embeddings = await self._embed(chunks)
        return self._calculate_drift(embeddings)

# ✅ CORRECT: Agent delegates to services
class Agent:
    async def analyze(self, filing):
        chunks = self.text_chunker.chunk(filing)  # Service
        embeddings = await self.embedder.embed(chunks)  # Service
        return self.drift_calculator.calculate(embeddings)  # Service
```

### 2. Expert Implementation

```python
from insights.experts.base import BaseExpert
from insights.core.types import ExpertResult

class RiskExpert(BaseExpert):
    """Risk analysis expert using FinBERT and SEC MCP."""
    
    async def equip_tools(self) -> list:
        return [
            self.sec_toolkit,  # MCP wrapper
            self.risk_toolkit, # Domain service wrapper
        ]
    
    def get_system_prompt(self) -> str:
        return self.prompt_manager.render(
            "experts/risk_analyst.jinja2",
            company=self.context.company
        )
    
    async def conduct_research(self, query: str) -> ExpertResult:
        # Expert uses tools to gather data, then reasons
        response = await self.agent.arun(query)
        return ExpertResult(
            expert_id="risk_analyst",
            findings=response.content,
            confidence=0.85,
            sources=response.tool_calls
        )
```

### 3. Domain Service Pattern

```python
# Pure Python, no async, no LLM calls
from dataclasses import dataclass
import numpy as np

@dataclass
class DriftResult:
    similarity: float
    is_material: bool
    zone: str

class DriftCalculator:
    """Calculate semantic drift between risk factors."""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
    
    def calculate(
        self,
        old_embedding: np.ndarray,
        new_embedding: np.ndarray
    ) -> DriftResult:
        similarity = self._cosine_similarity(old_embedding, new_embedding)
        return DriftResult(
            similarity=similarity,
            is_material=similarity < self.threshold,
            zone=self._classify_zone(similarity)
        )
```

### 4. Multi-Provider LLM Usage

```python
from insights.adapters.models.factory import get_chat_model

# Auto-routes based on model prefix
gpt4 = get_chat_model("openai/gpt-4o")  # → OpenRouter
deepseek = get_chat_model("deepseek-ai/DeepSeek-V3")  # → SiliconFlow
qwen = get_chat_model("Qwen/Qwen2.5-72B-Instruct")  # → SiliconFlow

# Explicit provider override
model = get_chat_model("openai/gpt-4o", provider="siliconflow")
```

### 5. MCP Tool Wrapper

```python
from agno.tools import Toolkit
from insights.adapters.mcp.client import MCPClient

class SECToolkit(Toolkit):
    def __init__(self, mcp_client: MCPClient):
        super().__init__(name="sec_tools")
        self.mcp = mcp_client
        self.register(self.get_filing)
        self.register(self.search_filings)
    
    async def get_filing(self, ticker: str, form_type: str) -> str:
        """Fetch SEC filing for a company."""
        return await self.mcp.call_tool(
            "get_filing_content",
            {"ticker": ticker, "form_type": form_type}
        )
```

## File Templates

### New Expert

```python
# python/insights/experts/{name}.py
from insights.experts.base import BaseExpert
from insights.core.types import ExpertResult

class {Name}Expert(BaseExpert):
    """Description of what this expert analyzes."""
    
    async def equip_tools(self) -> list:
        return [self.{toolkit}_toolkit]
    
    def get_system_prompt(self) -> str:
        return self.prompt_manager.render("experts/{name}.jinja2")
    
    def can_handle(self, query: str) -> bool:
        keywords = ["keyword1", "keyword2"]
        return any(kw in query.lower() for kw in keywords)
    
    async def conduct_research(self, query: str) -> ExpertResult:
        response = await self.agent.arun(query)
        return ExpertResult(
            expert_id="{name}",
            findings=response.content,
            confidence=0.8,
            sources=[]
        )
```

### New Domain Service

```python
# python/insights/services/{domain}/{name}.py
from dataclasses import dataclass
from typing import List

@dataclass
class {Name}Result:
    """Result from {name} calculation."""
    value: float
    metadata: dict

class {Name}Calculator:
    """Pure Python service for {description}."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def calculate(self, input_data: List[float]) -> {Name}Result:
        # Pure computation, no I/O
        result = sum(input_data) / len(input_data)
        return {Name}Result(value=result, metadata={})
```

### New API Endpoint

```python
# python/insights/server/api/{name}.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/{name}", tags=["{name}"])

class {Name}Request(BaseModel):
    field: str

class {Name}Response(BaseModel):
    result: str

@router.post("/", response_model={Name}Response)
async def create_{name}(request: {Name}Request):
    # Implementation
    return {Name}Response(result="done")
```

## Documentation References

| Topic | Spec File |
|-------|-----------|
| Architecture | `docs/specs/01_ARCHITECTURE_AND_STRUCTURE.md` |
| Database | `docs/specs/02_DATABASE_SCHEMA.md` |
| API | `docs/specs/03_API_SPECIFICATION.md` |
| MCP Bridge | `docs/specs/04_MODULE_MCP_BRIDGE.md` |
| FinBERT | `docs/specs/05_MODULE_FINBERT.md` |
| Agents | `docs/specs/06_AGENT_ORCHESTRATION.md` |
| LLM Providers | `docs/specs/07_MODEL_MANAGEMENT.md` |
| Deployment | `docs/specs/08_ENV_AND_DEPLOYMENT.md` |
| Testing | `docs/specs/09_TESTING_STRATEGY.md` |
| Prompts | `docs/specs/10_PROMPT_ENGINEERING.md` |
| Services | `docs/specs/11_DOMAIN_SERVICES.md` |
| Workers | `docs/specs/12_BACKGROUND_WORKERS.md` |
| Experts | `docs/specs/13_EXPERT_REGISTRY.md` |

## Testing Patterns

```python
# Mock LLM responses
@pytest.fixture
def mock_llm():
    with patch("insights.adapters.models.factory.get_chat_model") as mock:
        mock.return_value.generate.return_value = '{"key": "value"}'
        yield mock

# Mock MCP calls
@pytest.fixture
def mock_mcp():
    with patch("insights.adapters.mcp.client.MCPClient") as mock:
        mock.return_value.call_tool.return_value = "filing content"
        yield mock

# Test domain service (no mocks needed)
def test_drift_calculator():
    calc = DriftCalculator(threshold=0.85)
    result = calc.calculate(np.array([1,2,3]), np.array([1,2,4]))
    assert result.similarity > 0.9
```

## Common Mistakes to Avoid

1. **Don't put business logic in agents** - Use domain services
2. **Don't hardcode model IDs** - Use config files
3. **Don't skip structured outputs** - Always use Pydantic models
4. **Don't forget circuit breakers** - Record success/failure
5. **Don't block the event loop** - FinBERT runs as microservice in prod
