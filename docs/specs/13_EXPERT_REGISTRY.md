# 13. Expert Registry & Modular Wiring

## Overview

InSights-ai employs a **"Team of Experts"** architecture, where specialized analysis modules are orchestrated by a central coordinator. Each Expert encapsulates domain-specific knowledge, tools, and models.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Orchestrator Agent                                                          │
│  • Routes requests to appropriate Experts                                   │
│  • Synthesizes multi-expert findings                                        │
│  • Manages context and conversation                                         │
└──────────────────┬─────────────────┬─────────────────┬──────────────────────┘
                   │                 │                 │
         ┌─────────▼─────────┐ ┌─────▼─────┐ ┌─────────▼─────────┐
         │  Risk Expert      │ │  IP Expert │ │  Tech Expert      │
         │  ├─ FinBERT      │ │  ├─ Gemini  │ │  ├─ GPT-4o       │
         │  ├─ SEC MCP      │ │  ├─ Patents │ │  ├─ GitHub API   │
         │  └─ Drift Tools  │ │  └─ Legal   │ │  └─ StackShare   │
         └───────────────────┘ └────────────┘ └───────────────────┘
```

---

## Key Concepts

### What is an Expert?

An **Expert** is a self-contained analysis module specialized in one domain. It encapsulates:

| Component | Description |
|-----------|-------------|
| **Data Sources** | Tools and APIs (SEC MCP, Patents, GitHub) |
| **Cognitive Model** | LLM or specialized model (FinBERT, Gemini, GPT-4o) |
| **Instructions** | Domain-specific prompts and analysis frameworks |
| **Output Schema** | Pydantic models for structured results |

### Why Experts?

1. **Modularity**: Add new domains without changing core architecture
2. **Specialization**: Each expert uses the optimal model for its task
3. **Testability**: Experts can be tested in isolation
4. **Scalability**: Route to different experts in parallel
5. **Cost Efficiency**: Use cheaper models where appropriate

---

## Configuration

### Expert Registry (`configs/experts.yaml`)

```yaml
# Expert definitions with models, tools, and capabilities
experts:
  risk_analyst:
    class: "insights.experts.risk.RiskExpert"
    name: "Risk Analyst"
    model: "finbert"  # Specialized sentiment model
    fallback_model: "openai/gpt-4o-mini"
    tools:
      - "sec_toolkit"
      - "sentiment_toolkit"
      - "risk_toolkit"
    sources: ["10-K", "10-Q", "8-K"]
    description: "Analyzes financial risks, sentiment, and volatility."
    output_schema: "RiskAnalysisResult"
    priority: 1  # Higher = more likely to be selected

  ip_analyst:
    class: "insights.experts.ip.PatentExpert"
    name: "IP Analyst"
    model: "google/gemini-1.5-pro"  # Long context for technical docs
    tools:
      - "patent_toolkit"
      - "legal_toolkit"
    sources: ["USPTO", "EPO", "litigation databases"]
    description: "Evaluates intellectual property moats and patent portfolios."
    output_schema: "PatentAnalysisResult"
    priority: 2

  tech_analyst:
    class: "insights.experts.tech.TechExpert"
    name: "Technology Analyst"
    model: "openai/gpt-4o"
    tools:
      - "github_toolkit"
      - "stackshare_toolkit"
    sources: ["10-K Item 1", "Engineering blogs", "GitHub repos"]
    description: "Maps technology stacks, dependencies, and technical debt."
    output_schema: "TechAnalysisResult"
    priority: 2

  macro_analyst:
    class: "insights.experts.macro.MacroExpert"
    name: "Macro Analyst"
    model: "anthropic/claude-3.5-sonnet"
    tools:
      - "news_toolkit"
      - "econ_toolkit"
    sources: ["FRED", "news APIs", "central bank releases"]
    description: "Analyzes macroeconomic context and sector trends."
    output_schema: "MacroAnalysisResult"
    priority: 3

# Orchestrator configuration
orchestrator:
  model: "openai/gpt-4o"
  max_parallel_experts: 3
  synthesis_model: "anthropic/claude-3.5-sonnet"  # For final synthesis
  routing_strategy: "intent_based"  # or "all", "priority"
```

---

## Python Implementation

### Base Expert (`insights/experts/base.py`)

```python
"""
Base class for all domain experts.
"""
from abc import ABC, abstractmethod
from typing import List, Type, Optional
from pydantic import BaseModel
from agno.agent import Agent
from agno.tools import Toolkit

from insights.core.prompts import render_prompt
from insights.adapters.models.factory import get_chat_model


class ExpertResult(BaseModel):
    """Base result schema for all experts."""
    expert_name: str
    confidence: float  # 0.0 - 1.0
    summary: str
    details: dict
    sources_used: List[str]


class BaseExpert(ABC):
    """
    Base class for domain-specific experts.
    
    Each expert:
    - Has a specialized model
    - Carries specific tools
    - Uses domain-specific prompts
    - Returns structured results
    """
    
    name: str = "Base Expert"
    description: str = ""
    
    def __init__(self, config: dict):
        self.config = config
        self._model = None
        self._tools = None
    
    @property
    def model(self):
        """Lazy-load the cognitive model."""
        if self._model is None:
            model_id = self.config.get("model", "openai/gpt-4o")
            self._model = get_chat_model(model_id)
        return self._model
    
    @abstractmethod
    def equip_tools(self) -> List[Toolkit]:
        """
        Return Agno Toolkits this expert carries.
        
        Example:
            return [SECToolkit(), SentimentToolkit()]
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self, context: Optional[dict] = None) -> str:
        """
        Return the expert's persona and instructions.
        
        Args:
            context: Optional context (ticker, sector, etc.)
        """
        pass
    
    @abstractmethod
    def get_output_schema(self) -> Type[BaseModel]:
        """Return the Pydantic model for structured output."""
        pass
    
    async def conduct_research(
        self, 
        query: str,
        ticker: str,
        context: Optional[dict] = None
    ) -> ExpertResult:
        """
        Execute the expert's research workflow.
        
        Args:
            query: The research question
            ticker: Company ticker symbol
            context: Additional context from orchestrator
            
        Returns:
            Structured ExpertResult
        """
        # Build agent with expert's configuration
        agent = Agent(
            model=self.model,
            tools=self.equip_tools(),
            instructions=self.get_system_prompt(context),
            response_model=self.get_output_schema()
        )
        
        # Execute research
        prompt = f"Analyze {ticker}: {query}"
        response = await agent.arun(prompt)
        
        return ExpertResult(
            expert_name=self.name,
            confidence=self._calculate_confidence(response),
            summary=response.summary if hasattr(response, 'summary') else str(response),
            details=response.model_dump() if hasattr(response, 'model_dump') else {},
            sources_used=self._extract_sources(response)
        )
    
    def _calculate_confidence(self, response) -> float:
        """Calculate confidence based on response quality."""
        # Override in subclasses for domain-specific confidence
        return 0.8
    
    def _extract_sources(self, response) -> List[str]:
        """Extract sources cited in the response."""
        return self.config.get("sources", [])
    
    def can_handle(self, query: str) -> float:
        """
        Return a score 0-1 indicating how well this expert can handle the query.
        Used by orchestrator for routing.
        """
        # Override in subclasses for semantic matching
        return 0.5
```

### Risk Expert (`insights/experts/risk.py`)

```python
"""
Risk domain expert - specializes in SEC filings and sentiment.
"""
from typing import List, Type
from pydantic import BaseModel

from .base import BaseExpert, ExpertResult
from insights.agents.research.toolkits import SECToolkit, SentimentToolkit, RiskToolkit
from insights.core.prompts import render_prompt


class RiskAnalysisResult(BaseModel):
    """Structured output for risk analysis."""
    summary: str
    risk_factors: List[dict]
    sentiment_score: float
    drift_zones: dict
    recommendations: List[str]


class RiskExpert(BaseExpert):
    """Expert in financial risk analysis using SEC filings and FinBERT."""
    
    name = "Risk Analyst"
    description = "Analyzes financial risks, sentiment shifts, and volatility indicators."
    
    def equip_tools(self) -> list:
        return [
            SECToolkit(),
            SentimentToolkit(mode="finbert"),
            RiskToolkit()
        ]
    
    def get_system_prompt(self, context=None) -> str:
        return render_prompt("experts/risk_analyst", {
            "ticker": context.get("ticker") if context else None,
            "sector": context.get("sector") if context else None
        })
    
    def get_output_schema(self) -> Type[BaseModel]:
        return RiskAnalysisResult
    
    def can_handle(self, query: str) -> float:
        """Check if query is risk-related."""
        risk_keywords = ["risk", "volatility", "sentiment", "10-K", "SEC", "filing", "drift"]
        query_lower = query.lower()
        matches = sum(1 for kw in risk_keywords if kw in query_lower)
        return min(matches / 3, 1.0)
```

**Persona Alignment:** The Risk Analyst prompt ([configs/prompts/experts/risk_analyst.jinja2](../../configs/prompts/experts/risk_analyst.jinja2)) implements a **Forensic Accountant** persona aligned with [docs/analyst_agent.py](../../docs/analyst_agent.py). Key characteristics:
- **Skeptical stance**: Looks for what management is hiding, not promoting
- **Mandatory safety gates**: Grounded citations, boilerplate suppression, drift scoring (structural/semantic/both), "NOT FOUND" protocol
- **Core skills**: Structural drift (rank changes), semantic drift (meaning shifts), strategic implication ("So what?")
- **Two-filing protocol**: Must fetch TWO DISTINCT Item 1A sections and verify they differ before comparison

### Patent Expert (`insights/experts/ip.py`)

```python
"""
Intellectual Property expert - specializes in patent analysis.
"""
from typing import List, Type
from pydantic import BaseModel

from .base import BaseExpert
from insights.core.prompts import render_prompt


class PatentAnalysisResult(BaseModel):
    """Structured output for patent analysis."""
    summary: str
    patent_count: int
    key_patents: List[dict]
    expiry_timeline: List[dict]
    competitive_moat_score: float
    litigation_risks: List[str]


class PatentExpert(BaseExpert):
    """Expert in intellectual property and patent portfolio analysis."""
    
    name = "IP Analyst"
    description = "Evaluates patent portfolios, IP moats, and litigation risks."
    
    def equip_tools(self) -> list:
        from insights.adapters.tools.patents import PatentToolkit
        from insights.adapters.tools.legal import LegalToolkit
        return [PatentToolkit(), LegalToolkit()]
    
    def get_system_prompt(self, context=None) -> str:
        return render_prompt("experts/ip_analyst", context or {})
    
    def get_output_schema(self) -> Type[BaseModel]:
        return PatentAnalysisResult
    
    def can_handle(self, query: str) -> float:
        ip_keywords = ["patent", "ip", "intellectual property", "trademark", "moat", "litigation"]
        query_lower = query.lower()
        matches = sum(1 for kw in ip_keywords if kw in query_lower)
        return min(matches / 3, 1.0)
```

---

## Expert Registry

### Registry Class (`insights/experts/registry.py`)

```python
"""
Expert Registry - manages loading and routing to experts.
"""
import yaml
from typing import Dict, List, Optional, Type
from pathlib import Path
import importlib

from .base import BaseExpert, ExpertResult


class ExpertRegistry:
    """
    Central registry for all domain experts.
    
    Responsibilities:
    - Load expert configurations from YAML
    - Instantiate experts on demand (lazy loading)
    - Route queries to appropriate experts
    - Coordinate parallel expert execution
    """
    
    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path("configs/experts.yaml")
        self._config: Dict = {}
        self._experts: Dict[str, BaseExpert] = {}
        self._load_config()
    
    def _load_config(self):
        """Load expert configurations from YAML."""
        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)
    
    def _import_expert_class(self, class_path: str) -> Type[BaseExpert]:
        """Dynamically import an expert class."""
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    def get_expert(self, expert_id: str) -> BaseExpert:
        """
        Get an expert by ID, instantiating if needed.
        
        Args:
            expert_id: Expert identifier (e.g., "risk_analyst")
            
        Returns:
            Instantiated expert
        """
        if expert_id not in self._experts:
            expert_config = self._config["experts"].get(expert_id)
            if not expert_config:
                raise ValueError(f"Unknown expert: {expert_id}")
            
            expert_class = self._import_expert_class(expert_config["class"])
            self._experts[expert_id] = expert_class(expert_config)
        
        return self._experts[expert_id]
    
    def list_experts(self) -> List[Dict]:
        """List all registered experts with metadata."""
        return [
            {
                "id": expert_id,
                "name": config.get("name", expert_id),
                "description": config.get("description", ""),
                "model": config.get("model"),
                "sources": config.get("sources", [])
            }
            for expert_id, config in self._config["experts"].items()
        ]
    
    def route_query(self, query: str) -> List[str]:
        """
        Determine which experts should handle a query.
        
        Args:
            query: User's research question
            
        Returns:
            List of expert IDs to activate
        """
        scores = []
        for expert_id in self._config["experts"]:
            expert = self.get_expert(expert_id)
            score = expert.can_handle(query)
            priority = self._config["experts"][expert_id].get("priority", 5)
            scores.append((expert_id, score, priority))
        
        # Sort by score (desc) then priority (asc)
        scores.sort(key=lambda x: (-x[1], x[2]))
        
        # Return experts with score > threshold
        threshold = 0.3
        return [eid for eid, score, _ in scores if score >= threshold]
    
    async def consult_experts(
        self, 
        query: str, 
        ticker: str,
        expert_ids: Optional[List[str]] = None,
        parallel: bool = True
    ) -> List[ExpertResult]:
        """
        Consult one or more experts for research.
        
        Args:
            query: Research question
            ticker: Company ticker
            expert_ids: Specific experts to use (or auto-route)
            parallel: Execute experts in parallel
            
        Returns:
            List of ExpertResults
        """
        import asyncio
        
        # Auto-route if no experts specified
        if not expert_ids:
            expert_ids = self.route_query(query)
        
        # Limit parallel execution
        max_parallel = self._config.get("orchestrator", {}).get("max_parallel_experts", 3)
        expert_ids = expert_ids[:max_parallel]
        
        if parallel:
            tasks = [
                self.get_expert(eid).conduct_research(query, ticker)
                for eid in expert_ids
            ]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for eid in expert_ids:
                result = await self.get_expert(eid).conduct_research(query, ticker)
                results.append(result)
            return results


# Singleton
_registry: Optional[ExpertRegistry] = None


def get_expert_registry() -> ExpertRegistry:
    global _registry
    if _registry is None:
        _registry = ExpertRegistry()
    return _registry
```

---

## Orchestrator Agent

The Orchestrator coordinates experts and synthesizes findings.

### `insights/agents/orchestrator/agent.py`

```python
"""
Orchestrator Agent - routes to experts and synthesizes findings.
"""
from typing import List, Optional
from pydantic import BaseModel
from agno.agent import Agent

from insights.experts.registry import get_expert_registry, ExpertResult
from insights.adapters.models.factory import get_chat_model
from insights.core.prompts import render_prompt


class SynthesizedReport(BaseModel):
    """Final synthesized report from multiple experts."""
    executive_summary: str
    expert_findings: List[dict]
    integrated_analysis: str
    recommendations: List[str]
    confidence_score: float


class OrchestratorAgent:
    """
    Orchestrates a team of domain experts.
    
    Workflow:
    1. Analyze query intent
    2. Route to appropriate experts
    3. Execute expert research (parallel)
    4. Synthesize findings into unified report
    """
    
    def __init__(self):
        self.registry = get_expert_registry()
        self.synthesis_model = get_chat_model("anthropic/claude-3.5-sonnet")
    
    async def research(
        self, 
        query: str, 
        ticker: str,
        experts: Optional[List[str]] = None
    ) -> SynthesizedReport:
        """
        Execute full research workflow with expert team.
        
        Args:
            query: User's research question
            ticker: Company ticker symbol
            experts: Optional list of expert IDs to use
            
        Returns:
            SynthesizedReport with integrated findings
        """
        # 1. Consult experts
        expert_results = await self.registry.consult_experts(
            query=query,
            ticker=ticker,
            expert_ids=experts,
            parallel=True
        )
        
        # 2. Synthesize findings
        synthesis_prompt = render_prompt("orchestrator/synthesis", {
            "query": query,
            "ticker": ticker,
            "expert_results": [r.model_dump() for r in expert_results]
        })
        
        synthesis = await self.synthesis_model.generate(
            messages=[{"role": "user", "content": synthesis_prompt}],
            response_model=SynthesizedReport
        )
        
        return synthesis
    
    def list_available_experts(self) -> List[dict]:
        """List all available experts."""
        return self.registry.list_experts()
```

---

## Prompt Templates

### `configs/prompts/experts/risk_analyst.jinja2`

**Updated with Forensic Accountant persona aligned with [docs/analyst_agent.py](../../docs/analyst_agent.py):**

```jinja2
You are a **FORENSIC ACCOUNTANT** analyzing SEC filings for InSights-ai.

## Your Persona
- Skeptical and Professional: Look for what management is HIDING
- Forensic and Direct: Ignore buzzwords unless strategic shift
- Evidence-Based: Every claim grounded in quotes and citations
- Mission: Find "hidden truth" via semantic shifts and strategic pivots

## Your Core Skills
1. STRUCTURAL DRIFT ANALYSIS: Track rank changes (e.g., "#10 to #1")
2. SEMANTIC DRIFT DETECTION: Where "potential" became "material"
3. STRATEGIC IMPLICATION: "So What?" for every finding

## MANDATORY Safety Gates
1. GROUNDED CITATIONS: Exact quote + paragraph reference
2. BOILERPLATE SUPPRESSION: Ignore minor wording tweaks
3. DRIFT SCORING: Classify as Structural/Semantic/Both
4. "NOT FOUND": If no evidence, say "Not found." Don't guess.

## CRITICAL PROTOCOL: Two-Filing Comparison
MUST fetch TWO DISTINCT Item 1A sections:
- get_filing_section(ticker, section="1A", year=YEAR_CURRENT)
- get_filing_section(ticker, section="1A", year=YEAR_PREVIOUS)
- Verify different years/accessions - DON'T compare filing to itself

{% if ticker %}
## Current Assignment
Analyzing **{{ ticker }}** {% if sector %}in the {{ sector }} sector{% endif %}.
{% endif %}
```

### `configs/prompts/experts/lead_strategist.jinja2`

```jinja2
You are the **Chief Research Synthesizer** for InSights-ai.

## Task
Synthesize findings from multiple domain experts into a unified investment research report.

## Expert Findings
{% for result in expert_results %}
### {{ result.expert_name }}
**Confidence**: {{ result.confidence }}

{{ result.summary }}

**Details**: {{ result.details | tojson }}
---
{% endfor %}

## Synthesis Requirements
1. **Integrate** findings across all experts
2. **Identify** corroborating evidence
3. **Highlight** contradictions or uncertainties
4. **Weight** by confidence scores
5. **Conclude** with unified recommendations

## Output Structure
- Executive Summary (2-3 sentences)
- Integrated Analysis (key themes across experts)
- Risk-Reward Assessment
- Actionable Recommendations (prioritized)
```

---

## Adding a New Expert

To add a new domain expert:

### 1. Create Expert Class

```python
# insights/experts/esg.py
from .base import BaseExpert

class ESGExpert(BaseExpert):
    name = "ESG Analyst"
    
    def equip_tools(self):
        return [ESGToolkit(), SustainabilityToolkit()]
    
    def get_system_prompt(self, context=None):
        return render_prompt("experts/esg_analyst", context or {})
    
    # ... other methods
```

### 2. Register in YAML

```yaml
# configs/experts.yaml
experts:
  esg_analyst:
    class: "insights.experts.esg.ESGExpert"
    name: "ESG Analyst"
    model: "anthropic/claude-3.5-sonnet"
    tools: ["esg_toolkit", "sustainability_toolkit"]
    sources: ["sustainability reports", "CDP", "MSCI ESG"]
    description: "Evaluates environmental, social, and governance factors."
    priority: 2
```

### 3. Create Prompt Template

```jinja2
{# configs/prompts/experts/esg_analyst.jinja2 #}
You are a Senior ESG Analyst...
```

---

## Testing

```python
# tests/unit/experts/test_registry.py
import pytest
from insights.experts.registry import ExpertRegistry

def test_expert_routing():
    registry = ExpertRegistry()
    
    # Risk query should route to risk expert
    experts = registry.route_query("What are the main risk factors for AAPL?")
    assert "risk_analyst" in experts
    
    # Patent query should route to IP expert
    experts = registry.route_query("Analyze Apple's patent portfolio")
    assert "ip_analyst" in experts


@pytest.mark.asyncio
async def test_parallel_consultation():
    registry = ExpertRegistry()
    results = await registry.consult_experts(
        query="Full analysis of AAPL",
        ticker="AAPL",
        expert_ids=["risk_analyst", "tech_analyst"]
    )
    assert len(results) == 2
```