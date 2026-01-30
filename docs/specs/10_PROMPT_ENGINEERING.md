# 10. Prompt Engineering

## Overview

All system prompts are stored as **Jinja2 templates** in `configs/prompts/`. This approach enables:
- Version control for prompts
- Dynamic injection of context
- A/B testing of prompt variants
- Separation of prompts from code

---

## Directory Structure

```
configs/
└── prompts/
    ├── research_agent_system.jinja2
    ├── risk_extractor.jinja2
    ├── report_generator.jinja2
    └── analysis_summary.jinja2
```

---

## Research Agent System Prompt

### `configs/prompts/research_agent_system.jinja2`

```jinja2
{#
  Research Agent System Prompt
  Version: 1.0.0
  Updated: 2024-01-15
#}

You are a **Deep Research Analyst** specializing in SEC filing analysis, risk assessment, and financial sentiment analysis.

## Your Role
You analyze SEC 10-K and 10-Q filings to identify risk factors, track changes over time, and provide actionable insights for investors and analysts.

## Core Principles
1. **Accuracy First**: Only report information directly from SEC filings. Never fabricate data.
2. **Cite Sources**: Always reference specific filing sections (e.g., "Item 1A, Risk Factors").
3. **Quantify Changes**: Express rank changes as numbers (e.g., "+3 positions").
4. **Analyze, Don't Summarize**: Provide insights, not just summaries.

## Available Tools
You have access to these toolkits:
{% for toolkit in toolkits %}
- **{{ toolkit.name }}**: {{ toolkit.description }}
{% endfor %}

## Workflow for Risk Drift Analysis
1. **Retrieve Filings**: Use `SECToolkit.get_filing_section()` to fetch Risk Factors (Item 1A) for both years.
2. **Extract Risks**: Use `FilingToolkit.extract_risk_factors()` to parse individual risks.
3. **Analyze Sentiment**: Use `SentimentToolkit.analyze_sentiment()` on each risk.
4. **Calculate Drift**: Use `RiskToolkit.calculate_drift()` to compare years.
5. **Generate Heatmap**: Use `RiskToolkit.generate_heatmap()` for visualization.
6. **Synthesize Report**: Combine findings into executive summary.

{% if company_context %}
## Company Context
- **Ticker**: {{ company_context.ticker }}
- **Sector**: {{ company_context.sector }}
- **Market Cap**: {{ company_context.market_cap }}
{% endif %}

## Output Format
Your final response MUST include:

### Executive Summary
2-3 sentences highlighting the most critical risk changes.

### Risk Drift Analysis
| Risk | Current Rank | Δ Rank | Zone | Insight |
|------|--------------|--------|------|---------|
| ... | ... | ... | ... | ... |

### Zone Distribution
- 🔴 **Critical**: [count] risks with major escalation
- 🟠 **Warning**: [count] risks with moderate change
- 🔵 **New**: [count] newly identified risks
- ⚪ **Stable**: [count] unchanged risks

### Strategic Recommendations
1. [Recommendation based on critical risks]
2. [Recommendation based on new risks]
3. [Recommendation for monitoring]

## Constraints
- Maximum response length: 4000 tokens
- Do NOT perform calculations yourself - always use tools
- Do NOT make up data - report errors if tools fail
- Current date: {{ current_date }}
```

---

## Prompt Loader

### `insights/core/prompts.py`

```python
"""
Prompt template management.
Loads Jinja2 templates from configs/prompts/.
"""
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, Template
from datetime import datetime, UTC

PROMPTS_DIR = Path(__file__).parent.parent.parent / "configs" / "prompts"


class PromptManager:
    """
    Manages prompt templates with Jinja2.
    
    Usage:
        manager = PromptManager()
        prompt = manager.render("research_agent_system", context={...})
    """
    
    def __init__(self, prompts_dir: Path = PROMPTS_DIR):
        self.env = Environment(
            loader=FileSystemLoader(prompts_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self._cache: Dict[str, Template] = {}
    
    def render(
        self, 
        template_name: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Render a prompt template with context.
        
        Args:
            template_name: Template file name (without .jinja2)
            context: Variables to inject
            
        Returns:
            Rendered prompt string
        """
        if not template_name.endswith(".jinja2"):
            template_name = f"{template_name}.jinja2"
        
        if template_name not in self._cache:
            self._cache[template_name] = self.env.get_template(template_name)
        
        # Default context
        default_context = {
            "current_date": datetime.now(UTC).strftime("%Y-%m-%d"),
            "toolkits": []
        }
        
        if context:
            default_context.update(context)
        
        return self._cache[template_name].render(**default_context)


# Singleton
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def render_prompt(template_name: str, **kwargs) -> str:
    """Convenience function to render a prompt."""
    return get_prompt_manager().render(template_name, kwargs)
```

---

## Usage in Agent

```python
# insights/agents/research/agent.py

from insights.core.prompts import render_prompt


def get_research_agent() -> Agent:
    # Render system prompt with context
    system_prompt = render_prompt(
        "research_agent_system",
        toolkits=[
            {"name": "SECToolkit", "description": "SEC filing retrieval"},
            {"name": "SentimentToolkit", "description": "FinBERT sentiment"},
            {"name": "RiskToolkit", "description": "Drift analysis"},
        ],
        company_context=None  # Populated per request
    )
    
    return Agent(
        model=get_chat_model(),
        description=system_prompt,
        # ... other config
    )
```

---

## Additional Prompts

### `configs/prompts/risk_extractor.jinja2`

```jinja2
{# Extracts individual risk factors from 10-K text #}

Extract each distinct risk factor from the following SEC 10-K filing text.

For each risk, provide:
1. **Title**: A concise title (max 10 words)
2. **Content**: The full text of that risk factor
3. **Rank**: Order of appearance (1 = first mentioned)

## Rules
- Each risk should be a separate item
- Preserve the original language
- Do not combine multiple risks
- Ignore Table of Contents entries

## Filing Text
{{ filing_text }}

## Output Format (JSON)
[
  {"rank": 1, "title": "...", "content": "..."},
  {"rank": 2, "title": "...", "content": "..."}
]
```

### `configs/prompts/report_generator.jinja2`

```jinja2
{# Generates final Markdown report from analysis #}

Generate a professional risk drift report for {{ ticker }}.

## Analysis Data
{{ drift_data | tojson }}

## Heatmap Data
{{ heatmap_data | tojson }}

## Report Requirements
1. Executive summary (2-3 sentences)
2. Key findings table
3. Zone breakdown
4. Strategic recommendations

Use Markdown formatting. Be concise and actionable.
```

---

## Prompt Testing

```python
# tests/unit/test_prompts.py

from insights.core.prompts import render_prompt


def test_research_agent_prompt_renders():
    """Test prompt renders without errors."""
    prompt = render_prompt("research_agent_system")
    
    assert "Deep Research Analyst" in prompt
    assert "Current date" not in prompt  # Should be replaced


def test_prompt_context_injection():
    """Test context variables are injected."""
    prompt = render_prompt(
        "research_agent_system",
        company_context={"ticker": "AAPL", "sector": "Technology"}
    )
    
    assert "AAPL" in prompt
    assert "Technology" in prompt
```

---

## Team of Experts Prompts

The current implementation uses a **Team of Experts** architecture with an Orchestrator coordinating specialized experts. Key prompts:

- **Risk Analyst** ([configs/prompts/experts/risk_analyst.jinja2](../../configs/prompts/experts/risk_analyst.jinja2)): Implements **Forensic Accountant** persona aligned with [docs/analyst_agent.py](../../docs/analyst_agent.py). Includes mandatory safety gates (grounded citations, boilerplate suppression, "NOT FOUND" protocol) and **CRITICAL PROTOCOL** requiring TWO DISTINCT filings for comparison.
- **Lead Strategist** ([configs/prompts/experts/lead_strategist.jinja2](../../configs/prompts/experts/lead_strategist.jinja2)): Lead Investment Strategist prompt for synthesizing multi-expert findings with emphasis on two-period comparisons and material drift (structural/semantic).

See [13_EXPERT_REGISTRY.md](13_EXPERT_REGISTRY.md) for full documentation of the expert system.

---

## Best Practices

1. **Version prompts** with comments at the top
2. **Test prompts** independently of agents
3. **Log prompt versions** for debugging
4. **Use structured output** hints where possible
5. **Keep prompts focused** - one task per prompt