---
description: Create a new Expert for the Team of Experts system
---

# Create New Expert Workflow

Reference: `docs/specs/13_EXPERT_REGISTRY.md` (Adding New Experts section)

## Required Information

Ask the user for:
1. **Expert Name** (e.g., `macro_analyst`, `esg_expert`)
2. **Domain Description** (what does this expert analyze?)
3. **Preferred Model** (e.g., `openai/gpt-4o`, `deepseek-ai/DeepSeek-V3`)
4. **Required Tools** (e.g., SEC MCP, news API, custom toolkit)
5. **Data Sources** (e.g., 10-K filings, patents, GitHub repos)
6. **Output Schema** (what fields should the analysis contain?)

## Implementation Steps

1. **Add to Registry Config**
   - Edit `configs/experts.yaml`
   - Add new expert entry with class, model, tools, sources

2. **Create Expert Class**
   - Create `python/insights/experts/{name}.py`
   - Extend `BaseExpert`
   - Implement `equip_tools()`, `get_system_prompt()`, `conduct_research()`

3. **Create Prompt Template**
   - Create `configs/prompts/experts/{name}.jinja2`
   - Include role, scope, output format instructions

4. **Create Toolkit (if needed)**
   - If new tools are required, create in `python/insights/adapters/mcp/`
   - Register with Agno toolkit pattern

5. **Add Output Schema**
   - Add Pydantic model in `python/insights/core/types.py`
   - Ensure it matches the expert's output format

6. **Write Tests**
   - Create `python/tests/unit/experts/test_{name}.py`
   - Test `can_handle()` routing
   - Test `conduct_research()` with mocked LLM

7. **Verify Registration**
   - Run the registry loader to ensure expert is discovered
   - Test routing with sample query

## Template Files

```python
# python/insights/experts/{name}.py
from insights.experts.base import BaseExpert

class {ClassName}Expert(BaseExpert):
    async def equip_tools(self) -> list:
        # Return Agno toolkits
        pass
    
    def get_system_prompt(self) -> str:
        # Load Jinja2 template
        pass
    
    async def conduct_research(self, query: str) -> ExpertResult:
        # Run analysis
        pass
```

## Verification

// turbo
- Run `uv run pytest python/tests/unit/experts/ -v`
- Check expert appears in registry list
