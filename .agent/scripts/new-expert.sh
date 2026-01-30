#!/bin/bash
# Generate expert from template
# Usage: .agent/scripts/new-expert.sh <expert_name> <model_id>
# Example: .agent/scripts/new-expert.sh macro_analyst openai/gpt-4o

set -e

EXPERT_NAME=$1
MODEL_ID=${2:-"openai/gpt-4o"}

if [ -z "$EXPERT_NAME" ]; then
    echo "Usage: $0 <expert_name> [model_id]"
    echo "Example: $0 macro_analyst openai/gpt-4o"
    exit 1
fi

# Convert to class name (snake_case to PascalCase)
CLASS_NAME=$(echo "$EXPERT_NAME" | sed -r 's/(^|_)([a-z])/\U\2/g')

echo "🔧 Creating expert: $EXPERT_NAME (class: ${CLASS_NAME}Expert)"

# Create expert Python file
EXPERT_FILE="python/insights/experts/${EXPERT_NAME}.py"
mkdir -p python/insights/experts

cat > "$EXPERT_FILE" << EOF
"""${CLASS_NAME} Expert for InSights-ai."""
from insights.experts.base import BaseExpert
from insights.core.types import ExpertResult

class ${CLASS_NAME}Expert(BaseExpert):
    """
    Expert for ${EXPERT_NAME//_/ } analysis.
    
    Model: ${MODEL_ID}
    """
    
    async def equip_tools(self) -> list:
        """Return Agno toolkits for this expert."""
        return []  # TODO: Add relevant toolkits
    
    def get_system_prompt(self) -> str:
        """Load and render the system prompt."""
        return self.prompt_manager.render(
            "experts/${EXPERT_NAME}.jinja2",
            context=self.context
        )
    
    def can_handle(self, query: str) -> bool:
        """Check if this expert can handle the query."""
        keywords = []  # TODO: Add routing keywords
        return any(kw in query.lower() for kw in keywords)
    
    async def conduct_research(self, query: str) -> ExpertResult:
        """Run the expert's analysis."""
        response = await self.agent.arun(query)
        return ExpertResult(
            expert_id="${EXPERT_NAME}",
            findings=response.content,
            confidence=0.8,
            sources=[]
        )
EOF

echo "✅ Created: $EXPERT_FILE"

# Create prompt template
PROMPT_FILE="configs/prompts/experts/${EXPERT_NAME}.jinja2"
mkdir -p configs/prompts/experts

cat > "$PROMPT_FILE" << EOF
{# ${CLASS_NAME} Expert System Prompt #}
You are a specialized ${EXPERT_NAME//_/ } for InSights-ai.

## Your Role
Analyze companies focusing on ${EXPERT_NAME//_/ } factors.

## Your Capabilities
- Use available tools to gather data
- Provide structured, evidence-based analysis
- Cite sources for all findings

## Output Format
Structure your analysis as:
1. **Key Findings**: Main observations
2. **Supporting Evidence**: Data and sources
3. **Confidence Level**: Your certainty (0-1)
4. **Recommendations**: Actionable insights

{% if company %}
## Current Analysis Target
Company: {{ company.name }}
Ticker: {{ company.ticker }}
{% endif %}
EOF

echo "✅ Created: $PROMPT_FILE"

# Create test file
TEST_FILE="python/tests/unit/experts/test_${EXPERT_NAME}.py"
mkdir -p python/tests/unit/experts

cat > "$TEST_FILE" << EOF
"""Tests for ${CLASS_NAME}Expert."""
import pytest
from unittest.mock import AsyncMock, patch

from insights.experts.${EXPERT_NAME} import ${CLASS_NAME}Expert


class Test${CLASS_NAME}Expert:
    """Test suite for ${CLASS_NAME}Expert."""
    
    def test_can_handle_relevant_query(self):
        """Expert should handle relevant queries."""
        expert = ${CLASS_NAME}Expert(config={})
        # TODO: Update with actual keywords
        assert expert.can_handle("relevant query") or True  # Placeholder
    
    def test_can_handle_irrelevant_query(self):
        """Expert should reject irrelevant queries."""
        expert = ${CLASS_NAME}Expert(config={})
        # Placeholder - update when keywords are set
        pass
    
    @pytest.mark.asyncio
    async def test_conduct_research(self):
        """Expert should return structured result."""
        expert = ${CLASS_NAME}Expert(config={})
        expert.agent = AsyncMock()
        expert.agent.arun.return_value = AsyncMock(content="Analysis result")
        
        result = await expert.conduct_research("test query")
        
        assert result.expert_id == "${EXPERT_NAME}"
        assert result.confidence == 0.8
EOF

echo "✅ Created: $TEST_FILE"

# Add to experts.yaml
YAML_FILE="configs/experts.yaml"
if [ -f "$YAML_FILE" ]; then
    echo ""
    echo "📝 Add this entry to $YAML_FILE:"
    echo ""
fi

cat << EOF
  ${EXPERT_NAME}:
    class: insights.experts.${EXPERT_NAME}.${CLASS_NAME}Expert
    model: "${MODEL_ID}"
    tools: []
    sources: []
    description: "${EXPERT_NAME//_/ } analysis"
    output_schema: ExpertResult
    priority: 5
EOF

echo ""
echo "🎉 Expert scaffold complete!"
echo "Next: Implement the expert logic and add routing keywords"
