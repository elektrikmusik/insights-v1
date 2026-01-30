---
description: Create a new Agno agent or workflow component.
---

# Create Agno Agent

1.  **Identify the Agent's Role**: Determine what task this agent will perform (e.g., "Researcher", "Writer", "Reviewer").
2.  **Choose a Template**:
    - For a standard agent, use `.agent/skills/agno-developer/templates/agent_template.py`.
    - For a workflow (orchestration), use `.agent/skills/agno-developer/templates/workflow_template.py`.
3.  **Create the File**:
    - Run: `cp .agent/skills/agno-developer/templates/agent_template.py backend/src/agents/[agent_name]_agent.py` (or appropriate path).
4.  **Customize**:
    - Update the class name (e.g., `ResearcherAgent`).
    - Update the `instructions` in the `__init__` method.
    - Add any specific tools or methods needed.
5.  **Register**:
    - If applicable, import and add the new agent to `backend/src/agents/manager_agent.py` or wherever orchestration happens.

