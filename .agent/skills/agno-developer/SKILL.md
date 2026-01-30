---
name: agno-developer
description: Expert patterns and workflows for building agents with Agno framework.
---

# Agno Developer Skill

This skill provides expert guidance on building, testing, and deploying agents using the Agno framework. It encapsulates best practices from the Agno documentation and project-specific patterns.

## Core Concepts

Agno is an agent platform that simplifies building multi-agent systems. Key components:
- **Agent**: The fundamental unit of intelligence. Can use tools, memory, and knowledge.
- **Team**: A group of agents working together.
- **Workflow**: A deterministic sequence of steps (Agents, Teams, or Functions).

## Best Practices

### 1. Agent definition
- **Role & Mission**: Clearly define the agent's persona and objective in the `instructions`.
- **Model Selection**: Use `agno.models.google.Gemini` (e.g., `gemini-1.5-flash` or `gemini-2.0-flash`) for optimal performance/cost balance in this project.
- **Tools**: Equip agents with specific Python functions as tools. Ensure tools have proper type hints and docstrings.
- **Storage**: Use Supabase or local storage for persisting agent sessions.

### 2. Project Structure
- `backend/src/agents/`: specialized agent classes (e.g., `AnalystAgent`, `ManagerAgent`).
- `backend/src/tools/`: shared tools and utilities.
- `backend/src/workflows/`: (Optional) If using Agno Workflows, place them here.

### 3. Common Patterns
- **Orchestrator Pattern**: Use a `ManagerAgent` to coordinate sub-agents (`SecretaryAgent`, `AnalystAgent`).
- **State Management**: Persist critical state to a database (Supabase) rather than relying solely on agent memory for long-term data.
- **Structured Output**: Where possible, use Pydantic models for `response_model` to get structured data, or clear markdown instructions if text is preferred.

## Workflow: Creating a New Agent

To create a new agent, follow these steps:
1.  **Define the Class**: Create a new file in `backend/src/agents/` (e.g., `researcher_agent.py`).
2.  **Initialize**: In `__init__`, setup the `agno.agent.Agent` with model, description, and instructions.
3.  **Implement Methods**: Add async methods for specific tasks the agent performs (e.g., `analyze_report`, `find_data`).
4.  **Register**: If part of a team, import and instantiate it in the Manager/Orchestrator.

## Resources
- [Agno Documentation](https://docs.agno.com)
- [Project Agents](backend/src/agents/)

## Templates
See `templates/` directory for starter code.
