---
description: Initialize the project structure for Vibe Coding (docs/, agent_docs/, AGENTS.md, etc.)
---

# Vibe Init

Initialize the recommended project structure for AI-powered MVP development.

## Project Structure
The following structure will be created (excluding `src/` as requested):
```
.
├── docs/                # Research, PRD, and Tech Design documents
├── agent_docs/          # Modular instructions for the AI Agent
└── AGENTS.md            # Universal master plan for AI assistants
```

## Steps
// turbo-all
1. Create the `docs/` directory:
   `mkdir -p docs`

2. Create the `agent_docs/` directory:
   `mkdir -p agent_docs`

3. Create a placeholder `AGENTS.md` in the root (to be filled in Step 4):
   `touch AGENTS.md`

4. Create a `GEMINI.md` file in the root for Antigravity context:
   `touch GEMINI.md`

5. (Optional) Create common agent doc files:
   `touch agent_docs/tech_stack.md agent_docs/code_patterns.md agent_docs/project_brief.md agent_docs/product_requirements.md agent_docs/testing.md`

## Next Steps
After initialization, start with Step 1:
`/vibe-step1`
