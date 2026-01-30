---
description: Perform Step 4 of the Vibe Workflow: Generate AI Agent Instructions. Creates AGENTS.md and agent_docs/.
---

# Vibe Step 4: Generate Instructions

Converts all documentation into step-by-step coding instructions for AI agents.

## Prerequisites
- Approved PRD and Technical Design from `docs/`.

## Workflow

1. **Information Extraction**:
   - Read the PRD and Tech Design.
   - Extract: Features, Stack, Roadmap, Folder Structure, Constraints.

2. **Generate AGENTS.md**:
   - Create the Master Plan in `AGENTS.md`.
   - Include the "How I Should Think" and "Plan -> Execute -> Verify" rules.
   - Standardize on the "Progressive Disclosure" system.

3. **Generate agent_docs/**:
   - Populated detailed files:
     - `agent_docs/tech_stack.md`
     - `agent_docs/code_patterns.md`
     - `agent_docs/project_brief.md`
     - `agent_docs/product_requirements.md`
     - `agent_docs/testing.md`

4. **Generate Antigravity Config**:
   - Create `GEMINI.md` (or update it) with specific directives for Antigravity to always read `AGENTS.md` first.

## Next Step
Ready to start building:
`/vibe-step5`
