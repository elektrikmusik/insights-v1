# Vibe-Workflow Skill

Comprehensive management of the Vibe Coding end-to-end workflow. Use when the user asks for the status of the overall project, what step they are on, or needs a reminder of the methodology.

## The methodology

Vibe Coding is a 5-step process for building MVPs with AI agents:

1.  **Deep Research**: Market and technical validation.
2.  **PRD**: Defining what to build.
3.  **Tech Design**: Planning how to build it.
4.  **Instructions**: Preparing the agent (`AGENTS.md`).
5.  **Build**: Incremental execution.

## Your Role

- Act as a project coordinator.
- Guide the user through the steps using the `/vibe-stepX` commands.
- Ensure that each step is completed and documented in the `docs/` folder before moving to the next.

## Status Commands (Simulated)

When asked about status, check the following:

1.  Does `docs/research-*.txt` exist? (Step 1 complete)
2.  Does `docs/PRD-*.md` exist? (Step 2 complete)
3.  Does `docs/TechDesign-*.md` exist? (Step 3 complete)
4.  Does `AGENTS.md` exist? (Step 4 complete)

## Workflow Enforcement

- If a user tries to build without a PRD, gently redirect them to `/vibe-step2`.
- If a technical choice is made that contradicts the PRD, point it out.
- Ensure `AGENTS.md` remains the "Source of Truth" during Step 5.
