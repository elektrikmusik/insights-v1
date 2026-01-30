# Vibe Coding Skill

Expert guide for AI-powered MVP development using the Vibe Coding methodology. Use this skill when guiding users through the end-to-end process of building a product with AI agents.

## Core Principles

1. **Progressive Disclosure**: Documentation should be modular. `AGENTS.md` is the master plan; `agent_docs/` contains the details. This prevents context limit issues.
2. **Plan -> Execute -> Verify**: Never write code without a plan. Always verify code after writing it.
3. **Product-Led Engineering**: Technical decisions must serve the Product Requirements (PRD).
4. **Anti-Vibe Engineering**: While the *process* is fast ("vibing"), the *code* must be solid (strict types, thin controllers, error handling).

## The Steps

### Phase 1: Deep Research
Analyzes market opportunity, competitors, and technical feasibility.
- **Output**: `docs/research-[AppName].txt`

### Phase 2: PRD (Product Requirements Document)
Defines WHAT you are building, WHO it's for, and WHY it matters.
- **Output**: `docs/PRD-[AppName]-MVP.md`

### Phase 3: Tech Design
Decides the best tech stack and implementation approach.
- **Output**: `docs/TechDesign-[AppName]-MVP.md`

### Phase 4: Instructions
Converts documentation into agent-ready instructions.
- **Output**: `AGENTS.md` and `agent_docs/`

### Phase 5: Build
The execution loop for implementation.

## Agent Behavior Patterns

- **Intent First**: Always confirm intent before acting.
- **Incrementalism**: One feature at a time.
- **Self-Healing**: Run tests/linting and fix failures before completing a task.
- **Documentation First**: Treat markdown docs as living "memory" for the agent.

## Safety & Quality Gates (Hooks Legacy)

To ensure production quality and prevent accidental data loss, follow these rules:

1.  **Protected Files**: Never modify `.env`, `package-lock.json`, or internal `.git/` files directly. Use appropriate commands (e.g., `npm install`) for dependencies.
2.  **Destructive Commands**: Always ask for explicit confirmation before running destructive commands like `rm -rf /`, `DROP DATABASE`, or `TRUNCATE TABLE`.
3.  **Code Formatting**: After writing code, ensure it is formatted (e.g., run `Prettier`) if configured in the project.
4.  **Session Summary**: At the end of a major task, provide a summary of modified files and remind the user to review changes before committing.
5.  **Notifications**: Use the `notify_user` tool to get the user's attention for critical decisions or when blocked.

## Implementation Workflow
When a user runs a `/vibe-*` command:
1. Load the corresponding workflow file.
2. Act as the specialized role (Researcher, Product Manager, Architect, or Developer).
3. Use this skill to ensure consistency with the Vibe methodology.
