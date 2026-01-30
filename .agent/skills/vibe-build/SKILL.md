# Vibe-Build Skill

Build your MVP following the `AGENTS.md` plan. Use when the user wants to start building, implement features, or says "build my MVP", "start coding", or "implement the project".

## Your Role

Execute the plan in `AGENTS.md` to build the MVP incrementally, testing after each feature.

## Prerequisites

Check for required files:

1. `AGENTS.md` - REQUIRED (master plan)
2. `agent_docs/` directory - REQUIRED (detailed specs)
3. `docs/PRD-*.md` - Reference for requirements
4. `docs/TechDesign-*.md` - Reference for implementation

If missing, suggest running `/vibe-step4` first.

## Workflow: Plan -> Execute -> Verify

### 1. Plan Phase

Before any coding:

1. Read `AGENTS.md` to understand current phase and tasks
2. Load relevant `agent_docs/` files for the current task
3. Propose a brief implementation plan
4. Wait for user approval before proceeding

Example:
> **Plan for: User Authentication**
> 1. Set up auth provider (Supabase/Firebase)
> 2. Create login/signup components
> 3. Add protected route wrapper
> 4. Test login flow
>
> Shall I proceed?

### 2. Execute Phase

After approval:

1. Implement ONE feature at a time
2. Follow patterns in `agent_docs/code_patterns.md`
3. Use tech stack from `agent_docs/tech_stack.md`
4. Keep changes focused and minimal
5. Commit after each working feature

### 3. Verify Phase

After each feature:

1. Run tests: `npm test` (or equivalent)
2. Run linter: `npm run lint`
3. Manual smoke test if needed
4. Fix any issues before moving on
5. Update `AGENTS.md` current state

## Build Order

Follow the phases in `AGENTS.md`:

### Phase 1: Foundation
1. Initialize project with chosen stack
2. Set up development environment
3. Configure database connection
4. Set up authentication
5. Create basic project structure

### Phase 2: Core Features
Build each feature from the PRD:
1. Identify the simplest implementation
2. Create database schema if needed
3. Build backend logic
4. Create frontend components
5. Connect and test end-to-end

### Phase 3: Polish
1. Add error handling
2. Improve mobile responsiveness
3. Add loading states
4. Optimize performance
5. Add analytics

### Phase 4: Launch
1. Deploy to production
2. Set up monitoring
3. Run through launch checklist
4. Document any manual steps

## Progress Updates

After completing each feature, update `AGENTS.md`:

```markdown
## Current State
**Last Updated:** [Today's date]
**Working On:** [Next task]
**Recently Completed:** [What was just finished]
**Blocked By:** None
```

Mark completed items in the roadmap:
```markdown
### Phase 2: Core Features
- [x] User authentication
- [ ] [Next feature]
```

## Error Handling

If something breaks:

1. Don't apologize - just fix it
2. Explain briefly what went wrong
3. Show the fix
4. Verify it works
5. Move on

## What NOT To Do

- Do NOT delete files without confirmation
- Do NOT change database schemas without backup plan
- Do NOT add features outside current phase
- Do NOT skip verification steps
- Do NOT use deprecated patterns
- Do NOT over-engineer simple features
