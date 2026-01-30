---
name: github-specialist
description: Expert patterns for source control, GitHub Actions, and CI/CD pipelines.
---

# GitHub Specialist Skill

This skill provides patterns and best practices for managing source control and CI/CD pipelines using GitHub.

## Branching Strategy
- **main**: Production-ready code. No direct commits (ideally).
- **feat/* / bugfix/* / chore/***: Feature and fix branches.
- **Pull Requests**: Every change should go through a PR with CI verification.

## Commit Message Standards
Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat: add new analysis tool`
- `fix: resolve auth timeout issue`
- `docs: update setup guide`
- `ci: add frontend build workflow`

## GitHub Actions Patterns

### 1. Backend CI (Python)
- **Path**: `.github/workflows/backend-ci.yml`
- **Actions**: Checkout, Set up Python, Install dependencies, Run tests (pytest), Lint (ruff).

### 2. Frontend CI (Next.js/React)
- **Path**: `.github/workflows/frontend-ci.yml`
- **Actions**: Checkout, Set up Node, Cache dependencies, Install, Lint, Build.

### 3. Deployment (Optional)
- **Vercel**: Automatic for frontend.
- **Fly.io/Render/AWS**: For backend.
- **Supabase**: Migration management with `supabase-mcp-server`.

## Best Practices
- **Path Filtering**: Only trigger workflows if relevant files change (e.g., `paths: ['backend/**']`).
- **Caching**: Use `actions/cache` or built-in caching in `setup-python`/`setup-node` to speed up builds.
- **Secrets Management**: Use GitHub Actions Secrets for `SUPABASE_KEY`, `GEMINI_API_KEY`, etc. NEVER hardcode secrets.
- **Matrix Builds**: Test against multiple Python/Node versions if necessary.

## Workflow: Creating a New CI Pipeline
1. Determine the build/test commands for the component.
2. Create the `.yml` file in `.github/workflows/`.
3. Add path filtering to avoid unnecessary runs.
4. Push to a branch and verify the "Actions" tab in GitHub.
