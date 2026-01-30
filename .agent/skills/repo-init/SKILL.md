---
name: repo-init
description: Professional repository initialization patterns including .github, .gitignore, and project structure.
---

# Repository Initialization Skill

This skill provides patterns for setting up a new project or sub-module with professional standards.

## Essential Files

Every professional repository should have:
1.  **.gitignore**: Specific to the language and framework.
2.  **.github/**: Workflows for CI/CD, Issue Templates, and Pull Request Templates.
3.  **README.md**: Clear setup instructions and project overview.
4.  **.env.example**: Shared environment variable keys without secrets.

## Scaffolding Patterns

### 1. Root Monorepo structure
Ensure the root `.gitignore` covers global patterns (OS files, IDE files) and ignores common build artifacts from sub-packages.

### 2. GitHub Workflows
- `ci.yml`: For building and testing.
- `release.yml`: For automated versioning and deployment.
- `dependabot.yml`: To keep dependencies up to date.

### 3. README Best Practices
- Badges for CI status.
- "Quick Start" section.
- "Architecture" overview.
- "Contributing" guide.

## templates/ Directory
This skill includes ready-to-use templates for:
- `root.gitignore`
- `python.gitignore`
- `node.gitignore`
- `.github/workflows/` (via `github-specialist` skill)

## Workflow: Initialize Repository
Use the `/repo-init` workflow to apply these patterns to the current directory.
