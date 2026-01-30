---
description: Initialize a project with professional .github and .gitignore defaults.
---

# Project Initialization Workflow

Use this workflow to quickly scaffold the necessary meta-files for a new repository or monorepo.

## 1. Initialize .gitignore
- **For Root Monorepo**: 
  - Run: `cp .agent/skills/repo-init/templates/root.gitignore .gitignore`
- **For Python Sub-projects**:
  - Run: `cp .agent/skills/repo-init/templates/python.gitignore [sub-dir]/.gitignore`
- **For Node/Next.js Sub-projects**:
  - Run: `cp .agent/skills/repo-init/templates/node.gitignore [sub-dir]/.gitignore`

## 2. Initialize .github Folder
- Create the target directory: `mkdir -p .github/workflows`
- **Add CI Workflows**:
  - Use the patterns from `github-specialist` skill.
  - Run: `cat > .github/workflows/monorepo-ci.yml <<EOF ...` (or use existing template files).

## 3. Setup Environment Examples
- For each directory containing a `.env` file, ensure there is a `.env.example`.
- Strip all sensitive values and keep only the keys.

## 4. Verify Git Status
- Run `git status` to ensure all necessary files are tracked and ignored files are hidden.
- Review `.gitignore` if any unwanted files (like `node_modules` or `__pycache__`) are still shown.

## 5. Commit Initial Meta-files
- `git add .github .gitignore README.md`
- `git commit -m "chore: initialize project meta-files and CI/CD"`
