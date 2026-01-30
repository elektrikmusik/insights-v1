---
description: Manage source control and CI/CD for GitHub.
---

# GitHub Source Control and CI/CD Workflow

This workflow covers pushing code, managing Pull Requests, and ensuring CI/CD pipelines are green.

## 1. Prepare for Push
- **Verify Build**: Run local build/test commands before pushing.
  - Backend: `pytest`
  - Frontend: `npm run build`
- **Lint**: Ensure code follows style guidelines.
  - Backend: `ruff check`
  - Frontend: `npm run lint`

## 2. Commit and Push
- **Create Branch**: `git checkout -b feat/your-feature`
- **Stage Changes**: `git add .`
- **Commit**: Use Conventional Commits (e.g., `feat: add risk analysis agent`)
- **Push**: `git push origin feat/your-feature`

## 3. Create Pull Request
- Use the GitHub CLI (`gh pr create`) or the GitHub web interface.
- Add a clear description of the changes and link to any relevant issues.
- Assign reviewers if applicable.

## 4. Monitor CI/CD
- Navigate to the **Actions** tab on GitHub.
- Ensure all checks pass:
  - `Backend CI`: Pytest and Linting.
  - `Frontend CI`: Linting and Next.js Build.
- If a check fails, fix it locally, commit, and push to the same branch.

## 5. Merging
- Once CI is green and reviews are approved, merge the PR into `main`.
- Delete the feature branch after merging to keep the repo clean.

## Troubleshooting CI/CD
- **Dependency Issues**: Check `requirements.txt` or `package.lock.json`. Ensure all new dependencies are committed.
- **Path Filtering**: Verify the `paths` section in `.github/workflows/*.yml` if a workflow isn't triggering.
- **Environment Variables**: Use GitHub Secrets for any required API keys during CI (e.g., E2E tests).
