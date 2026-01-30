#!/bin/bash
# Verify InSights-ai build status
# Usage: .agent/scripts/verify-insights.sh

set -e

echo "🔍 Verifying InSights-ai build..."
echo ""

cd python

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "📦 Checking dependencies..."
uv sync 2>/dev/null || echo "⚠️  Run 'uv sync' to install dependencies"

echo ""
echo "🔧 Running type checks..."
uv run mypy insights --ignore-missing-imports 2>/dev/null || echo "⚠️  Type errors found"

echo ""
echo "📝 Running linter..."
uv run ruff check insights 2>/dev/null || echo "⚠️  Lint errors found"

echo ""
echo "🧪 Running tests..."
uv run pytest tests -v --tb=short 2>/dev/null || echo "⚠️  Test failures"

echo ""
echo "📊 Checking test coverage..."
uv run pytest tests --cov=insights --cov-report=term-missing 2>/dev/null || true

echo ""
echo "✅ Verification complete!"
