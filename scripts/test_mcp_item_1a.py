#!/usr/bin/env python
"""
Test MCP call to ensure we can return Item 1A (Risk Factors) from SEC filings.

Uses the same SECToolkit/MCP client as the InSights API to call sec-edgar-mcp,
fetches Item 1A for a given ticker and year, and writes the result to JSON.

Usage:
  # From repo root (requires MCP_SEC_URL and sec-edgar-mcp running)
  PYTHONPATH=python python scripts/test_mcp_item_1a.py [--ticker NVDA] [--year 2024] [--output item_1a_mcp_test.json]

  # From python dir
  uv run python ../scripts/test_mcp_item_1a.py --output ../scripts/item_1a_mcp_test.json
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure insights package is on path when run from scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_DIR = _REPO_ROOT / "python"
if _PYTHON_DIR.exists() and str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from insights.adapters.mcp.toolkit import SECToolkit
from insights.adapters.mcp.client import MCPClient, MCPConfig, get_mcp_client
from insights.core.config import settings


async def fetch_item_1a(ticker: str, year: int) -> dict:
    """Call MCP via SECToolkit to get Item 1A for a 10-K. Returns a result dict."""
    client = get_mcp_client()
    toolkit = SECToolkit(mcp_client=client)
    section_text = await toolkit.get_filing_section(
        ticker=ticker,
        section="1A",
        form_type="10-K",
        year=year,
    )
    return {
        "ticker": ticker,
        "year": year,
        "section": "1A",
        "form_type": "10-K",
        "content": section_text,
        "content_length": len(section_text),
        "success": (
            not section_text.startswith("Section '1A' not found")
            and not section_text.startswith("No filing found")
            and not section_text.startswith("Error parsing")
            and len(section_text.strip()) > 200
        ),
    }


async def run_test(ticker: str, year: int, output_path: str | None) -> int:
    """Run the Item 1A MCP test and optionally write JSON output."""
    mcp_url = getattr(settings, "MCP_SEC_URL", os.environ.get("MCP_SEC_URL", "http://localhost:8080/sse"))
    print("=" * 60)
    print("MCP Item 1A (Risk Factors) test")
    print("=" * 60)
    print(f"  Ticker:     {ticker}")
    print(f"  Year:       {year}")
    print(f"  MCP URL:    {mcp_url}")
    print("=" * 60)

    try:
        result = await fetch_item_1a(ticker, year)
    except Exception as e:
        print(f"  Error: {e}")
        report = {
            "success": False,
            "error": str(e),
            "ticker": ticker,
            "year": year,
        }
        if output_path:
            Path(output_path).write_text(json.dumps(report, indent=2))
            print(f"  Report saved to: {output_path}")
        return 1

    # Build report (excerpt for readability in JSON)
    excerpt = result["content"][:2000] + ("..." if len(result["content"]) > 2000 else "")
    report = {
        "success": result["success"],
        "ticker": result["ticker"],
        "year": result["year"],
        "section": result["section"],
        "form_type": result["form_type"],
        "content_length": result["content_length"],
        "content_excerpt": excerpt,
        "full_content": result["content"] if result["content_length"] < 50_000 else None,
    }
    if result["content_length"] >= 50_000:
        report["note"] = "full_content omitted (too long); see content_excerpt."

    if result["success"]:
        print(f"  Item 1A returned: {result['content_length']} chars")
        print(f"  Excerpt: {result['content'][:300]}...")
    else:
        print(f"  Item 1A not returned or too short. Response: {result['content'][:500]}")

    if output_path:
        out = Path(output_path)
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"  Report saved to: {out.absolute()}")

    print("=" * 60)
    print("  Result:", "PASS – Item 1A returned from filings" if result["success"] else "FAIL – Item 1A not returned")
    print("=" * 60)
    return 0 if result["success"] else 1


def main():
    parser = argparse.ArgumentParser(description="Test MCP call for Item 1A from SEC filings")
    parser.add_argument("--ticker", default="NVDA", help="Ticker symbol (default: NVDA)")
    parser.add_argument("--year", type=int, default=2024, help="Fiscal year (default: 2024)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    args = parser.parse_args()
    exit_code = asyncio.run(run_test(args.ticker, args.year, args.output))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
