#!/usr/bin/env python
"""
Test script for NVDA Risk Analysis via API endpoints.

Submits an analysis request and streams the response via SSE,
saving a comprehensive JSON output for evaluation.

Usage:
    python scripts/test_nvda_risk_analysis.py [--base-url URL] [--years YEAR1 YEAR2] [--output FILE]

Example:
    python scripts/test_nvda_risk_analysis.py --years 2024 2023 --output nvda_analysis.json
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import httpx

# Default configuration
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TICKER = "NVDA"
DEFAULT_YEARS = [2024, 2023]


class RiskAnalysisClient:
    """Client for interacting with the InSights AI API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.events: list[dict[str, Any]] = []
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

    async def check_health(self) -> bool:
        """Check if the API is healthy."""
        # trust_env=False avoids using proxy settings from env vars which might be causing 503s
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), trust_env=False) as client:
            try:
                url = f"{self.base_url}/health"
                print(f"  🔍 Checking health at {url}...")
                response = await client.get(url)
                print(f"  📡 Response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"  📝 Response body: {response.text[:200]}")
                return response.status_code == 200
            except httpx.ConnectError as e:
                print(f"❌ Connection failed: {e}")
                print(f"   Is the API running at {self.base_url}?")
                return False
            except httpx.RequestError as e:
                print(f"❌ Health check failed ({type(e).__name__}): {e}")
                return False

    async def submit_analysis(
        self,
        ticker: str,
        years: list[int],
        options: dict[str, Any] | None = None
    ) -> str | None:
        """Submit an analysis request and return the job ID."""
        payload = {
            "ticker": ticker,
            "years": years,
            "analysis_type": "risk_drift",
            "options": options or {
                "include_sentiment": True,
                "generate_heatmap": True,
                "depth": "deep"
            }
        }

        async with httpx.AsyncClient(trust_env=False) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/analyze",
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                return data.get("job_id")
            except httpx.HTTPStatusError as e:
                print(f"❌ Analysis submission failed: {e.response.status_code}")
                print(f"   Response: {e.response.text}")
                return None
            except httpx.RequestError as e:
                print(f"❌ Request failed: {e}")
                return None

    async def stream_progress(self, job_id: str) -> dict[str, Any]:
        """Stream SSE events for a job and collect all progress updates."""
        self.start_time = datetime.now(UTC)
        self.events = []
        final_result = {}

        print(f"\n📡 Streaming progress for job: {job_id}")
        print("=" * 60)

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), trust_env=False) as client:
            try:
                async with client.stream(
                    "GET",
                    f"{self.base_url}/api/v1/stream/{job_id}",
                    headers={"Accept": "text/event-stream"}
                ) as response:
                    event_type = None
                    event_data = ""

                    async for line in response.aiter_lines():
                        line = line.strip()

                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            event_data = line[5:].strip()
                        elif line == "" and event_data:
                            # End of event
                            event = self._parse_event(event_type, event_data)
                            if event:
                                self.events.append(event)
                                self._print_event(event)

                                # Check for terminal events
                                if event.get("event") in ("job_completed", "job_failed"):
                                    final_result = event
                                    break

                            event_type = None
                            event_data = ""

            except httpx.ReadTimeout:
                print("⏰ Stream timeout - job may still be processing")
            except httpx.RequestError as e:
                print(f"❌ Stream error: {e}")

        self.end_time = datetime.now(UTC)
        return final_result

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get the current status of a job."""
        async with httpx.AsyncClient(trust_env=False) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/api/v1/jobs/{job_id}",
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                print(f"❌ Failed to get job status: {e.response.status_code}")
                return None
            except httpx.RequestError as e:
                print(f"❌ Request failed: {e}")
                return None

    def _parse_event(self, event_type: str | None, data: str) -> dict[str, Any] | None:
        """Parse an SSE event."""
        try:
            parsed = json.loads(data) if data else {}
            parsed["event"] = event_type or parsed.get("event", "unknown")
            parsed["received_at"] = datetime.now(UTC).isoformat()
            return parsed
        except json.JSONDecodeError:
            return {"event": event_type, "raw_data": data, "received_at": datetime.now(UTC).isoformat()}

    def _print_event(self, event: dict[str, Any]):
        """Print an event to console with formatting."""
        event_type = event.get("event", "unknown")
        timestamp = event.get("received_at", "")[:19]

        if event_type == "connected":
            print(f"  🔗 [{timestamp}] Connected to stream")
        elif event_type == "progress":
            step = event.get("step", "working")
            progress = event.get("progress", 0)
            message = event.get("message", "")
            bar = "█" * int(progress / 10) + "░" * (10 - int(progress / 10))
            print(f"  ⏳ [{timestamp}] [{bar}] {progress}% - {step}: {message}")
        elif event_type == "job_started":
            print(f"  📨 [{timestamp}] Job started")
        elif event_type == "tool_used":
            toolkit = event.get("toolkit", "")
            tool = event.get("tool", "")
            message = event.get("message", "")
            analyzer = event.get("analyzer")
            if toolkit == "sentiment_toolkit" and analyzer:
                label = "FinBERT" if analyzer == "finbert" else "sentiment (no-op)"
                print(f"  🔧 [{timestamp}] {toolkit}.{tool} ({label}): {message}")
            else:
                print(f"  🔧 [{timestamp}] {toolkit}.{tool}: {message}")
        elif event_type == "job_completed":
            print(f"  ✅ [{timestamp}] Job completed!")
        elif event_type == "job_failed":
            error = event.get("error", "Unknown error")
            print(f"  ❌ [{timestamp}] Job failed: {error}")
        else:
            print(f"  📨 [{timestamp}] {event_type}: {json.dumps(event, indent=2)[:200]}")

    def generate_report(
        self,
        ticker: str,
        years: list[int],
        job_id: str,
        final_result: dict[str, Any],
        job_status: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Generate a comprehensive JSON report for evaluation."""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        report = {
            "metadata": {
                "test_id": f"risk_analysis_{ticker}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                "ticker": ticker,
                "years": years,
                "base_url": self.base_url,
                "job_id": job_id,
                "started_at": self.start_time.isoformat() if self.start_time else None,
                "completed_at": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": duration,
                "total_events": len(self.events)
            },
            "request": {
                "endpoint": "/api/v1/analyze",
                "method": "POST",
                "payload": {
                    "ticker": ticker,
                    "years": years,
                    "analysis_type": "risk_drift",
                    "options": {
                        "include_sentiment": True,
                        "generate_heatmap": True,
                        "depth": "deep"
                    }
                }
            },
            "stream": {
                "endpoint": f"/api/v1/stream/{job_id}",
                "events": self.events
            },
            "final_result": final_result,
            "job_status": job_status,
            "evaluation": {
                "success": final_result.get("event") == "job_completed",
                "has_result": bool(final_result.get("result") or (job_status and job_status.get("result_summary"))),
                "event_count": len(self.events),
                "duration_seconds": duration
            }
        }

        # Extract key findings for quick evaluation
        result_summary = job_status.get("result_summary") if job_status else final_result.get("result")
        if result_summary:
            report["analysis_output"] = result_summary

        return report


async def run_test(
    base_url: str,
    ticker: str,
    years: list[int],
    output_file: str | None
) -> int:
    """Run the full risk analysis test."""
    client = RiskAnalysisClient(base_url)

    print("=" * 60)
    print(f"🔬 InSights AI - Risk Analysis Test")
    print("=" * 60)
    print(f"  Ticker:   {ticker}")
    print(f"  Years:    {years}")
    print(f"  Base URL: {base_url}")
    print("=" * 60)

    # Step 1: Health check
    print("\n📋 Step 1: Checking API health...")
    if not await client.check_health():
        print("❌ API is not healthy. Make sure the server is running.")
        print(f"   Try: cd python && uv run uvicorn insights.api.main:app --reload")
        return 1
    print("✅ API is healthy")

    # Step 2: Submit analysis
    print(f"\n📋 Step 2: Submitting analysis for {ticker}...")
    job_id = await client.submit_analysis(ticker, years)
    if not job_id:
        print("❌ Failed to submit analysis job")
        return 1
    print(f"✅ Job submitted: {job_id}")

    # Step 3: Stream progress
    print("\n📋 Step 3: Streaming progress...")
    final_result = await client.stream_progress(job_id)

    # Step 4: Get final job status
    print("\n📋 Step 4: Fetching final job status...")
    job_status = await client.get_job_status(job_id)
    if job_status:
        print(f"✅ Job status: {job_status.get('status', 'unknown')}")
    else:
        print("⚠️  Could not fetch job status")

    # Step 5: Generate report
    print("\n📋 Step 5: Generating comprehensive report...")
    report = client.generate_report(ticker, years, job_id, final_result, job_status)

    # Output report
    report_json = json.dumps(report, indent=2, default=str)

    if output_file:
        output_path = Path(output_file)
        output_path.write_text(report_json)
        print(f"✅ Report saved to: {output_path.absolute()}")
    else:
        # Print to stdout
        print("\n" + "=" * 60)
        print("📄 COMPREHENSIVE JSON REPORT")
        print("=" * 60)
        print(report_json)

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    eval_data = report.get("evaluation", {})
    print(f"  Success:        {'✅ Yes' if eval_data.get('success') else '❌ No'}")
    print(f"  Has Result:     {'✅ Yes' if eval_data.get('has_result') else '❌ No'}")
    print(f"  Event Count:    {eval_data.get('event_count', 0)}")
    print(f"  Duration:       {eval_data.get('duration_seconds', 'N/A')} seconds")
    print("=" * 60)

    return 0 if eval_data.get("success") else 1


def main():
    parser = argparse.ArgumentParser(
        description="Test NVDA Risk Analysis via InSights AI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic test (uses defaults: NVDA, 2024 vs 2023)
    python scripts/test_nvda_risk_analysis.py

    # Custom years
    python scripts/test_nvda_risk_analysis.py --years 2025 2024

    # Save output to file
    python scripts/test_nvda_risk_analysis.py --output results/nvda_analysis.json

    # Custom API endpoint
    python scripts/test_nvda_risk_analysis.py --base-url http://localhost:8080
        """
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("INSIGHTS_API_URL", DEFAULT_BASE_URL),
        help=f"API base URL (default: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help=f"Stock ticker to analyze (default: {DEFAULT_TICKER})"
    )
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        default=DEFAULT_YEARS,
        metavar=("CURRENT", "PREVIOUS"),
        help=f"Years to compare (default: {DEFAULT_YEARS[0]} {DEFAULT_YEARS[1]})"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for JSON report (default: print to stdout)"
    )

    args = parser.parse_args()

    # Run the async test
    exit_code = asyncio.run(run_test(
        base_url=args.base_url,
        ticker=args.ticker,
        years=args.years,
        output_file=args.output
    ))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
