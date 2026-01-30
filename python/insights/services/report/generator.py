"""
Report Generator - Formats analysis results into Markdown reports.
"""
from datetime import UTC, datetime
from typing import Any

# Using Any for input types to avoid circular imports or strict coupling
# if core.types aren't fully populated yet, but ideally we use core.types.
from insights.core.types import Company, ExpertResult


class ReportGenerator:
    """Generates formatted reports from analysis data."""

    def generate_markdown_report(
        self,
        company: Company,
        results: list[dict[str, Any]], # List of drift result dicts
        expert_findings: list[ExpertResult],
        summary: str,
        generated_at: datetime | None = None
    ) -> str:
        """
        Generate a full Markdown report.

        Args:
            company: Company metadata
            results: List of drift results (dict or object)
            expert_findings: List of expert insights
            summary: Executive summary text
            generated_at: Timestamp

        Returns:
            Markdown string
        """
        timestamp = (generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M")

        sections = []

        # 1. Header
        sections.append(f"# Risk Assessment Report: {company.name} ({company.ticker})")
        sections.append(f"**Date:** {timestamp}")
        sections.append(f"**Sector:** {company.sector or 'N/A'}")
        sections.append("---")

        # 2. Executive Summary
        sections.append("## Executive Summary")
        sections.append(summary)
        sections.append("")

        # 3. Key Risk Changes (Top 5 by heat score)
        sections.append("## Critical Risk Factor Changes")

        # Sort by heat_score descending
        sorted_risks = sorted(
            results,
            key=lambda x: x.get('heat_score', 0),
            reverse=True
        )
        top_risks = sorted_risks[:5]

        if top_risks:
            for risk in top_risks:
                title = risk.get('risk_title') or risk.get('new_title')
                zone = risk.get('zone', 'unknown')
                delta = risk.get('rank_delta', 0)
                drift = risk.get('drift_type', 'unknown')

                icon = "🔴" if zone == "critical_red" else "🟠" if zone == "warning_orange" else "🔵" if zone == "new_blue" else "⚪"

                sections.append(f"### {icon} {title}")
                sections.append(f"- **Status:** {drift.replace('_', ' ').title()}")
                sections.append(f"- **Rank Change:** {delta:+d}")
                sections.append(f"- **Zone:** {zone}")
                sections.append("")
        else:
            sections.append("*No critical risk changes identified.*")

        sections.append("")

        # 4. Expert Findings
        if expert_findings:
            sections.append("## Expert Insights")
            for expert in expert_findings:
                sections.append(f"### {expert.expert_id.replace('_', ' ').title()}")
                sections.append(expert.findings)
                if expert.sources:
                    sections.append(f"**Sources:** {', '.join(expert.sources)}")
                sections.append("")

        return "\n".join(sections)
