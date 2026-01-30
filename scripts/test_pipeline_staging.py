#!/usr/bin/env python3
"""
Test Risk Drift Pipeline against live data in staging.

Usage:
    python scripts/test_pipeline_staging.py [TICKER] [CURRENT_YEAR] [PREVIOUS_YEAR]

Example:
    python scripts/test_pipeline_staging.py NVDA 2024 2023
"""
import asyncio
import sys
import json
from datetime import datetime

# Add python directory to path
sys.path.insert(0, 'python')

from insights.services.risk_drift import RiskDriftPipeline
from insights.adapters.db.manager import DBManager


async def test_pipeline(ticker: str, current_year: int, previous_year: int):
    """Run pipeline test against live data."""
    
    print("=" * 70)
    print(f"RISK DRIFT PIPELINE - STAGING TEST")
    print("=" * 70)
    print(f"Ticker: {ticker}")
    print(f"Years: {current_year} vs {previous_year}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    try:
        # Initialize services
        print("📦 Initializing pipeline...")
        db_manager = DBManager()
        pipeline = RiskDriftPipeline(db_manager=db_manager)
        
        # Run pipeline
        print(f"🚀 Running pipeline for {ticker}...")
        result = await pipeline.run(ticker, [current_year, previous_year])
        
        print()
        print("=" * 70)
        print("✅ PIPELINE EXECUTION SUCCESSFUL")
        print("=" * 70)
        print()
        
        # Display results
        print("📊 RESULTS SUMMARY")
        print("-" * 70)
        print(f"Company ID: {result.company_id}")
        print(f"Current Filing ID: {result.current_filing_id}")
        print(f"Previous Filing ID: {result.previous_filing_id}")
        print()
        
        # Full analysis summary
        full_analysis = result.full_analysis
        meta = full_analysis.get("meta", {})
        
        print("📈 ANALYSIS METRICS")
        print("-" * 70)
        print(f"Total Risks (Current): {meta.get('total_risks_current', 0)}")
        print(f"Total Risks (Previous): {meta.get('total_risks_prev', 0)}")
        print(f"Drift Count: {meta.get('drift_count', 0)}")
        print(f"Filing Date: {meta.get('filing_date', 'N/A')}")
        print()
        
        # Heatmap zones
        heatmap = full_analysis.get("heatmap", {})
        zones = heatmap.get("zones", {})
        
        print("🔥 HEATMAP ZONES")
        print("-" * 70)
        print(f"🔴 Critical Red: {len(zones.get('critical_red', []))} risks")
        print(f"🟠 Warning Orange: {len(zones.get('warning_orange', []))} risks")
        print(f"🔵 New Blue: {len(zones.get('new_blue', []))} risks")
        print(f"⚪ Stable Gray: {len(zones.get('stable_gray', []))} risks")
        print()
        
        # Removed risks
        removed_risks = full_analysis.get("removed_risks", [])
        print(f"❌ Removed Risks: {len(removed_risks)}")
        if removed_risks:
            for r in removed_risks[:3]:  # Show first 3
                print(f"   • {r.get('risk', 'Unknown')} (rank {r.get('rank_prev', 'N/A')})")
            if len(removed_risks) > 3:
                print(f"   ... and {len(removed_risks) - 3} more")
        print()
        
        # Visual priority map (top 5)
        priority_map = full_analysis.get("visual_priority_map", [])
        print("🎯 TOP 5 RISKS (Visual Priority Map)")
        print("-" * 70)
        for i, risk in enumerate(priority_map[:5], 1):
            status_icon = {
                "New": "🆕",
                "Climbed": "⬆️",
                "Fell": "⬇️",
                "Stable": "➡️"
            }.get(risk.get("status", "Unknown"), "❓")
            
            print(f"{i}. {status_icon} {risk.get('risk', 'Unknown')[:60]}...")
            print(f"   Rank: {risk.get('rank')} (prev: {risk.get('previous_rank', 'N/A')}), "
                  f"Status: {risk.get('status', 'Unknown')}")
        print()
        
        # Markdown report preview
        markdown = result.markdown_report
        print("📝 MARKDOWN REPORT (Preview)")
        print("-" * 70)
        print(markdown[:500] + "..." if len(markdown) > 500 else markdown)
        print()
        
        # Database verification
        print("💾 DATABASE VERIFICATION")
        print("-" * 70)
        
        # Check company
        company = await db_manager.get_company_by_ticker(ticker)
        print(f"✓ Company in DB: {company.name if company else 'NOT FOUND'}")
        
        # Check filings
        if company:
            filings = await db_manager.get_filings_by_company(
                company.id, form_type="10-K", years=[current_year, previous_year]
            )
            print(f"✓ Filings in DB: {len(filings)}")
            
            # Check risk factors
            if filings:
                for filing in filings[:2]:
                    risks = await db_manager.get_risk_factors_by_filing(filing.id)
                    print(f"✓ Risk factors ({filing.fiscal_year}): {len(risks)} "
                          f"(embeddings: {sum(1 for r in risks if r.embedding)})")
            
            # Check risk drifts
            drifts = await db_manager.get_risk_drifts_by_company(company.id)
            print(f"✓ Risk drifts in DB: {len(drifts)}")
        
        print()
        print("=" * 70)
        print("🎉 TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        # Save full analysis to file
        output_file = f"staging_test_{ticker}_{current_year}_{previous_year}.json"
        with open(output_file, 'w') as f:
            json.dump(full_analysis, f, indent=2)
        print(f"\n💾 Full analysis saved to: {output_file}")
        
        return result
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ PIPELINE EXECUTION FAILED")
        print("=" * 70)
        print(f"Error: {type(e).__name__}: {str(e)}")
        import traceback
        print()
        print("Traceback:")
        traceback.print_exc()
        return None


async def main():
    """Main entry point."""
    # Parse arguments
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    current_year = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    previous_year = int(sys.argv[3]) if len(sys.argv) > 3 else 2023
    
    result = await test_pipeline(ticker, current_year, previous_year)
    
    if result:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
