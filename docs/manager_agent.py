"""
Manager Agent - Strategic Orchestrator
Coordinates the Secretary and Analyst agents, manages state in Supabase.
"""

from agno.agent import Agent
from agno.models.google import Gemini
from supabase import create_client, Client
from typing import Optional
from datetime import datetime, UTC
import asyncio
import json
import uuid

from src.config import settings, models_config
from src.agents.secretary_agent import SecretaryAgent
from src.agents.analyst_agent import AnalystAgent
import logging

logger = logging.getLogger(__name__)


class ManagerAgent:
    """
    The Manager Agent is the CEO of the research squad.
    
    Mission: Coordinate the Secretary and Analyst, manage research session
    state.
    """
    
    def __init__(self):
        # Initialize Supabase client
        from supabase import ClientOptions
        self.supabase: Client = create_client(
            settings.supabase_url,
            settings.supabase_service_key,
            options=ClientOptions(postgrest_client_timeout=60)
        )
        
        # Initialize sub-agents
        self.secretary = SecretaryAgent()
        self.analyst = AnalystAgent()
        
        # Configure Agno Agent
        self.agent = Agent(
            model=Gemini(
                id=models_config.manager.id,
                api_key=settings.gemini_api_key,
                **models_config.manager.params.model_dump()
            ),
            description="Strategic Research Orchestrator",
            instructions="""You are the CEO of the SEC-Insight-Agent research squad.

## CRITICAL PROTOCOL FOR RISK DRIFT:
1. When fetching filings for comparison, you MUST fetch TWO DISTINCT filings.
2. Use the 'get_filing_list' tool first to see available dates.
3. Then call 'get_filing_section' TWICE:
   - Once with filing_index=0 (Current)
   - Once with filing_index=1 (Previous)
4. DO NOT compare a filing to itself. Check the dates/accession numbers.

## Your Role:
- Coordinate the Secretary (data) and Analyst (reasoning) agents
- Manage user session state
- Persist all findings to the database

## Decision Making:
1. When user asks about a company -> Task Secretary to verify and fetch filings
2. When filings are ready -> Task Analyst to perform analysis
3. When analysis complete -> Store in database and format report
""",
            tools=[self.analyze_risk_drift, self.get_company_history]
        )
    
    async def analyze_risk_drift(
        self,
        ticker: str,
        item_number: str = "1A",
        previous_year: Optional[str] = None,
        current_year: Optional[str] = None,
        overwrite: bool = False
    ) -> dict:
        """
        Perform a complete risk drift analysis for a company.
        
        Orchestrates Secretary and Analyst to compare current vs previous
        year's risk factors.
        """
        # Step 1: Verify company
        verification = await self.secretary.verify_company(ticker)
        if not verification.get("verified"):
            return {"error": f"Could not verify company: {ticker}"}
        
        # Step 2: Get current and previous year's filings
        # If specific years provided, fetch them. Otherwise rely on index 0 (current) and 1 (prev).
        current_section = await self.secretary.get_filing_section(
            ticker=ticker,
            filing_type="10-K",
            item_number=item_number,
            filing_index=0,
            filing_year=current_year
        )
        
        previous_section = await self.secretary.get_filing_section(
            ticker=ticker,
            filing_type="10-K",
            item_number=item_number,
            filing_index=1 if not previous_year else 0, # If explicit year, index is relative to that year (so 0)
            filing_year=previous_year
        )
        
        if not current_section.get("success") or not previous_section.get("success"):
            return {"error": "Could not retrieve filings for comparison"}
        
        # Step 3: Perform semantic drift analysis
        # Using Analyst's Agno functionality
        # Step 3: Perform semantic drift analysis
        # Using Analyst's Agno functionality
        # Step 3: Perform semantic drift analysis
        analysis_result = await self.analyst.generate_risk_report(
            text_curr=current_section["content"],
            text_prev=previous_section["content"],
            date_curr=current_section.get("filing_date"),
            date_prev=previous_section.get("filing_date")
        )
        
        if "error" in analysis_result:
            return {"error": analysis_result["error"]}

        risk_report = analysis_result["report"]
        risks_curr = analysis_result["risks_curr"]
        risks_prev = analysis_result["risks_prev"]

        # Step 4: Sync Filings and Embeddings to Supabase
        await asyncio.gather(
            self._sync_filing_to_db(ticker, current_section, risks_curr),
            self._sync_filing_to_db(ticker, previous_section, risks_prev)
        )

        # Step 5: Save and return
        result = {
            "ticker": ticker,
            "analysis_type": "risk_drift",
            "section": f"Item {item_number}",
            "current_filing": {
                "source_url": current_section["source_url"],
                "word_count": current_section["word_count"]
            },
            "previous_filing": {
                "source_url": previous_section["source_url"],
                "word_count": previous_section["word_count"]
            },
            "report": risk_report, # Structured data
            "generated_at": datetime.now(UTC).isoformat()
        }
        
        
        # Extract years from filing dates or provided metadata
        def extract_year(date_str):
            if not date_str: return None
            return date_str.split("-")[0]

        current_year = extract_year(current_section.get("filing_date"))
        previous_year = extract_year(previous_section.get("filing_date"))

        # Enrich Heatmap Title
        if "heatmap" in risk_report:
            company_name = verification.get("company_name", ticker)
            risk_report["heatmap"]["heatmap_title"] = f"{company_name} ({ticker}) - Risk Profile Evolution ({previous_year} vs {current_year})"

        # Enrich report with metadata for full_analysis JSON
        # This preserves the original report structure while adding top-level metadata
        full_analysis_data = {
            "company_name": verification.get("company_name"),
            "cik": verification.get("cik"),
            "current_year": current_year,
            "previous_year": previous_year,
            "filing_type": "10-K",
            **risk_report # Unpack original report
        }
        
        # Serialize to JSON
        full_analysis_json = json.dumps(full_analysis_data)
        
        # filing_year from filing_date (e.g., "2025-01-26" -> "2025")
        filing_year = risk_report['meta']['filing_date'].split("-")[0] if risk_report['meta'].get('filing_date') else None
        
        # summary_text
        summary_text = f"Analyzed {risk_report['meta']['total_risks_current']} risks. Detected {risk_report['meta']['drift_count']} material drifts."
        
        # drift_score
        drift_score = risk_report['meta']['drift_count'] / max(1, risk_report['meta']['total_risks_current'])
        
        # Extract materiality summary from flags
        # Extract materiality summary from flags
        flags = risk_report.get("materiality_flags", [])
        
        # Compute overall materiality level for DB ENUM (LOW, MEDIUM, HIGH)
        # Logic: If any flag is High Intensity -> HIGH, else MEDIUM if structural/semantic, etc.
        # Simple heuristic:
        materiality_enum = "LOW"
        if flags:
            priorities = [f.get("intensity", "LOW").upper() for f in flags]
            if "HIGH" in priorities:
                materiality_enum = "HIGH"
            elif "MEDIUM" in priorities:
                materiality_enum = "MEDIUM"
            else:
                materiality_enum = "LOW" # Default if flags exist but marked low?
                
            # Allow "Critical shifts" text to be appended to summary or just kept in full_analysis
            materiality_text = f"Critical shifts in: {', '.join([f.get('risk')[:60] + '...' for f in flags[:3]])}"
            if len(flags) > 3:
                materiality_text += f" (+{len(flags)-3} more)"
            
            # append valid text to summary if not already there
            summary_text += f" {materiality_text}"
        else:
            materiality_enum = "LOW"
        
        # Get snippets from the first material drift if available
        first_drift = risk_report.get("sentiment_shift_indicators", [])[0] if risk_report.get("sentiment_shift_indicators") else None
        orig_snippet = first_drift.get("original_snippet") if first_drift else None
        new_snippet = first_drift.get("new_snippet") if first_drift else None
        conf_score = first_drift.get("confidence") if first_drift else 0.9

        # Save to DB
        await self.save_analysis(
            ticker=ticker,
            analysis_type="risk_drift",
            summary=summary_text,
            drift_score=drift_score, 
            source_url=current_section["source_url"],
            full_analysis=full_analysis_json,
            confidence_score=conf_score,
            materiality=materiality_enum,
            original_text_snippet=orig_snippet,
            new_text_snippet=new_snippet,
            filing_year=filing_year,
            risk_category="Item 1A - Risk Factors"
        )
        
        return result
    
    def get_company_history(self, ticker: str) -> dict:
        """
        Get the analysis history for a company from the database.
        """
        try:
            result = self.supabase.table("risk_drifts")\
                .select("*")\
                .eq("company_ticker", ticker)\
                .order("created_at", desc=True)\
                .limit(10)\
                .execute()
            
            return {
                "ticker": ticker,
                "history": result.data,
                "count": len(result.data)
            }
        except Exception as e:
            return {"error": str(e), "history": []}

    def get_latest_analysis(self, ticker: str) -> Optional[dict]:
        """
        Check if a recent analysis already exists for this ticker.
        """
        try:
            result = self.supabase.table("risk_drifts")\
                .select("*")\
                .eq("company_ticker", ticker)\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            
            return result.data[0] if result.data else None
        except Exception:
            return None
    
    async def _sync_filing_to_db(self, ticker: str, section_data: dict, risks: list) -> Optional[str]:
        """
        Ensures company, filing, and embeddings exist in the database.
        """
        try:
            # 1. Ensure Company Exists
            cik = section_data.get("cik")
            if not cik:
                return None
                
            def _upsert_company():
                return self.supabase.table("companies").upsert({
                    "cik": cik,
                    "ticker": ticker.upper(),
                    "name": ticker.upper() # Fallback name
                }).execute()
            
            await asyncio.to_thread(_upsert_company)

            # 2. Ensure Filing Exists
            accession = section_data.get("accession_number")
            if not accession:
                return None
                
            # Infer fiscal year from filing date if not present
            f_date = section_data.get("filing_date")
            f_year = section_data.get("fiscal_year")
            if not f_year and f_date:
                # 10-K is usually filed in the year following the fiscal year end, or same year.
                # Heuristic: If filed Jan-Mar, fiscal year is Year-1.
                # Actually, strictly capturing "fiscal_year" is hard without data.
                # But we can at least capture details.
                try:
                    dt = datetime.strptime(f_date, "%Y-%m-%d")
                    # Store calendar year as fallback
                    f_year = dt.year 
                except:
                    pass

            filing_record = {
                "accession_number": accession,
                "cik": cik,
                "filing_type": section_data["metadata"]["filing_type"],
                "filing_date": f_date,
                "fiscal_year": f_year,
                "fiscal_quarter":  None if section_data["metadata"]["filing_type"] == "10-K" else 4, # Placeholder
                "edgar_url": section_data.get("source_url"),
                "word_count": section_data.get("word_count")
            }
            
            def _upsert_filing():
                return self.supabase.table("filings").upsert(filing_record, on_conflict="accession_number").execute()
                
            filing_res = await asyncio.to_thread(_upsert_filing)
            filing_id = filing_res.data[0]["id"] if filing_res.data else None
            
            if not filing_id:
                # Fallback: get by accession if upsert didn't return (though it should)
                def _get_filing():
                    return self.supabase.table("filings").select("id").eq("accession_number", accession).execute()
                filing_res = await asyncio.to_thread(_get_filing)
                filing_id = filing_res.data[0]["id"] if filing_res.data else None

            if not filing_id:
                return None

            # 3. Save Embeddings
            # First, check if embeddings already exist for this filing
            def _check_embeddings():
                return self.supabase.table("filing_embeddings").select("id", count="exact").eq("filing_id", filing_id).execute()
            
            count_res = await asyncio.to_thread(_check_embeddings)
            if count_res.count and count_res.count > 0:
                # logger.info(f"Embeddings already exist for filing {accession}")
                return filing_id

            # Prepare embedding records
            embedding_records = []
            for risk in risks:
                if not risk.embedding:
                    continue
                    
                embedding_records.append({
                    "filing_id": filing_id,
                    "section_type": "Item 1A",
                    "chunk_index": risk.rank,
                    "content_preview": risk.title,
                    "embedding": risk.embedding
                })
            
            if embedding_records:
                # Insert in chunks to avoid timeouts/limits
                chunk_size = 10
                for i in range(0, len(embedding_records), chunk_size):
                    chunk = embedding_records[i : i + chunk_size]
                    def _insert_embeddings():
                        return self.supabase.table("filing_embeddings").insert(chunk).execute()
                    await asyncio.to_thread(_insert_embeddings)
                logger.info(f"Saved {len(embedding_records)} embeddings for filing {accession}")
            
            return filing_id
            
        except Exception as e:
            logger.error(f"Failed to sync filing/embeddings to DB: {e}")
            return None

    async def save_analysis(
        self,
        ticker: str,
        analysis_type: str,
        summary: str,
        drift_score: float,
        source_url: str,
        full_analysis: str,
        confidence_score: Optional[float] = None,
        materiality: Optional[str] = None,
        original_text_snippet: Optional[str] = None,
        new_text_snippet: Optional[str] = None,
        filing_year: Optional[str] = None,
        risk_category: Optional[str] = None,
        overwrite: bool = False
    ) -> dict:
        """
        Save an analysis result to the database.
        """
        try:
            # Overwrite Logic
            if overwrite and filing_year:
                def _delete_existing():
                     return self.supabase.table("risk_drifts").delete()\
                         .eq("company_ticker", ticker)\
                         .eq("filing_year", filing_year)\
                         .execute()
                await asyncio.to_thread(_delete_existing)

            record = {
                "id": str(uuid.uuid4()),
                "company_ticker": ticker,
                "analysis_type": analysis_type,
                "summary": summary,
                "drift_score": drift_score,
                "source_url": source_url,
                "full_analysis": full_analysis,
                "created_at": datetime.now(UTC).isoformat(),
                # V2 Columns
                "confidence_score": confidence_score or 0.9,
                "materiality": materiality or "LOW",
                "original_text_snippet": original_text_snippet,
                "new_text_snippet": new_text_snippet,
                "filing_year": filing_year,
                "risk_category": risk_category
            }
            
            # Run blocking Supabase call in a separate thread
            def _insert():
                return self.supabase.table("risk_drifts").insert(record).execute()
                
            result = await asyncio.to_thread(_insert) # Wrapped blocking call
            
            return {
                "success": True,
                "record_id": record["id"]
            }
        except Exception as e:
            print(f"Failed to save analysis: {e}")
            if hasattr(e, 'message'):
                 print(f"Error message: {e.message}")
            return {"success": False, "error": str(e)}
