"""
Secretary Agent - Data Ingestion Specialist
Responsible for fetching and parsing SEC filings from EDGAR.
"""

import logging
import asyncio
from typing import Optional, Any

from agno.agent import Agent
from agno.models.google import Gemini
from mcp import ClientSession, StdioServerParameters
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

from src.tools.sec_filing_tool import SecFilingTool
from src.config import settings, models_config

logger = logging.getLogger(__name__)

class SecretaryAgent:
    """
    The Secretary Agent is responsible for data acquisition.
    
    Mission: Fetch, clean, and structurally parse SEC filings from EDGAR.
    """
    
    def __init__(self):
        self.sec_tool = SecFilingTool()
        self.mcp_enabled = settings.sec_edgar_mcp_enabled
        
        # Configure Agno Agent
        # Note: SecretaryAgent is mostly tool-based, but we wrap it in Agno
        # to allow for future reasoning about extraction strategies.
        self.agent = Agent(
            model=Gemini(
                id=models_config.secretary.id,
                api_key=settings.gemini_api_key,
                **models_config.secretary.params.model_dump()
            ),
            description="SEC Filing Data Acquisition Specialist",
            instructions="""You are the Data Acquisition Specialist.
Your job is to fetch and return clean SEC filing data.
Always verify the CIK before fetching.
""",
            # Note: We expose the async methods, but Agno might need an adapter if used directly by LLM.
            # Since ManagerAgent calls these directly as python methods, we are good.
            tools=[self.verify_company, self.get_filing_section, self.get_filing_list]
        )
    
    async def _call_mcp(self, tool_name: str, arguments: dict) -> Any:
        """Helper to call MCP tools via stdio client."""
        if not self.mcp_enabled:
            return None
            
        try:
            if settings.sec_edgar_mcp_mode == "sse":
                # Connect via SSE (HTTP)
                async with sse_client(settings.sec_edgar_mcp_sse_url) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments)
                        return result
            else:
                # Default: Connect via Stdio (Docker/Process)
                command = settings.sec_edgar_mcp_command
                args = settings.get_sec_edgar_mcp_args
                
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=None
                )
                
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments)
                        return result

        except Exception as e:
            logger.warning(f"MCP tool call failed ({tool_name}) using {settings.sec_edgar_mcp_mode}: {e}")
            return None

    async def verify_company(self, ticker: str) -> dict:
        """
        Verify a company ticker and return its CIK.
        
        Safety Gate: Always use this before fetching filings.
        """
        # Try MCP first
        if self.mcp_enabled:
            mcp_result = await self._call_mcp("get_cik_by_ticker", {"ticker": ticker})
            if mcp_result and not mcp_result.isError:
                # Expecting dictionary with cik
                data = mcp_result.content[0].text if mcp_result.content else "{}"
                # Parse if it's JSON string, or if mcp returns object
                # Usually mcp returns TextContent.
                import json
                try:
                    # The tool likely returns a JSON string or dict in text
                    # We need to handle this based on actual tool output.
                    # Assuming the tool returns a JSON string in content[0].text
                    parsed = json.loads(data)
                    cik = parsed.get("cik")
                    if cik:
                         return {
                            "verified": True,
                            "ticker": ticker.upper(),
                            "cik": cik,
                            "edgar_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
                        }
                except Exception:
                    pass

        # Fallback to local tool
        company_info = self.sec_tool.get_company_info(ticker)
        if company_info:
            cik = company_info["cik"]
            return {
                "verified": True,
                "ticker": ticker.upper(),
                "cik": cik,
                "company_name": company_info["name"],
                "edgar_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
            }
        return {
            "verified": False,
            "ticker": ticker,
            "error": f"Could not find CIK for ticker: {ticker}"
        }
    
    async def get_filing_list(
        self, 
        ticker: str, 
        filing_type: str,
        limit: int = 5
    ) -> list[dict]:
        """
        Get list of available SEC filings for a company.
        """
        # Try MCP first
        if self.mcp_enabled:
            # Resolve CIK locally to ensure MCP doesn't fail on ticker lookup
            cik = self.sec_tool.get_cik_for_ticker(ticker)
            identifier = cik if cik else ticker
            
            # Map parameters
            args = {
                "identifier": identifier,
                "form_type": filing_type,
                "limit": limit
            }
            mcp_result = await self._call_mcp("get_recent_filings", args)
            if mcp_result and not mcp_result.isError:
                import json
                try:
                    data = mcp_result.content[0].text
                    parsed = json.loads(data)
                    if isinstance(parsed, list):
                        # The MCP tool returns a list of filings, we might need to map keys
                        # Expected local format: ticker, cik, filing_type, filing_date, accession_number, primary_document
                        mapped_filings = []
                        for f in parsed:
                            mapped_filings.append({
                                "ticker": ticker,
                                "cik": f.get("cik", ""),
                                "filing_type": f.get("form_type", filing_type),
                                "filing_date": f.get("filing_date", ""),
                                "accession_number": f.get("accession_number", ""),
                                "primary_document": f.get("primary_document", ""),
                                "fiscal_year": f.get("fiscal_year"),
                                "fiscal_period": f.get("fiscal_period")
                            })
                        return mapped_filings
                except Exception:
                   pass

        # Fallback
        return self.sec_tool.fetch_filing_list(ticker, filing_type, limit)
    
    async def get_filing_section(
        self,
        ticker: str,
        filing_type: str,
        item_number: str,
        filing_index: int = 0,
        filing_year: Optional[str] = None
    ) -> dict:
        """
        Extract a specific section from an SEC filing.
        
        Returns the COMPLETE section text (no chunking).
        """
        # Note: MCP 'get_filing_sections' typically returns ALL sections or expects an accession number.
        # To maintain the specific 'filing_index' logic (0 = most recent), we first need the accession number.
        
        accession_number = None
        cik = None
        filing_date = None
        target_filing = None
        
        # Get list first if we don't have accession
        # If filing_year is requested, fetch a larger list to ensure we cover the year
        limit = filing_index + 1 if not filing_year else 20
        filings = await self.get_filing_list(ticker, filing_type, limit=limit)
        
        if filing_year:
            # Filter by year
            year_filings = [f for f in filings if f["filing_date"].startswith(str(filing_year))]
            if len(year_filings) > filing_index:
                target_filing = year_filings[filing_index]
            else:
                return {
                    "success": False,
                    "error": f"Could not find {filing_type} filings for {ticker} in {filing_year}"
                }
        elif filings and len(filings) > filing_index:
            target_filing = filings[filing_index]
            
        if target_filing:
            accession_number = target_filing["accession_number"]
            cik = target_filing["cik"]
            filing_date = target_filing["filing_date"]
        else:
             return {
                "success": False,
                "error": f"Could not find {filing_type} filings for {ticker}"
            }

        # Try MCP to get sections
        if self.mcp_enabled and accession_number:
            # Resolve CIK locally
            cik_lookup = self.sec_tool.get_cik_for_ticker(ticker)
            identifier = cik_lookup if cik_lookup else ticker
            
            args = {
                "identifier": identifier,
                "accession_number": accession_number,
                "form_type": filing_type
            }
            # Try 'get_filing_sections' from MCP
            mcp_result = await self._call_mcp("get_filing_sections", args)
            if mcp_result and not mcp_result.isError:
                 import json
                 try:
                    data = mcp_result.content[0].text
                    sections = json.loads(data)
                    # Look for the specific item
                    # section keys might be "Item 1A" or "1A"
                    # We normalize keys to match item_number
                    
                    target_content = None
                    target_title = f"Item {item_number}"
                    
                    # Normalize search
                    normalized_item = item_number.lower().replace("item ", "").strip()
                    
                    for key, content in sections.items():
                        if normalized_item in key.lower():
                            target_content = content
                            target_title = key
                            break
                    
                    if target_content:
                        return {
                            "success": True,
                            "item_number": item_number,
                            "title": target_title,
                            "content": target_content,
                            "source_url": f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_number.replace('-', '')}/{target_filing['primary_document']}",
                            "word_count": len(target_content.split()),
                            "filing_date": filing_date,
                            "accession_number": accession_number,
                            "cik": cik,
                            "fiscal_year": target_filing.get("fiscal_year"),
                            "fiscal_period": target_filing.get("fiscal_period"),
                            "metadata": {
                                "ticker": ticker,
                                "filing_type": filing_type,
                                "filing_index": filing_index
                            }
                            }

                 except Exception as e:
                     logger.warning(f"MCP section extraction failed: {e}")

        # Fallback to local tool
        if accession_number and target_filing:
            # We identified the specific filing (e.g. by year), so we must fetch it directly
            # by accession to avoid the tool reverting to the default index.
            try:
                cik_stripped = cik.lstrip("0")
                html_content = self.sec_tool.fetch_filing_content(
                    cik_stripped, 
                    accession_number, 
                    target_filing["primary_document"]
                )
                
                accession_formatted = accession_number.replace("-", "")
                source_url_constructed = f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{accession_formatted}/{target_filing['primary_document']}"

                section = self.sec_tool.extract_section(
                    html_content, 
                    item_number, 
                    source_url_constructed,
                    filing_date=filing_date,
                    accession_number=accession_number,
                    cik=cik,
                    fiscal_year=target_filing.get("fiscal_year"),
                    fiscal_period=target_filing.get("fiscal_period")
                )
            except Exception as e:
                logger.error(f"Direct fallback fetch failed: {e}")
                section = None
        else:
            # Default to index-based fetch if we didn't identify a target
            section = self.sec_tool.get_filing_section(
                ticker, 
                filing_type, 
                item_number, 
                filing_index
            )
        
        if section:
            return {
                "success": True,
                "item_number": section.item_number,
                "title": section.title,
                "content": section.content,
                "source_url": section.source_url,
                "word_count": section.word_count,
                "filing_date": section.filing_date,
                "accession_number": section.accession_number,
                "cik": section.cik,
                "fiscal_year": section.fiscal_year,
                "fiscal_period": section.fiscal_period,
                "metadata": {
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "filing_index": filing_index
                }
            }
        
        return {
            "success": False,
            "error": f"Could not extract Item {item_number} from {ticker} {filing_type}"
        }
