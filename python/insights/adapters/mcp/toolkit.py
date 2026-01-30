"""
Agno Toolkit wrappers for MCP tools.
These make MCP tools available to Agno agents.
"""
import json
from typing import List, Optional, Dict, Any, cast

from agno.tools import Toolkit
from pydantic import BaseModel, Field

from insights.core.config import settings
from insights.core.events import publish_tool_used
from .client import MCPClient, MCPConfig, get_mcp_client


class FilingResult(BaseModel):
    """Structured result from SEC filing retrieval."""
    ticker: str
    accession_number: str
    form_type: str
    filing_date: str
    content: str
    sections: dict = Field(default_factory=dict)


class SECToolkit(Toolkit):
    """
    SEC EDGAR data access toolkit for Agno agents.
    """
    
    def __init__(self, mcp_client: Optional[MCPClient] = None):
        super().__init__(name="sec_toolkit")
        self.client = mcp_client or get_mcp_client()
        
        self.register(self.get_filing)
        self.register(self.get_filing_section)
        self.register(self.search_filings)
        self.register(self.get_company_info)
    
    async def get_filing(
        self, 
        ticker: str, 
        form_type: str = "10-K",
        year: Optional[int] = None
    ) -> str:
        """Retrieve the full text of an SEC filing."""
        publish_tool_used("sec_toolkit", "get_filing", message=f"Fetching full {form_type} for {ticker}", ticker=ticker, form_type=form_type, year=year)
        args = {
            "identifier": ticker,
            "accession_number": await self._get_latest_accession(ticker, form_type, year)
        }
        result = await self.client.call_tool("get_filing_content", args)
        return str(result)
    
    async def get_filing_section(
        self,
        ticker: str,
        section: str,
        form_type: str = "10-K",
        year: Optional[int] = None
    ) -> str:
        """Retrieve a specific section from an SEC filing."""
        publish_tool_used("sec_toolkit", "get_filing_section", message=f"Fetching section {section} from {form_type} for {ticker}", ticker=ticker, section=section, form_type=form_type, year=year)
        accession = await self._get_latest_accession(ticker, form_type, year)
        if not accession:
            return f"No filing found for {ticker} {form_type} ({year or 'latest'})"
        
        result_str = str(await self.client.call_tool("get_filing_sections", {
            "identifier": ticker,
            "accession_number": accession,
            "form_type": form_type
        }))
        
        try:
            data = json.loads(result_str)
        except json.JSONDecodeError:
            return f"Error parsing sections response: {result_str[:100]}"
        
        # MCP may return { "success", "form_type", "sections": {...}, "available_sections": [...] }
        sections = data.get("sections", data) if isinstance(data, dict) else data
        if not isinstance(sections, dict):
            return "Invalid response format"
            
        section_key = section.lower()
        # MCP sec-edgar may use "risk_factors" for Item 1A
        alias_map = {"1a": "risk_factors", "risk_factors": "risk_factors"}
        keys_to_try = [
            section_key,
            f"item_{section_key}",
            section_key.replace("item_", ""),
            alias_map.get(section_key),
        ]
        for key in keys_to_try:
            if key and key in sections:
                return str(sections[key])
        available = data.get("available_sections", list(sections.keys()))
        return f"Section '{section}' not found. Available sections: {available}"
    
    async def search_filings(
        self,
        ticker: str,
        form_types: Optional[List[str]] = None,
        days: int = 365,
        limit: int = 10
    ) -> str:
        """Search for recent filings."""
        publish_tool_used("sec_toolkit", "search_filings", message=f"Searching filings for {ticker}", ticker=ticker, limit=limit)
        args = {
            "identifier": ticker,
            "days": days,
            "limit": limit
        }
        if form_types:
            args["form_type"] = form_types[0] 
        
        return str(await self.client.call_tool("get_recent_filings", args))
    
    async def get_company_info(self, ticker: str) -> str:
        """Get company information."""
        publish_tool_used("sec_toolkit", "get_company_info", message=f"Fetching company info for {ticker}", ticker=ticker)
        return str(await self.client.call_tool("get_company_info", {
            "identifier": ticker
        }))
    
    async def _get_latest_accession(
        self, 
        ticker: str, 
        form_type: str,
        year: Optional[int] = None
    ) -> str:
        """Helper to get accession number for a filing."""
        result_str = str(await self.client.call_tool("get_recent_filings", {
            "identifier": ticker,
            "form_type": form_type,
            "limit": 20
        }))
        
        try:
            data = json.loads(result_str)
            if isinstance(data, list):
                filings = data
            elif isinstance(data, dict) and "filings" in data:
                filings = data["filings"]
            else:
                return ""
            
            if not isinstance(filings, list):
                return ""
                
            for filing in filings:
                if not isinstance(filing, dict):
                    continue
                if year:
                    f_date = str(filing.get("filing_date", ""))
                    if f_date.startswith(str(year)):
                        return str(filing.get("accession_number", ""))
                else:
                    return str(filing.get("accession_number", ""))
            return ""
        except json.JSONDecodeError:
            return ""


class SearchToolkit(Toolkit):
    """
    Web search toolkit for supplementary data.
    Uses Brave Search MCP.
    """
    
    def __init__(self, mcp_client: Optional[MCPClient] = None):
        super().__init__(name="search_toolkit")
        
        if mcp_client:
            self.client = mcp_client
        else:
            # Brave uses different URL
            config = MCPConfig(server_url=settings.MCP_BRAVE_URL)
            self.client = MCPClient(config)
            
        self.register(self.web_search)
    
    async def web_search(
        self,
        query: str,
        count: int = 10
    ) -> str:
        """
        Search the web for current information.
        
        Args:
            query: Search query
            count: Number of results
            
        Returns:
            JSON string with search results
        """
        return str(await self.client.call_tool("brave_web_search", {
            "q": query,
            "count": count
        }))
