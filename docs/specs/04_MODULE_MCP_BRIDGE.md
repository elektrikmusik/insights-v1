# 04. Module: MCP Bridge (Agno Integration)

## Overview

The MCP Bridge connects InSights-ai to external data sources via the Model Context Protocol (MCP). Primary integration is with `sec-edgar-mcp` for SEC filings, with planned support for `brave-search-mcp` for news sentiment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Agno Agent                                                     │
│  ├── SECToolkit (registered)                                   │
│  └── SearchToolkit (registered)                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Toolkit Layer (insights/adapters/mcp/toolkit.py)              │
│  ├── SECToolkit.get_filing()                                   │
│  ├── SECToolkit.search_filings()                               │
│  └── SearchToolkit.web_search()                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  MCP Client (insights/adapters/mcp/client.py)                  │
│  ├── Persistent SSE/Stdio connection                           │
│  ├── Automatic reconnection                                     │
│  └── @retry decorators                                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  External MCP Servers                                           │
│  ├── sec-edgar-mcp (Docker container)                          │
│  └── brave-search-mcp (future)                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## MCP Client Implementation

### `insights/adapters/mcp/client.py`

```python
"""
MCP Client with persistent connection and automatic retry.
"""
import asyncio
import json
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from insights.core.errors import MCPConnectionError, MCPToolError

logger = logging.getLogger(__name__)


class MCPTransport(Enum):
    SSE = "sse"
    STDIO = "stdio"


@dataclass
class MCPConfig:
    server_url: str
    transport: MCPTransport = MCPTransport.SSE
    timeout: float = 30.0
    max_retries: int = 3
    user_agent: str = "InSights-ai/1.0"


class MCPClient:
    """
    Model Context Protocol client with persistent connection.
    
    Usage:
        client = MCPClient(MCPConfig(server_url="http://localhost:8080/sse"))
        await client.connect()
        result = await client.call_tool("get_filing_content", {"ticker": "AAPL"})
        await client.disconnect()
    """
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.session: Optional[httpx.AsyncClient] = None
        self._connected = False
        self._server_capabilities: Dict[str, Any] = {}
    
    async def connect(self) -> None:
        """Establish connection to MCP server."""
        if self._connected:
            return
        
        self.session = httpx.AsyncClient(
            base_url=self.config.server_url,
            timeout=self.config.timeout,
            headers={"User-Agent": self.config.user_agent}
        )
        
        try:
            # Initialize MCP handshake
            response = await self.session.post("/initialize", json={
                "protocol_version": "1.0",
                "client_info": {
                    "name": "insights-ai",
                    "version": "1.0.0"
                }
            })
            response.raise_for_status()
            
            init_result = response.json()
            self._server_capabilities = init_result.get("server_info", {})
            self._connected = True
            
            logger.info(f"Connected to MCP server: {self._server_capabilities.get('name', 'unknown')}")
            
        except httpx.HTTPError as e:
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}")
    
    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        if self.session:
            await self.session.aclose()
            self.session = None
        self._connected = False
    
    async def ensure_connected(self) -> None:
        """Ensure connection is active, reconnect if needed."""
        if not self._connected or not self.session:
            await self.connect()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        before_sleep=lambda retry_state: logger.warning(
            f"MCP call failed, retrying (attempt {retry_state.attempt_number})..."
        )
    )
    async def call_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the MCP tool (e.g., "get_filing_content")
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            MCPConnectionError: If connection fails
            MCPToolError: If tool execution fails
        """
        await self.ensure_connected()
        
        try:
            response = await self.session.post("/tools/call", json={
                "name": tool_name,
                "arguments": arguments
            })
            
            if response.status_code == 404:
                raise MCPToolError(f"Tool '{tool_name}' not found on MCP server")
            
            response.raise_for_status()
            result = response.json()
            
            # Check for tool-level errors
            if result.get("isError"):
                error_content = result.get("content", [{}])[0]
                raise MCPToolError(
                    f"Tool '{tool_name}' failed: {error_content.get('text', 'Unknown error')}"
                )
            
            # Extract text content
            content = result.get("content", [])
            if content and content[0].get("type") == "text":
                return content[0].get("text", "")
            
            return result
            
        except httpx.TimeoutException as e:
            logger.error(f"MCP call timeout for {tool_name}: {e}")
            raise MCPConnectionError(f"Timeout calling MCP tool '{tool_name}'")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"MCP HTTP error for {tool_name}: {e}")
            raise MCPToolError(f"HTTP error calling MCP tool '{tool_name}': {e.response.status_code}")
    
    async def list_tools(self) -> list:
        """Get list of available tools from MCP server."""
        await self.ensure_connected()
        
        response = await self.session.post("/tools/list", json={})
        response.raise_for_status()
        return response.json().get("tools", [])
    
    async def list_resources(self) -> list:
        """Get list of available resources from MCP server."""
        await self.ensure_connected()
        
        response = await self.session.post("/resources/list", json={})
        response.raise_for_status()
        return response.json().get("resources", [])


# Singleton instance
_mcp_client: Optional[MCPClient] = None


def get_mcp_client(config: Optional[MCPConfig] = None) -> MCPClient:
    """Get or create singleton MCP client."""
    global _mcp_client
    
    if _mcp_client is None:
        if config is None:
            from insights.core.config import settings
            config = MCPConfig(server_url=settings.MCP_SERVER_URL)
        _mcp_client = MCPClient(config)
    
    return _mcp_client
```

---

## Agno Toolkit Wrapper

### `insights/adapters/mcp/toolkit.py`

```python
"""
Agno Toolkit wrappers for MCP tools.
These are thin wrappers that make MCP tools available to Agno agents.
"""
from typing import Optional, List
from agno.tools import Toolkit
from pydantic import BaseModel, Field

from .client import MCPClient, get_mcp_client


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
    
    Exposes SEC filing retrieval and search capabilities via MCP.
    """
    
    def __init__(self, mcp_client: Optional[MCPClient] = None):
        super().__init__(name="sec_toolkit")
        self.client = mcp_client or get_mcp_client()
        
        # Register tools with Agno
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
        """
        Retrieve the full text of an SEC filing.
        
        Use this when you need to read and analyze the complete filing document.
        
        Args:
            ticker: Company stock ticker (e.g., "AAPL")
            form_type: Filing type ("10-K", "10-Q", "8-K")
            year: Fiscal year (default: most recent)
            
        Returns:
            Full text content of the filing
        """
        args = {
            "identifier": ticker,
            "form_type": form_type
        }
        if year:
            args["fiscal_year"] = year
        
        result = await self.client.call_tool("get_filing_content", args)
        return result
    
    async def get_filing_section(
        self,
        ticker: str,
        section: str,
        form_type: str = "10-K",
        year: Optional[int] = None
    ) -> str:
        """
        Retrieve a specific section from an SEC filing.
        
        Use this when you only need a particular section (e.g., Risk Factors).
        
        Args:
            ticker: Company stock ticker
            section: Section identifier ("1A" for Risk Factors, "7" for MD&A, etc.)
            form_type: Filing type
            year: Fiscal year
            
        Returns:
            Text content of the requested section
        """
        accession = await self._get_latest_accession(ticker, form_type, year)
        
        result = await self.client.call_tool("get_filing_sections", {
            "identifier": ticker,
            "accession_number": accession,
            "form_type": form_type
        })
        
        # Extract the specific section
        sections = result if isinstance(result, dict) else {}
        
        section_key = f"item_{section.lower()}" if section.isdigit() else section
        return sections.get(section_key, sections.get("risk_factors", ""))
    
    async def search_filings(
        self,
        ticker: str,
        form_types: Optional[List[str]] = None,
        days: int = 365,
        limit: int = 10
    ) -> str:
        """
        Search for recent filings for a company.
        
        Use this to discover what filings are available before fetching content.
        
        Args:
            ticker: Company stock ticker
            form_types: List of form types to filter (default: all)
            days: Look back period in days
            limit: Maximum results
            
        Returns:
            JSON string with filing metadata
        """
        args = {
            "identifier": ticker,
            "days": days,
            "limit": limit
        }
        if form_types:
            args["form_type"] = form_types[0]  # MCP may support single type
        
        result = await self.client.call_tool("get_recent_filings", args)
        return result
    
    async def get_company_info(self, ticker: str) -> str:
        """
        Get company information from SEC records.
        
        Args:
            ticker: Company stock ticker
            
        Returns:
            JSON string with company details (name, CIK, sector, etc.)
        """
        result = await self.client.call_tool("get_company_info", {
            "identifier": ticker
        })
        return result
    
    async def _get_latest_accession(
        self, 
        ticker: str, 
        form_type: str,
        year: Optional[int] = None
    ) -> str:
        """Helper to get accession number for a filing."""
        # Search for filings
        filings_result = await self.client.call_tool("get_recent_filings", {
            "identifier": ticker,
            "form_type": form_type,
            "limit": 5
        })
        
        # Parse and find matching filing
        # This is a simplified version; actual implementation would parse the response
        # and match by year if specified
        if isinstance(filings_result, dict) and "filings" in filings_result:
            filings = filings_result["filings"]
            if filings:
                return filings[0].get("accession_number", "")
        
        return ""


class SearchToolkit(Toolkit):
    """
    Web search toolkit for supplementary data.
    
    Uses Brave Search MCP for news and current events.
    """
    
    def __init__(self, mcp_client: Optional[MCPClient] = None):
        super().__init__(name="search_toolkit")
        self.client = mcp_client  # Separate MCP client for Brave
        
        if self.client:
            self.register(self.web_search)
            self.register(self.news_search)
    
    async def web_search(
        self,
        query: str,
        count: int = 10
    ) -> str:
        """
        Search the web for current information.
        
        Use this to find recent news, analysis, or context about a company.
        
        Args:
            query: Search query
            count: Number of results
            
        Returns:
            JSON string with search results
        """
        if not self.client:
            return "Web search not configured"
        
        result = await self.client.call_tool("brave_web_search", {
            "query": query,
            "count": count
        })
        return result
    
    async def news_search(
        self,
        query: str,
        freshness: str = "pw"  # past week
    ) -> str:
        """
        Search for recent news articles.
        
        Use this to understand current market sentiment and news context.
        
        Args:
            query: Search query (e.g., "AAPL earnings")
            freshness: Time filter ("pd"=past day, "pw"=past week, "pm"=past month)
            
        Returns:
            JSON string with news articles
        """
        if not self.client:
            return "News search not configured"
        
        result = await self.client.call_tool("brave_web_search", {
            "query": query,
            "freshness": freshness,
            "news": True
        })
        return result
```

---

## Error Handling

### `insights/core/errors.py`

```python
"""Custom exceptions for MCP operations."""

class MCPError(Exception):
    """Base exception for MCP errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""
    
    def __init__(self, message: str, server_url: str = None):
        self.server_url = server_url
        super().__init__(message)


class MCPToolError(MCPError):
    """Raised when an MCP tool execution fails."""
    
    def __init__(self, message: str, tool_name: str = None, arguments: dict = None):
        self.tool_name = tool_name
        self.arguments = arguments
        super().__init__(message)


class MCPTimeoutError(MCPConnectionError):
    """Raised when MCP call times out."""
    pass
```

---

## Configuration

### `configs/mcp/sec_edgar.yaml`

```yaml
sec_edgar_mcp:
  server_url: "${MCP_SERVER_URL:-http://localhost:8080/sse}"
  transport: "sse"
  timeout: 30.0
  max_retries: 3
  user_agent: "InSights-ai/1.0 (contact@insights.ai)"
  
  # Rate limiting (SEC requires identification)
  rate_limit:
    requests_per_second: 10
    burst_size: 20

brave_search_mcp:
  server_url: "${BRAVE_MCP_URL:-http://localhost:8081/sse}"
  transport: "sse"
  timeout: 15.0
  max_retries: 2
  
  # API key passed to MCP server
  api_key: "${BRAVE_API_KEY}"
```

---

## Docker Compose Integration

```yaml
# docker-compose.yml (excerpt)
services:
  sec-mcp:
    image: ghcr.io/stefanoamorelli/sec-edgar-mcp:latest
    environment:
      - SEC_API_USER_AGENT=${SEC_USER_AGENT}
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  brave-mcp:
    image: ghcr.io/anthropics/brave-search-mcp:latest
    environment:
      - BRAVE_API_KEY=${BRAVE_API_KEY}
    ports:
      - "8081:8081"
```

---

## Usage in Agent

```python
# insights/agents/research/agent.py
from agno.agent import Agent
from insights.adapters.mcp.toolkit import SECToolkit, SearchToolkit
from insights.adapters.mcp.client import get_mcp_client, MCPConfig

def get_research_agent() -> Agent:
    # Initialize MCP clients
    sec_client = get_mcp_client(MCPConfig(server_url=settings.MCP_SERVER_URL))
    
    # Create toolkits
    sec_toolkit = SECToolkit(sec_client)
    
    return Agent(
        model=...,
        tools=[sec_toolkit],
        instructions=[
            "Use SECToolkit.get_filing() to fetch SEC filing content.",
            "Use SECToolkit.get_filing_section() for specific sections like Risk Factors.",
            "Always check filing availability with search_filings() first."
        ]
    )
```

---

## Available sec-edgar-mcp Tools

| Tool | Description | Key Arguments |
|------|-------------|---------------|
| `get_company_info` | Get company details | `identifier` (ticker/CIK) |
| `get_recent_filings` | Search filings | `identifier`, `form_type`, `days`, `limit` |
| `get_filing_content` | Get full filing text | `identifier`, `accession_number` |
| `get_filing_sections` | Get parsed sections | `identifier`, `accession_number`, `form_type` |
| `get_financials` | Get financial data | `identifier`, `statement_type` |
| `get_insider_transactions` | Get Form 4 data | `identifier`, `days`, `limit` |
| `search_companies` | Search by name | `query`, `limit` |