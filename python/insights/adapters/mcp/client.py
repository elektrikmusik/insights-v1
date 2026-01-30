"""
MCP Client using mcp-python SDK.
"""
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, TextContent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from insights.core.errors import MCPConnectionError, MCPToolError
from insights.core.config import settings

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
    Wraps mcp-python SDK.
    """
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self._exit_stack = AsyncExitStack()
        self._session: Optional[ClientSession] = None
        self._connected = False
    
    async def connect(self) -> None:
        """Establish connection to MCP server."""
        if self._connected and self._session:
            return
            
        logger.info(f"Connecting to MCP server at {self.config.server_url}")
        
        try:
            if self.config.transport != MCPTransport.SSE:
                raise NotImplementedError("Only SSE transport is currently implemented")
            
            # Enter sse_client context
            sse_transport = await self._exit_stack.enter_async_context(
                sse_client(self.config.server_url)
            )
            read_stream, write_stream = sse_transport
            
            # Enter session context
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            
            await self._session.initialize()
            self._connected = True
            logger.info("Connected to MCP server successfully")
            
        except Exception as e:
            await self.disconnect()
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}") from e
            
    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        await self._exit_stack.aclose()
        self._session = None
        self._connected = False
        
    async def ensure_connected(self) -> None:
        """Ensure connection is active."""
        if not self._connected or not self._session:
            await self.connect()
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((MCPConnectionError, TimeoutError, ConnectionError)),
        reraise=True
    )
    async def call_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> str:
        """
        Call a tool on the MCP server.
        """
        await self.ensure_connected()
        
        if not self._session:
            raise MCPConnectionError("Session not available")
            
        try:
            result: CallToolResult = await self._session.call_tool(tool_name, arguments)
            
            if result.isError:
                error_texts = [
                    blk.text for blk in result.content 
                    if isinstance(blk, TextContent)
                ]
                error_msg = "\n".join(error_texts) or "Unknown error"
                raise MCPToolError(tool_name, error_msg)
            
            content_texts = []
            for content in result.content:
                if isinstance(content, TextContent):
                    content_texts.append(content.text)
            
            return "\n".join(content_texts)
            
        except Exception as e:
            if isinstance(e, MCPToolError):
                raise
            raise MCPToolError(tool_name, f"Tool execution exception: {e}") from e


# Singleton instance
_mcp_client: Optional[MCPClient] = None


def get_mcp_client(config: Optional[MCPConfig] = None) -> MCPClient:
    """Get or create singleton MCP client."""
    global _mcp_client
    
    if _mcp_client is None:
        if config is None:
            config = MCPConfig(server_url=settings.MCP_SEC_URL)
        _mcp_client = MCPClient(config)
    
    return _mcp_client
