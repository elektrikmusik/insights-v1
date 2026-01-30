import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from mcp.types import CallToolResult, TextContent
from insights.adapters.mcp.client import MCPClient, MCPConfig, MCPToolError

@pytest.mark.asyncio
async def test_mcp_connect():
    config = MCPConfig(server_url="http://test")
    client = MCPClient(config)
    
    with patch("insights.adapters.mcp.client.sse_client") as mock_sse:
        with patch("insights.adapters.mcp.client.ClientSession") as mock_session_cls:
            # Setup mocks
            mock_transport = (AsyncMock(), AsyncMock())
            mock_sse.return_value.__aenter__.return_value = mock_transport
            
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            
            await client.connect()
            
            assert client._connected
            mock_session.initialize.assert_awaited_once()

@pytest.mark.asyncio
async def test_mcp_call_tool_success():
    config = MCPConfig(server_url="http://test")
    client = MCPClient(config)
    client._connected = True
    client._session = AsyncMock()
    
    # Mock result with TextContent
    result = CallToolResult(content=[TextContent(type="text", text="Success")])
    client._session.call_tool.return_value = result
    
    output = await client.call_tool("test_tool", {})
    assert output == "Success"

@pytest.mark.asyncio
async def test_mcp_call_tool_error():
    config = MCPConfig(server_url="http://test")
    client = MCPClient(config)
    client._connected = True
    client._session = AsyncMock()
    
    # Mock result with Error
    result = CallToolResult(content=[TextContent(type="text", text="Failure")], isError=True)
    client._session.call_tool.return_value = result
    
    with pytest.raises(MCPToolError) as exc:
        await client.call_tool("test_tool", {})
    
    # Verify we raised correct error (Checking substring of message)
    # The error message should contain "Tool 'test_tool' failed: Failure"
    assert "Tool 'test_tool' failed" in str(exc.value)
