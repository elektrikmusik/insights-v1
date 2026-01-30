"""
Shared pytest fixtures for InSights-ai.
"""
import os
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

# Select "test" environment
os.environ["ENV"] = "test"
os.environ["DEBUG"] = "true"

from insights.api.main import app

@pytest.fixture(scope="session")
def event_loop():
    """Create a persistent event loop for the session."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_client():
    """FastAPI TestClient for integration testing."""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_mcp_client():
    """Mock for MCP client tool calls."""
    mock = AsyncMock()
    mock.call_tool = AsyncMock(return_value="ITEM 1A. RISK FACTORS...")
    return mock

@pytest.fixture
def mock_openrouter():
    """Mock for OpenRouter LLM calls."""
    mock = AsyncMock()
    # Mocking Agno response structure
    mock_resp = MagicMock()
    mock_resp.content = '{"analysis": "Risk is moderate", "sentiment": "neutral"}'
    mock.arun = AsyncMock(return_value=mock_resp)
    return mock
