"""
Unit tests for Model Management layer.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from insights.adapters.models.factory import ModelFactory
from insights.adapters.models.openrouter import OpenRouterAdapter
from insights.adapters.models.siliconflow import SiliconFlowAdapter
from insights.core.errors import LLMProviderError


@pytest.fixture
def mock_openai():
    with patch("insights.adapters.models.openrouter.AsyncOpenAI") as mock:
        yield mock

@pytest.fixture
def mock_siliconflow():
    with patch("insights.adapters.models.siliconflow.AsyncOpenAI") as mock:
        yield mock

class TestModelFactory:
    def teardown_method(self):
        ModelFactory.clear_cache()

    def test_get_openrouter_model(self):
        adapter = ModelFactory.get_model("openrouter/openai/gpt-4o")
        assert isinstance(adapter, OpenRouterAdapter)
        assert adapter.model_id == "openai/gpt-4o"
        assert adapter.provider == "openrouter"

    def test_get_siliconflow_model(self):
        adapter = ModelFactory.get_model("siliconflow/deepseek-ai/DeepSeek-V3")
        assert isinstance(adapter, SiliconFlowAdapter)
        assert adapter.model_id == "deepseek-ai/DeepSeek-V3"
        assert adapter.provider == "siliconflow"

    def test_default_to_openrouter(self):
        adapter = ModelFactory.get_model("anthropic/claude-3-opus")
        assert isinstance(adapter, OpenRouterAdapter)
        assert adapter.model_id == "anthropic/claude-3-opus"

    def test_short_model_id(self):
        adapter = ModelFactory.get_model("openai/gpt-4o")
        # Should default to OpenRouter but keep ID
        assert isinstance(adapter, OpenRouterAdapter)
        assert adapter.model_id == "openai/gpt-4o"

    def test_caching(self):
        a1 = ModelFactory.get_model("openai/gpt-4o")
        a2 = ModelFactory.get_model("openai/gpt-4o")
        assert a1 is a2

class TestOpenRouterAdapter:
    @pytest.mark.asyncio
    async def test_invoke_success(self, mock_openai):
        # Setup mock
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello World"))]
        mock_client.chat.completions.create.return_value = mock_response

        adapter = OpenRouterAdapter("openai/gpt-4o")
        response = await adapter.invoke([{"role": "user", "content": "Hi"}])

        assert response == "Hello World"
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_failure(self, mock_openai):
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        adapter = OpenRouterAdapter("openai/gpt-4o")

        with pytest.raises(LLMProviderError) as exc:
            await adapter.invoke([{"role": "user", "content": "Hi"}])

        assert "openrouter" in str(exc.value)

    @pytest.mark.asyncio
    async def test_invoke_stream(self, mock_openai):
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        # Mock streaming response
        async def mock_stream(**kwargs):
            chunks = ["Hello", " ", "World"]
            for c in chunks:
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content=c))]
                yield chunk

        mock_client.chat.completions.create.side_effect = mock_stream

        adapter = OpenRouterAdapter("openai/gpt-4o")
        chunks = []
        async for chunk in adapter.invoke_stream([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert "".join(chunks) == "Hello World"

class TestSiliconFlowAdapter:
    @pytest.mark.asyncio
    async def test_invoke_success(self, mock_siliconflow):
        mock_client = AsyncMock()
        mock_siliconflow.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Ni Hao"))]
        mock_client.chat.completions.create.return_value = mock_response

        adapter = SiliconFlowAdapter("deepseek-ai/DeepSeek-V3")
        response = await adapter.invoke([{"role": "user", "content": "Hi"}])

        assert response == "Ni Hao"

class TestGoogleEmbeddingAdapter:
    @pytest.mark.asyncio
    async def test_embed_success(self):
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"embedding": {"values": [0.1, 0.2, 0.3]}}
            mock_post.return_value = mock_response

            from insights.adapters.models.embedding import GoogleEmbeddingAdapter
            adapter = GoogleEmbeddingAdapter()
            embedding = await adapter.embed("hello")

            assert len(embedding) == 3
            assert embedding == [0.1, 0.2, 0.3]
            mock_post.assert_called_once()

