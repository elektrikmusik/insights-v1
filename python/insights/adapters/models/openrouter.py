"""
OpenRouter adapter implementation.

Provides access to models via OpenRouter API using the OpenAI SDK.
Support for streaming and structured output (via json_mode or tool_calls).
"""
import logging
from collections.abc import AsyncGenerator
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from insights.adapters.models.interface import BaseChatModel
from insights.core.config import settings
from insights.core.errors import LLMProviderError
from insights.core.retry import retry_llm_call

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenRouterAdapter(BaseChatModel):
    """
    Adapter for OpenRouter API.

    Uses standard OpenAI Async SDK but points to OpenRouter base URL.
    Handles extra headers and provider-specific error mapping.
    """

    def __init__(
        self,
        model_id: str,
        context_window: int = 128000
    ):
        """
        Initialize OpenRouter adapter.

        Args:
            model_id: OpenRouter model ID (e.g., 'openai/gpt-4o')
            context_window: Max context window size
        """
        self._model_id = model_id
        self._context_window = context_window
        self._client = AsyncOpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            default_headers={
                "HTTP-Referer": "https://insights-ai.app",  # Required by OpenRouter
                "X-Title": "InSights AI",
            }
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "openrouter"

    @property
    def context_window(self) -> int:
        return self._context_window

    @retry_llm_call
    async def invoke(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> str:
        """Invoke OpenRouter model."""
        try:
            params = {
                "model": self.model_id,
                "messages": messages,
                "temperature": temperature,
                **kwargs
            }
            if max_tokens:
                params["max_tokens"] = max_tokens

            response = await self._client.chat.completions.create(**params)

            if not response.choices:
                raise LLMProviderError("openrouter", "No choices returned from OpenRouter")

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"OpenRouter invoke failed: {e}")
            raise LLMProviderError("openrouter", f"OpenRouter invoke failed: {e}") from e

    async def invoke_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Stream response from OpenRouter."""
        try:
            params = {
                "model": self.model_id,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
                **kwargs
            }
            if max_tokens:
                params["max_tokens"] = max_tokens

            stream = await self._client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenRouter stream failed: {e}")
            raise LLMProviderError("openrouter", f"OpenRouter stream failed: {e}") from e

    @retry_llm_call
    async def invoke_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        temperature: float = 0.0,
        **kwargs: Any
    ) -> T:
        """
        Invoke with structured output enforcement.

        Uses OpenAI's new parsing helper if available, or falls back to standard
        tool calling / json mode depending on support.
        """
        try:
            # OpenRouter supports various backends, so standardizing structured output
            # can be tricky. We use the instructor-like approach or OpenAI's beta parse method.
            # Here we use the beta.chat.completions.parse which is robust.

            params = {
                "model": self.model_id,
                "messages": messages,
                "response_format": response_model,
                "temperature": temperature,
                **kwargs
            }

            completion = await self._client.beta.chat.completions.parse(**params)

            parsed_response = completion.choices[0].message.parsed
            if not parsed_response:
                 raise LLMProviderError("openrouter", "Startuctured output parsing failed")

            return parsed_response # type: ignore

        except Exception as e:
            logger.error(f"OpenRouter structured invoke failed: {e}")
            # Fallback: detailed error handling or manual JSON parsing could go here
            raise LLMProviderError("openrouter", f"OpenRouter structured invoke failed: {e}") from e

    async def calculate_tokens(self, text: str) -> int:
        """
        Estimate token count using tiktoken encoding for GPT-4o.
        Not perfect for all models but a good approximation.
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback if tiktoken not installed or fails
            return len(text) // 4
