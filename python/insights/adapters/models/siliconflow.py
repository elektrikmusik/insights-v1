"""
SiliconFlow adapter implementation.

Provides access to models via SiliconFlow API using the OpenAI SDK.
Primarily for DeepSeek, Qwen and other Chinese models.
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


class SiliconFlowAdapter(BaseChatModel):
    """
    Adapter for SiliconFlow API.

    Uses standard OpenAI Async SDK but points to SiliconFlow base URL.
    """

    def __init__(
        self,
        model_id: str,
        context_window: int = 32000
    ):
        """
        Initialize SiliconFlow adapter.

        Args:
            model_id: SiliconFlow model ID (e.g., 'deepseek-ai/DeepSeek-V3')
            context_window: Max context window size
        """
        self._model_id = model_id
        self._context_window = context_window
        self._client = AsyncOpenAI(
            base_url=settings.SILICONFLOW_BASE_URL,
            api_key=settings.SILICONFLOW_API_KEY,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "siliconflow"

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
        """Invoke SiliconFlow model."""
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
                raise LLMProviderError("siliconflow", "No choices returned from SiliconFlow")

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"SiliconFlow invoke failed: {e}")
            raise LLMProviderError("siliconflow", f"SiliconFlow invoke failed: {e}") from e

    async def invoke_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Stream response from SiliconFlow."""
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
            logger.error(f"SiliconFlow stream failed: {e}")
            raise LLMProviderError("siliconflow", f"SiliconFlow stream failed: {e}") from e

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

        Using OpenAI beta parse method.
        """
        try:
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
                 raise LLMProviderError("siliconflow", "Startuctured output parsing failed")

            return parsed_response # type: ignore

        except Exception as e:
            logger.error(f"SiliconFlow structured invoke failed: {e}")
            raise LLMProviderError("siliconflow", f"SiliconFlow structured invoke failed: {e}") from e

    async def calculate_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Simple approximation for now as tokenizer access varies
        return len(text) // 4
