"""
Abstract base class for all LLM adapters.

All chat models must implement this interface to ensure consistent behavior
across the application regardless of the underlying provider.
"""
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseEmbeddingModel(ABC):
    """Abstract interface for embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension (e.g., 768)."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single string."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings."""
        pass


class BaseChatModel(ABC):
    """
    Abstract interface for chat-based LLM models.

    All provider adapters (OpenRouter, SiliconFlow, OpenAI, etc.) must
    implement this interface.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The model identifier (e.g., 'openai/gpt-4o')."""
        pass

    @property
    @abstractmethod
    def provider(self) -> str:
        """The provider name (e.g., 'openrouter', 'openai')."""
        pass

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum context window size in tokens."""
        pass

    @abstractmethod
    async def invoke(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> str:
        """
        Invoke the model with a list of messages.

        Args:
            messages: List of message dicts (role, content)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments

        Returns:
            The model's response text
        """
        pass

    @abstractmethod
    def invoke_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """
        Stream model response chunks.

        Args:
            messages: List of message dicts (role, content)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments

        Yields:
            Response text chunks
        """
        pass

    @abstractmethod
    async def invoke_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        temperature: float = 0.0,
        **kwargs: Any
    ) -> T:
        """
        Invoke model and enforce structured output schema.

        Args:
            messages: List of message dicts (role, content)
            response_model: Pydantic model class for validation
            temperature: Sampling temperature (usually low for structured)
            **kwargs: Additional provider-specific arguments

        Returns:
            Instance of response_model populated with data
        """
        pass

    @abstractmethod
    async def calculate_tokens(self, text: str) -> int:
        """
        Calculate token usage for text.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        pass
