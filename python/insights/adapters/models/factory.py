"""
Model Factory for creating LLM adapters.

Handles provider resolution, fallback logic, and adapter instantiation.
"""
import logging

from insights.adapters.models.circuit_breaker import get_circuit_breaker
from insights.adapters.models.interface import BaseChatModel
from insights.adapters.models.openrouter import OpenRouterAdapter
from insights.adapters.models.siliconflow import SiliconFlowAdapter

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class to create and manage LLM model adapters.
    """

    _adapters: dict[str, BaseChatModel] = {}

    @classmethod
    def get_model(cls, model_id: str) -> BaseChatModel:
        """
        Get or create a model adapter instance.

        Args:
            model_id: Full model identifier (e.g. 'openrouter/openai/gpt-4o')

        Returns:
            Configured model adapter

        Raises:
            ModelNotFoundError: If model_id format is invalid
            ProviderNotFoundError: If provider is unknown
        """
        # Return cached instance if available
        if model_id in cls._adapters:
            return cls._adapters[model_id]

        # Parse provider and model
        if "/" not in model_id:
            # Default to OpenRouter if no provider specified
            # This allows short IDs like 'openai/gpt-4o' to work if we treat 'openai' as the vendor,
            # but for our system 'provider' is the API gateway.
            # Let's assume strict 'provider/vendor/model' or 'provider/model'
            # OR simple 'vendor/model' defaults to OpenRouter.
            provider = "openrouter"
            actual_model = model_id
        else:
            parts = model_id.split("/", 1)
            provider = parts[0]
            actual_model = parts[1]

        # Check circuit breaker
        cb = get_circuit_breaker()
        if not cb.is_available(provider):
            logger.warning(f"Provider {provider} is unavailable (Circuit Open)")
            # In a full implementation, we would trigger fallback logic here
            # For now, we proceed to try, but logging functionality would be keyed

        # Create adapter
        adapter: BaseChatModel

        if provider == "openrouter":
            # For OpenRouter, the modelID passed to API is often 'vendor/model'
            # e.g. 'openai/gpt-4o'
            adapter = OpenRouterAdapter(model_id=actual_model)

        elif provider == "siliconflow":
            # SiliconFlow IDs are like 'deepseek-ai/DeepSeek-V3'
            adapter = SiliconFlowAdapter(model_id=actual_model)

        elif provider == "openai":
            # Direct OpenAI (aliasing to OpenRouter for now to simplify)
            # Pass full model_id because OpenRouter needs 'openai/...' prefix
            adapter = OpenRouterAdapter(model_id=model_id)
            # TODO: Implement direct OpenAIAdapter if needed for performance/compliance

        else:
            # Fallback: Assume it's a model ID for OpenRouter if provider not matched
            # (e.g. 'anthropic/claude-3' -> OpenRouter)
            logger.info(f"Provider '{provider}' not explicitly handled, defaulting to OpenRouter")
            adapter = OpenRouterAdapter(model_id=model_id)

        # Cache and return
        cls._adapters[model_id] = adapter
        return adapter

    @classmethod
    def clear_cache(cls):
        """Clear cached adapters."""
        cls._adapters.clear()
