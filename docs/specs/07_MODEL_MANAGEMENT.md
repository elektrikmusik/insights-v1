# 07. Model Management & Factory

## Overview

The Model Management layer abstracts LLM provider complexity, implementing:
- **Unified Interface**: Single API for all providers
- **Multi-Provider Support**: OpenRouter, SiliconFlow, Azure, Direct APIs
- **Provider Aggregators**: OpenRouter & SiliconFlow for model variety
- **Circuit Breaker**: Auto-failover on provider failures
- **Usage Tracking**: Cost auditing per request
- **Structured Outputs**: Pydantic model responses

---

## Supported Providers

| Provider | Type | Models | Best For |
|----------|------|--------|----------|
| **OpenRouter** | Aggregator | GPT-4o, Claude, DeepSeek, Gemini, Llama | US/EU users, widest selection |
| **SiliconFlow** | Aggregator | Qwen, DeepSeek, Yi, GLM | Asia users, Chinese models |
| **Azure OpenAI** | Direct | GPT-4, GPT-4o | Enterprise, compliance |
| **OpenAI** | Direct | GPT-4o, o1 | Lowest latency for OpenAI |
| **Anthropic** | Direct | Claude 3.5, Claude 3 | Fastest Claude access |
| **Google AI** | Direct | Gemini Pro, Gemini Flash | Embeddings, long context |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Agent / Expert Layer                                                        │
│  └── model = ModelFactory.get_chat_model("openai/gpt-4o")                   │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Model Factory (insights/adapters/models/factory.py)                         │
│  ├── Resolves model ID → Provider                                            │
│  ├── Applies Circuit Breaker per provider                                    │
│  └── Returns singleton adapter instance                                      │
└──────────────────────────┬──────────────────────────────────────────────────┘
                           │
       ┌───────────────────┼─────────────────────┬─────────────────────┐
       │                   │                     │                     │
       ▼                   ▼                     ▼                     ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ OpenRouterAdptr │ │ SiliconFlowAdpt │ │ AzureOpenAIAdpt │ │ DirectAdapters  │
│ (Aggregator)    │ │ (Aggregator)    │ │ (Enterprise)    │ │ (OpenAI/Anthr)  │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │                   │
         ▼                   ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  External APIs                                                               │
│  ├── OpenRouter → GPT-4o, Claude, DeepSeek, Gemini, Llama                   │
│  ├── SiliconFlow → Qwen, DeepSeek, Yi, GLM, Llama                          │
│  ├── Azure OpenAI → GPT-4, GPT-4o (enterprise)                              │
│  ├── OpenAI API → GPT-4o, o1 (direct)                                       │
│  └── Anthropic API → Claude 3.5 Sonnet, Claude 3 Opus (direct)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Abstract Interface

### `insights/adapters/models/interface.py`

```python
"""
Abstract base class for all LLM adapters.
All chat models must implement this interface.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator, Optional, Type
from pydantic import BaseModel


class BaseChatModel(ABC):
    """
    Abstract interface for chat-based LLM models.
    
    All provider adapters must implement this interface to ensure
    consistent behavior across the application.
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
        """Maximum context window in tokens."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Type[BaseModel]] = None
    ) -> str:
        """
        Standard request-response generation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum output tokens
            response_format: Optional Pydantic model for structured output
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0
    ) -> AsyncGenerator[str, None]:
        """
        Streaming response generation.
        
        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            
        Yields:
            Text chunks as they are generated
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the model/provider is available.
        
        Returns:
            True if healthy, False otherwise
        """
        pass


class ModelUsage(BaseModel):
    """Token usage and cost tracking."""
    request_id: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
```

---

## OpenRouter Adapter

### `insights/adapters/models/openrouter.py`

```python
"""
OpenRouter adapter - Primary LLM provider.

OpenRouter provides unified access to multiple models:
- OpenAI: GPT-4o, GPT-4-turbo
- Anthropic: Claude 3.5 Sonnet, Claude 3 Opus
- DeepSeek: DeepSeek-V3, DeepSeek-Chat
- Google: Gemini Pro
- Meta: Llama 3.1 405B
"""
import time
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional, Type
from uuid import uuid4
import httpx
from pydantic import BaseModel
import tiktoken

from insights.core.config import settings
from .interface import BaseChatModel, ModelUsage

logger = logging.getLogger(__name__)


# Cost per 1K tokens (approximate, check OpenRouter for latest)
MODEL_COSTS = {
    "openai/gpt-4o": {"input": 0.0025, "output": 0.01},
    "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "anthropic/claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    "anthropic/claude-3-opus": {"input": 0.015, "output": 0.075},
    "deepseek/deepseek-chat": {"input": 0.00014, "output": 0.00028},
    "google/gemini-pro-1.5": {"input": 0.00125, "output": 0.005},
    "meta-llama/llama-3.1-405b-instruct": {"input": 0.003, "output": 0.003},
}


class OpenRouterAdapter(BaseChatModel):
    """
    OpenRouter API adapter.
    
    Provides access to multiple LLM providers through a single API.
    """
    
    API_BASE = "https://openrouter.ai/api/v1"
    
    def __init__(
        self,
        model_id: str = "openai/gpt-4o",
        api_key: Optional[str] = None,
        timeout: float = 60.0
    ):
        self._model_id = model_id
        self._api_key = api_key or settings.OPENROUTER_API_KEY
        self._timeout = timeout
        
        # Context windows (approximate)
        self._context_windows = {
            "openai/gpt-4o": 128000,
            "openai/gpt-4o-mini": 128000,
            "anthropic/claude-3.5-sonnet": 200000,
            "anthropic/claude-3-opus": 200000,
            "deepseek/deepseek-chat": 64000,
            "google/gemini-pro-1.5": 1000000,
            "meta-llama/llama-3.1-405b-instruct": 128000,
        }
        
        self._client = httpx.AsyncClient(
            base_url=self.API_BASE,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "HTTP-Referer": settings.APP_URL or "https://insights.ai",
                "X-Title": "InSights-ai"
            },
            timeout=self._timeout
        )
        
        # Token counter (use GPT-4 tokenizer as approximation)
        try:
            self._tokenizer = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def provider(self) -> str:
        return "openrouter"
    
    @property
    def context_window(self) -> int:
        return self._context_windows.get(self._model_id, 128000)
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Type[BaseModel]] = None
    ) -> str:
        """Generate a response using OpenRouter."""
        start_time = time.time()
        request_id = str(uuid4())
        
        # Build request body
        body = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            body["max_tokens"] = max_tokens
        
        # Structured output support
        if response_format:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": response_format.model_json_schema()
                }
            }
        
        try:
            response = await self._client.post("/chat/completions", json=body)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract response
            content = data["choices"][0]["message"]["content"]
            
            # Track usage
            usage = data.get("usage", {})
            await self._log_usage(
                request_id=request_id,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                latency_ms=int((time.time() - start_time) * 1000)
            )
            
            return content
            
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter request failed: {e}")
            raise
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0
    ) -> AsyncGenerator[str, None]:
        """Stream a response from OpenRouter."""
        body = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        
        async with self._client.stream("POST", "/chat/completions", json=body) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    import json
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self._tokenizer.encode(text))
    
    async def health_check(self) -> bool:
        """Check OpenRouter API availability."""
        try:
            response = await self._client.get("/models")
            return response.status_code == 200
        except Exception:
            return False
    
    async def _log_usage(
        self,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int
    ):
        """Log usage to audit table."""
        costs = MODEL_COSTS.get(self._model_id, {"input": 0, "output": 0})
        cost_usd = (
            (input_tokens / 1000 * costs["input"]) +
            (output_tokens / 1000 * costs["output"])
        )
        
        usage = ModelUsage(
            request_id=request_id,
            model_id=self._model_id,
            provider=self.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms
        )
        
        logger.info(f"LLM Usage: {usage.model_dump_json()}")
        
        # Async save to database (non-blocking)
        # This is fire-and-forget
        try:
            from insights.adapters.db.manager import db_manager
            await db_manager.log_llm_usage(
                job_id=None,  # Set by caller if available
                operation="generation",
                model_id=self._model_id,
                provider=self.provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms
            )
        except Exception as e:
            logger.warning(f"Failed to log usage to DB: {e}")
```

---

## SiliconFlow Adapter

### `insights/adapters/models/siliconflow.py`

SiliconFlow is a Chinese AI infrastructure provider offering access to multiple models including Qwen, DeepSeek, Yi, and GLM. It's particularly useful for:
- Users in Asia (lower latency)
- Chinese language models
- Cost-effective DeepSeek access

```python
"""
SiliconFlow adapter - Asian market LLM aggregator.

SiliconFlow provides access to:
- DeepSeek: DeepSeek-V3, DeepSeek-Chat
- Qwen: Qwen2.5-72B, Qwen2.5-32B
- Yi: Yi-1.5-34B
- GLM: GLM-4
- Llama: Llama 3.1 variants
"""
import time
import logging
from typing import List, Dict, Any, AsyncGenerator, Optional, Type
from uuid import uuid4
import httpx
from pydantic import BaseModel
import tiktoken

from insights.core.config import settings
from .interface import BaseChatModel, ModelUsage

logger = logging.getLogger(__name__)


# Cost per 1M tokens (SiliconFlow pricing in CNY, converted to USD approx)
SILICONFLOW_COSTS = {
    "deepseek-ai/DeepSeek-V3": {"input": 0.07, "output": 0.14},
    "deepseek-ai/DeepSeek-Chat": {"input": 0.07, "output": 0.14},
    "Qwen/Qwen2.5-72B-Instruct": {"input": 0.28, "output": 0.28},
    "Qwen/Qwen2.5-32B-Instruct": {"input": 0.14, "output": 0.14},
    "01-ai/Yi-1.5-34B-Chat": {"input": 0.14, "output": 0.14},
    "THUDM/glm-4-9b-chat": {"input": 0.07, "output": 0.07},
    "meta-llama/Llama-3.1-70B-Instruct": {"input": 0.28, "output": 0.28},
}


class SiliconFlowAdapter(BaseChatModel):
    """
    SiliconFlow API adapter.
    
    Provides access to Chinese and open-source models through SiliconFlow.
    """
    
    API_BASE = "https://api.siliconflow.cn/v1"
    
    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-V3",
        api_key: Optional[str] = None,
        timeout: float = 60.0
    ):
        self._model_id = model_id
        self._api_key = api_key or settings.SILICONFLOW_API_KEY
        self._timeout = timeout
        
        # Context windows
        self._context_windows = {
            "deepseek-ai/DeepSeek-V3": 64000,
            "deepseek-ai/DeepSeek-Chat": 64000,
            "Qwen/Qwen2.5-72B-Instruct": 128000,
            "Qwen/Qwen2.5-32B-Instruct": 128000,
            "01-ai/Yi-1.5-34B-Chat": 32000,
            "THUDM/glm-4-9b-chat": 128000,
            "meta-llama/Llama-3.1-70B-Instruct": 128000,
        }
        
        self._client = httpx.AsyncClient(
            base_url=self.API_BASE,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            },
            timeout=self._timeout
        )
        
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None
    
    @property
    def model_id(self) -> str:
        return self._model_id
    
    @property
    def provider(self) -> str:
        return "siliconflow"
    
    @property
    def context_window(self) -> int:
        return self._context_windows.get(self._model_id, 64000)
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[Type[BaseModel]] = None
    ) -> str:
        """Generate a response using SiliconFlow."""
        start_time = time.time()
        request_id = str(uuid4())
        
        body = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            body["max_tokens"] = max_tokens
        
        # Structured output support (JSON mode)
        if response_format:
            body["response_format"] = {"type": "json_object"}
            # Add schema hint in system message
            schema_hint = f"\n\nRespond with valid JSON matching this schema: {response_format.model_json_schema()}"
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += schema_hint
            else:
                messages.insert(0, {"role": "system", "content": schema_hint})
        
        try:
            response = await self._client.post("/chat/completions", json=body)
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Track usage
            usage = data.get("usage", {})
            await self._log_usage(
                request_id=request_id,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                latency_ms=int((time.time() - start_time) * 1000)
            )
            
            return content
            
        except httpx.HTTPStatusError as e:
            logger.error(f"SiliconFlow API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"SiliconFlow request failed: {e}")
            raise
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0
    ) -> AsyncGenerator[str, None]:
        """Stream a response from SiliconFlow."""
        body = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        
        async with self._client.stream("POST", "/chat/completions", json=body) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    import json
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return len(text) // 4  # Rough estimate
    
    async def health_check(self) -> bool:
        """Check SiliconFlow API availability."""
        try:
            response = await self._client.get("/models")
            return response.status_code == 200
        except Exception:
            return False
    
    async def _log_usage(
        self,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int
    ):
        """Log usage metrics."""
        costs = SILICONFLOW_COSTS.get(self._model_id, {"input": 0.1, "output": 0.1})
        cost_usd = (
            (input_tokens / 1_000_000 * costs["input"]) +
            (output_tokens / 1_000_000 * costs["output"])
        )
        
        usage = ModelUsage(
            request_id=request_id,
            model_id=self._model_id,
            provider=self.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms
        )
        
        logger.info(f"LLM Usage: {usage.model_dump_json()}")
```

---

## Circuit Breaker

### `insights/adapters/models/circuit_breaker.py`

```python
"""
Circuit Breaker pattern for LLM provider failover.

States:
- CLOSED: Normal operation, requests go to primary
- OPEN: Primary failed, all requests go to fallback
- HALF_OPEN: Testing if primary recovered
"""
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict
import asyncio

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failover active
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for LLM provider failover.
    
    Configuration:
    - failure_threshold: Number of failures before opening (default: 3)
    - recovery_timeout: Seconds before attempting recovery (default: 300)
    - half_open_requests: Test requests before closing (default: 1)
    """
    failure_threshold: int = 3
    recovery_timeout: int = 300  # 5 minutes
    half_open_requests: int = 1
    
    # State tracking
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time: float = field(default=0.0)
    half_open_successes: int = field(default=0)
    
    # Lock for thread safety
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    async def record_success(self):
        """Record a successful request."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                if self.half_open_successes >= self.half_open_requests:
                    logger.info("Circuit breaker closing - primary recovered")
                    self._close()
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    async def record_failure(self):
        """Record a failed request."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker opening - recovery failed")
                self._open()
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    logger.warning(
                        f"Circuit breaker opening after {self.failure_count} failures"
                    )
                    self._open()
    
    async def should_use_fallback(self) -> bool:
        """Check if fallback should be used."""
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return False
            
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout passed
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    logger.info("Circuit breaker half-opening - testing primary")
                    self._half_open()
                    return False  # Try primary
                return True  # Use fallback
            
            # HALF_OPEN: allow request to primary
            return False
    
    def _open(self):
        self.state = CircuitState.OPEN
        self.half_open_successes = 0
    
    def _half_open(self):
        self.state = CircuitState.HALF_OPEN
        self.half_open_successes = 0
    
    def _close(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_successes = 0


# Global circuit breakers per provider
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(provider: str) -> CircuitBreaker:
    """Get or create circuit breaker for a provider."""
    if provider not in _circuit_breakers:
        _circuit_breakers[provider] = CircuitBreaker()
    return _circuit_breakers[provider]
```

---

## Model Factory

### `insights/adapters/models/factory.py`

```python
"""
Model Factory - Central point for getting LLM instances.

Handles:
- Multi-provider resolution (OpenRouter, SiliconFlow, Azure, Direct)
- Singleton instances per model
- Circuit breaker integration per provider
- Configurable fallback chains
"""
import logging
from typing import Optional, Dict, Literal
from dataclasses import dataclass
from enum import Enum

from insights.core.config import settings
from .interface import BaseChatModel
from .openrouter import OpenRouterAdapter
from .siliconflow import SiliconFlowAdapter
from .circuit_breaker import get_circuit_breaker

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    """Supported LLM providers."""
    OPENROUTER = "openrouter"
    SILICONFLOW = "siliconflow"
    AZURE = "azure"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# Model ID prefixes that map to specific providers
PROVIDER_PREFIXES = {
    # SiliconFlow models (Chinese providers)
    "deepseek-ai/": Provider.SILICONFLOW,
    "Qwen/": Provider.SILICONFLOW,
    "01-ai/": Provider.SILICONFLOW,
    "THUDM/": Provider.SILICONFLOW,
    
    # Direct OpenAI (use for lowest latency)
    "gpt-": Provider.OPENAI,  # e.g., "gpt-4o" without prefix
    
    # Azure OpenAI (enterprise)
    "azure/": Provider.AZURE,
    
    # The rest go through OpenRouter (default)
    "openai/": Provider.OPENROUTER,
    "anthropic/": Provider.OPENROUTER,
    "google/": Provider.OPENROUTER,
    "meta-llama/": Provider.OPENROUTER,
}


@dataclass
class FallbackChain:
    """Fallback configuration per provider."""
    openrouter: tuple = ("openai/gpt-4o", "anthropic/claude-3.5-sonnet", "deepseek/deepseek-chat")
    siliconflow: tuple = ("deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct")


class ModelFactory:
    """
    Factory for creating LLM model instances.
    
    Supports multiple providers with automatic routing based on model ID.
    
    Usage:
        # OpenRouter (default for openai/, anthropic/ prefixes)
        model = ModelFactory.get_chat_model("openai/gpt-4o")
        
        # SiliconFlow (for Chinese models)
        model = ModelFactory.get_chat_model("deepseek-ai/DeepSeek-V3")
        
        # Explicit provider override
        model = ModelFactory.get_chat_model("openai/gpt-4o", provider="siliconflow")
    """
    
    _instances: Dict[str, BaseChatModel] = {}
    _fallback_chain: FallbackChain = FallbackChain()
    
    @classmethod
    def resolve_provider(cls, model_id: str) -> Provider:
        """Determine which provider to use for a model ID."""
        for prefix, provider in PROVIDER_PREFIXES.items():
            if model_id.startswith(prefix):
                return provider
        return Provider.OPENROUTER  # Default
    
    @classmethod
    def configure_fallback(
        cls, 
        provider: str,
        models: tuple[str, ...]
    ):
        """Configure fallback chain for a specific provider."""
        if provider == "openrouter":
            cls._fallback_chain.openrouter = models
        elif provider == "siliconflow":
            cls._fallback_chain.siliconflow = models
    
    @classmethod
    def get_chat_model(
        cls, 
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        use_fallback: bool = True
    ) -> BaseChatModel:
        """
        Get a chat model instance.
        
        Args:
            model_id: Model identifier (e.g., "openai/gpt-4o")
            provider: Force a specific provider (override auto-detection)
            use_fallback: Whether to apply circuit breaker fallback
            
        Returns:
            Configured model adapter
        """
        model_id = model_id or cls._fallback_chain.openrouter[0]
        
        # Determine provider
        if provider:
            resolved_provider = Provider(provider)
        else:
            resolved_provider = cls.resolve_provider(model_id)
        
        # Generate cache key
        cache_key = f"{resolved_provider.value}:{model_id}"
        
        # Check circuit breaker for the provider
        if use_fallback:
            cb = get_circuit_breaker(resolved_provider.value)
            # Sync check - in async context use async version
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # We're in async context, can't check synchronously
                # The caller should use get_chat_model_async instead
            except RuntimeError:
                # No running loop, safe to check
                if asyncio.run(cb.should_use_fallback()):
                    # Get fallback for this provider
                    fallbacks = getattr(cls._fallback_chain, resolved_provider.value, ())
                    if len(fallbacks) > 1:
                        model_id = fallbacks[1]  # Use secondary
                        cache_key = f"{resolved_provider.value}:{model_id}"
                        logger.info(f"Using fallback: {model_id}")
        
        # Return cached instance or create new
        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls._create_adapter(model_id, resolved_provider)
        
        return cls._instances[cache_key]
    
    @classmethod
    def _create_adapter(cls, model_id: str, provider: Provider) -> BaseChatModel:
        """Create the appropriate adapter for a model."""
        if provider == Provider.SILICONFLOW:
            return SiliconFlowAdapter(model_id=model_id)
        elif provider == Provider.AZURE:
            from .azure import AzureOpenAIAdapter
            return AzureOpenAIAdapter(model_id=model_id)
        elif provider == Provider.OPENAI:
            from .direct_openai import DirectOpenAIAdapter
            return DirectOpenAIAdapter(model_id=model_id)
        elif provider == Provider.ANTHROPIC:
            from .direct_anthropic import DirectAnthropicAdapter
            return DirectAnthropicAdapter(model_id=model_id)
        else:
            # Default: OpenRouter
            return OpenRouterAdapter(model_id=model_id)
    
    @classmethod
    def get_embedding_model(cls):
        """Get the configured embedding model."""
        from .embedding import GoogleEmbeddingAdapter
        return GoogleEmbeddingAdapter()
    
    @classmethod
    async def record_success(cls, model_id: str, provider: Optional[str] = None):
        """Record successful request for circuit breaker."""
        provider = provider or cls.resolve_provider(model_id).value
        cb = get_circuit_breaker(provider)
        await cb.record_success()
    
    @classmethod
    async def record_failure(cls, model_id: str, provider: Optional[str] = None):
        """Record failed request for circuit breaker."""
        provider = provider or cls.resolve_provider(model_id).value
        cb = get_circuit_breaker(provider)
        await cb.record_failure()


# Convenience function
def get_chat_model(
    model_id: Optional[str] = None,
    provider: Optional[str] = None
) -> BaseChatModel:
    """Get a chat model instance."""
    return ModelFactory.get_chat_model(model_id)
```

---

## Configuration

### `configs/providers/openrouter.yaml`

```yaml
# OpenRouter Configuration
openrouter:
  api_base: "https://openrouter.ai/api/v1"
  env_key: "OPENROUTER_API_KEY"
  timeout: 60
  
  # Default model selection
  default_model: "openai/gpt-4o"
  
  # Available models with metadata
  models:
    # OpenAI Models
    openai/gpt-4o:
      context_window: 128000
      supports_structured_output: true
      supports_vision: true
      cost_input_per_1k: 0.0025
      cost_output_per_1k: 0.01
    
    openai/gpt-4o-mini:
      context_window: 128000
      supports_structured_output: true
      cost_input_per_1k: 0.00015
      cost_output_per_1k: 0.0006
    
    # Anthropic Models
    anthropic/claude-3.5-sonnet:
      context_window: 200000
      supports_structured_output: true
      supports_vision: true
      cost_input_per_1k: 0.003
      cost_output_per_1k: 0.015
    
    anthropic/claude-3-opus:
      context_window: 200000
      supports_structured_output: true
      cost_input_per_1k: 0.015
      cost_output_per_1k: 0.075
    
    # DeepSeek (Cost-efficient)
    deepseek/deepseek-chat:
      context_window: 64000
      supports_structured_output: true
      cost_input_per_1k: 0.00014
      cost_output_per_1k: 0.00028
    
    # Google
    google/gemini-pro-1.5:
      context_window: 1000000
      supports_structured_output: true
      cost_input_per_1k: 0.00125
      cost_output_per_1k: 0.005
```

### `configs/providers/siliconflow.yaml`

```yaml
# SiliconFlow Configuration
siliconflow:
  api_base: "https://api.siliconflow.cn/v1"
  env_key: "SILICONFLOW_API_KEY"
  timeout: 60
  
  # Default model selection
  default_model: "deepseek-ai/DeepSeek-V3"
  
  # Best for: Asian users, Chinese language, cost-effective DeepSeek
  
  # Available models with metadata
  models:
    # DeepSeek Models (Most Cost-Effective)
    deepseek-ai/DeepSeek-V3:
      context_window: 64000
      supports_structured_output: true
      cost_input_per_1m: 0.07
      cost_output_per_1m: 0.14
    
    deepseek-ai/DeepSeek-Chat:
      context_window: 64000
      supports_structured_output: true
      cost_input_per_1m: 0.07
      cost_output_per_1m: 0.14
    
    # Qwen Models (Alibaba)
    Qwen/Qwen2.5-72B-Instruct:
      context_window: 128000
      supports_structured_output: true
      cost_input_per_1m: 0.28
      cost_output_per_1m: 0.28
    
    Qwen/Qwen2.5-32B-Instruct:
      context_window: 128000
      supports_structured_output: true
      cost_input_per_1m: 0.14
      cost_output_per_1m: 0.14
    
    # Yi Models (01.AI)
    01-ai/Yi-1.5-34B-Chat:
      context_window: 32000
      supports_structured_output: true
      cost_input_per_1m: 0.14
      cost_output_per_1m: 0.14
    
    # GLM Models (Tsinghua)
    THUDM/glm-4-9b-chat:
      context_window: 128000
      supports_structured_output: true
      cost_input_per_1m: 0.07
      cost_output_per_1m: 0.07
    
    # Llama via SiliconFlow
    meta-llama/Llama-3.1-70B-Instruct:
      context_window: 128000
      supports_structured_output: true
      cost_input_per_1m: 0.28
      cost_output_per_1m: 0.28
```

### `configs/providers/fallback.yaml`

```yaml
# Multi-Provider Fallback Chain Configuration
fallback:
  # Per-provider fallback chains
  openrouter:
    models:
      - "openai/gpt-4o"              # Primary
      - "anthropic/claude-3.5-sonnet" # Secondary
      - "deepseek/deepseek-chat"      # Tertiary (cheap)
    circuit_breaker:
      failure_threshold: 3
      recovery_timeout_seconds: 300
  
  siliconflow:
    models:
      - "deepseek-ai/DeepSeek-V3"     # Primary
      - "Qwen/Qwen2.5-72B-Instruct"   # Secondary
      - "THUDM/glm-4-9b-chat"         # Tertiary (cheap)
    circuit_breaker:
      failure_threshold: 3
      recovery_timeout_seconds: 300
  
  # Cross-provider fallback (when all models in a provider fail)
  cross_provider:
    enabled: true
    chain:
      - provider: openrouter
        model: "openai/gpt-4o"
      - provider: siliconflow
        model: "deepseek-ai/DeepSeek-V3"
      - provider: openrouter
        model: "deepseek/deepseek-chat"
```

---

## Usage Examples

### Basic Generation (OpenRouter)

```python
from insights.adapters.models.factory import get_chat_model

# OpenRouter (auto-detected from prefix)
model = get_chat_model("openai/gpt-4o")

response = await model.generate([
    {"role": "system", "content": "You are a financial analyst."},
    {"role": "user", "content": "Summarize the key risks for AAPL."}
])

print(response)
```

### Using SiliconFlow

```python
from insights.adapters.models.factory import get_chat_model

# SiliconFlow (auto-detected from deepseek-ai/, Qwen/, etc.)
model = get_chat_model("deepseek-ai/DeepSeek-V3")

response = await model.generate([
    {"role": "system", "content": "你是一个金融分析师。"},  # Chinese prompt
    {"role": "user", "content": "分析阿里巴巴的主要风险因素。"}
])

# Or explicitly specify provider
model = get_chat_model("Qwen/Qwen2.5-72B-Instruct", provider="siliconflow")
```

### Structured Output

```python
from pydantic import BaseModel
from typing import List

class RiskAnalysis(BaseModel):
    ticker: str
    top_risks: List[str]
    overall_sentiment: str
    confidence: float

model = get_chat_model()

response = await model.generate(
    messages=[
        {"role": "system", "content": "Extract risk analysis as JSON."},
        {"role": "user", "content": filing_text}
    ],
    response_format=RiskAnalysis
)

# Response is guaranteed to be valid JSON matching the schema
import json
data = RiskAnalysis.model_validate_json(response)
```

### Streaming

```python
model = get_chat_model()

async for chunk in model.stream([
    {"role": "user", "content": "Write a detailed risk report."}
]):
    print(chunk, end="", flush=True)
```

### Per-Provider Circuit Breaker

```python
from insights.adapters.models.factory import ModelFactory

# Each provider has its own circuit breaker
try:
    model = ModelFactory.get_chat_model("openai/gpt-4o")
    response = await model.generate(messages)
    await ModelFactory.record_success("openai/gpt-4o")
except Exception as e:
    await ModelFactory.record_failure("openai/gpt-4o")
    # OpenRouter circuit breaker will trip after 3 failures
    # Next call will use fallback: anthropic/claude-3.5-sonnet
    raise

# SiliconFlow models have separate circuit breaker
try:
    model = ModelFactory.get_chat_model("deepseek-ai/DeepSeek-V3")
    response = await model.generate(messages)
    await ModelFactory.record_success("deepseek-ai/DeepSeek-V3")
except Exception as e:
    await ModelFactory.record_failure("deepseek-ai/DeepSeek-V3")
    # SiliconFlow circuit breaker is independent
    raise
```

### Expert-Specific Model Selection

```python
# Each expert can use optimal provider/model
risk_expert_model = get_chat_model("openai/gpt-4o")  # OpenRouter
patent_expert_model = get_chat_model("Qwen/Qwen2.5-72B-Instruct")  # SiliconFlow
tech_expert_model = get_chat_model("anthropic/claude-3.5-sonnet")  # OpenRouter

# Cost comparison per 1M tokens:
# OpenRouter GPT-4o: $2.50 input / $10 output
# SiliconFlow DeepSeek-V3: $0.07 input / $0.14 output  (35x cheaper!)
# Use SiliconFlow for bulk processing, OpenRouter for precision
```