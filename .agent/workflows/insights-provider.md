---
description: Add a new LLM provider adapter
---

# Add LLM Provider Workflow

Reference: `docs/specs/07_MODEL_MANAGEMENT.md`

## Required Information

1. **Provider Name** (e.g., `azure`, `together`)
2. **API Base URL** (e.g., `https://api.together.xyz/v1`)
3. **Environment Variable** for API key
4. **Supported Models** and their costs

## Implementation Steps

1. **Create Adapter File**
   - Create `python/insights/adapters/models/{provider}.py`
   - Extend `BaseChatModel` interface
   
   ```python
   from insights.adapters.models.interface import BaseChatModel
   
   class {Provider}Adapter(BaseChatModel):
       API_BASE = "https://api.{provider}.com/v1"
       
       def __init__(self, model_id: str, api_key: str = None):
           self._model_id = model_id
           self._api_key = api_key or settings.{PROVIDER}_API_KEY
       
       @property
       def model_id(self) -> str:
           return self._model_id
       
       @property
       def provider(self) -> str:
           return "{provider}"
       
       async def generate(self, messages, **kwargs) -> str:
           # Implement API call
           pass
       
       async def stream(self, messages, **kwargs):
           # Implement streaming
           pass
   ```

2. **Update Provider Enum**
   - Edit `python/insights/adapters/models/factory.py`
   - Add to `Provider` enum
   - Add prefix mappings to `PROVIDER_PREFIXES`

3. **Update Factory**
   ```python
   @classmethod
   def _create_adapter(cls, model_id: str, provider: Provider):
       if provider == Provider.{PROVIDER}:
           from .{provider} import {Provider}Adapter
           return {Provider}Adapter(model_id=model_id)
       ...
   ```

4. **Create Config File**
   - Create `configs/providers/{provider}.yaml`
   
   ```yaml
   {provider}:
     api_base: "https://api.{provider}.com/v1"
     env_key: "{PROVIDER}_API_KEY"
     timeout: 60
     models:
       model-name:
         context_window: 128000
         cost_input_per_1k: 0.01
         cost_output_per_1k: 0.03
   ```

5. **Update Fallback Config**
   - Add provider to `configs/providers/fallback.yaml`

6. **Write Tests**
   ```python
   @pytest.mark.asyncio
   async def test_{provider}_generate():
       with patch("httpx.AsyncClient.post") as mock:
           mock.return_value.json.return_value = {
               "choices": [{"message": {"content": "response"}}]
           }
           adapter = {Provider}Adapter("model-id")
           result = await adapter.generate([{"role": "user", "content": "hi"}])
           assert result == "response"
   ```

7. **Update Settings**
   - Add `{PROVIDER}_API_KEY` to `python/insights/core/config.py`

## Verification

// turbo
- Run `uv run pytest python/tests/unit/adapters/models/ -v`
- Test manual generation with new provider
