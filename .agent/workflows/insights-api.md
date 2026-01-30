---
description: Add a new API endpoint to the FastAPI server
---

# Add API Endpoint Workflow

Reference: `docs/specs/03_API_SPECIFICATION.md`

## Required Information

1. **Endpoint Path** (e.g., `/api/v1/analyze`)
2. **HTTP Method** (GET, POST, PUT, DELETE)
3. **Request Schema** (Pydantic model for input)
4. **Response Schema** (Pydantic model for output)
5. **Authentication Required?** (Supabase JWT)

## Implementation Steps

1. **Define Request/Response Models**
   - Add to `python/insights/core/types.py` or create route-specific file
   
   ```python
   class AnalyzeRequest(BaseModel):
       ticker: str
       years: List[int] = [2024, 2023]
   
   class AnalyzeResponse(BaseModel):
       job_id: str
       status: str
   ```

2. **Create Route File**
   - Create `python/insights/server/api/{name}.py`
   - Use FastAPI router with prefix
   
   ```python
   from fastapi import APIRouter, Depends, HTTPException
   
   router = APIRouter(prefix="/api/v1/{name}", tags=["{name}"])
   
   @router.post("/", response_model=Response)
   async def create(request: Request):
       # Implementation
       pass
   ```

3. **Add Authentication (if needed)**
   ```python
   from insights.server.middleware.auth import get_current_user
   
   @router.post("/")
   async def protected_route(
       request: Request,
       user: dict = Depends(get_current_user)
   ):
       pass
   ```

4. **Register Router**
   - Update `python/insights/server/main.py`
   
   ```python
   from insights.server.api.{name} import router as {name}_router
   app.include_router({name}_router)
   ```

5. **Write Tests**
   - Create `python/tests/integration/api/test_{name}.py`
   - Use FastAPI TestClient

## SSE Streaming Endpoint

For streaming responses:

```python
from fastapi.responses import StreamingResponse

@router.get("/stream/{job_id}")
async def stream_results(job_id: str):
    async def event_generator():
        async for event in get_job_events(job_id):
            yield f"data: {event.model_dump_json()}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

## Verification

// turbo
- Run `uv run pytest python/tests/integration/api/ -v`
- Test with `curl http://localhost:8000/api/v1/{name}`
