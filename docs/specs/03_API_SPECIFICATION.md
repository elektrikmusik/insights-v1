# 03. API Specification

## Overview

The InSights-ai API is a RESTful service built with FastAPI. It supports:
- Async job submission for long-running analysis
- Server-Sent Events (SSE) for real-time progress updates
- Webhook callbacks for job completion
- Supabase JWT authentication

---

## Base Configuration

| Setting | Value |
|---------|-------|
| Base URL | `https://api.insights.ai/api/v1` |
| Authentication | Bearer JWT (Supabase Auth) |
| Content Type | `application/json` |
| Documentation | Swagger UI at `/docs` |

---

## Authentication

All endpoints (except health checks) require a valid Supabase JWT.

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Middleware Implementation:**

```python
# insights/server/middleware/auth.py
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer
from supabase import create_client

security = HTTPBearer()

async def verify_jwt(request: Request):
    """Validate Supabase JWT, attach user to request state."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    token = auth_header.split(" ")[1]
    
    try:
        # Verify with Supabase
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
        user = supabase.auth.get_user(token)
        request.state.user = user
        request.state.user_id = user.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")
```

---

## Endpoints

### 1. Research Analysis

#### Submit Analysis Job

**POST** `/api/v1/research/analyze`

Submits a long-running analysis job to the queue. Returns immediately with a job ID.

**Request:**
```json
{
  "ticker": "AAPL",
  "analysis_type": "risk_drift",
  "years": [2024, 2023],
  "options": {
    "include_sentiment": true,
    "generate_heatmap": true,
    "depth": "deep"
  },
  "webhook_url": "https://your-app.com/webhook/job-complete"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ticker` | string | Yes | Stock ticker symbol |
| `analysis_type` | string | Yes | `risk_drift`, `sentiment_trend`, `full_research` |
| `years` | array[int] | No | Fiscal years to compare (default: last 2) |
| `options.include_sentiment` | bool | No | Run FinBERT analysis (default: true) |
| `options.generate_heatmap` | bool | No | Generate heatmap data (default: true) |
| `options.depth` | string | No | `shallow` (LLM only) or `deep` (FinBERT + LLM) |
| `webhook_url` | string | No | URL to POST results when complete |

**Response (202 Accepted):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Analysis job submitted successfully",
  "estimated_duration_seconds": 120,
  "stream_url": "/api/v1/research/stream/550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-01-29T12:00:00Z"
}
```

**Error Responses:**

| Status | Code | Description |
|--------|------|-------------|
| 400 | `INVALID_TICKER` | Ticker not found in SEC database |
| 400 | `INVALID_YEARS` | Years must be within available filing range |
| 401 | `UNAUTHORIZED` | Invalid or missing JWT |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests |

---

#### Stream Job Progress (SSE)

**GET** `/api/v1/research/stream/{job_id}`

Server-Sent Events stream providing real-time updates on job progress.

**Event Types:**

```typescript
// Status update
{
  "type": "STATUS",
  "message": "Fetching SEC filing for AAPL 2024...",
  "progress": 20
}

// Tool invocation
{
  "type": "TOOL_USE",
  "tool": "sec-edgar-mcp",
  "action": "get_filing_content",
  "ticker": "AAPL",
  "year": 2024
}

// Progress update
{
  "type": "PROGRESS",
  "step": "sentiment_analysis",
  "current": 50,
  "total": 100,
  "message": "Analyzed 50/100 risk factor chunks"
}

// Partial result
{
  "type": "PARTIAL_RESULT",
  "data": {
    "risk_count": 25,
    "new_risks": 3,
    "critical_drifts": 2
  }
}

// Error (non-fatal)
{
  "type": "ERROR",
  "code": "FINBERT_TIMEOUT",
  "message": "FinBERT service timed out. Retrying...",
  "recoverable": true
}

// Final result
{
  "type": "RESULT",
  "job_id": "550e8400-...",
  "status": "completed",
  "data": {
    "heatmap": { ... },
    "drifts": [ ... ],
    "report_id": "...",
    "markdown_summary": "## Risk Analysis Report..."
  }
}

// Fatal error
{
  "type": "FAILED",
  "code": "MCP_CONNECTION_FAILED",
  "message": "Unable to connect to SEC EDGAR server",
  "recoverable": false
}
```

**Implementation:**

```python
# insights/server/api/research.py
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
import asyncio

router = APIRouter(prefix="/research")

@router.get("/stream/{job_id}")
async def stream_job(job_id: str, request: Request):
    """SSE stream for job progress."""
    
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            
            # Fetch job status from Redis or DB
            job = await get_job_status(job_id)
            
            if job.status == "queued":
                yield {
                    "event": "STATUS",
                    "data": json.dumps({
                        "type": "STATUS",
                        "message": "Job is queued...",
                        "progress": 0
                    })
                }
            
            elif job.status == "processing":
                yield {
                    "event": "PROGRESS",
                    "data": json.dumps({
                        "type": "PROGRESS",
                        "step": job.current_step,
                        "progress": job.progress,
                        "message": job.status_message
                    })
                }
            
            elif job.status == "completed":
                yield {
                    "event": "RESULT",
                    "data": json.dumps({
                        "type": "RESULT",
                        "job_id": job_id,
                        "status": "completed",
                        "data": job.result_summary
                    })
                }
                break
            
            elif job.status == "failed":
                yield {
                    "event": "FAILED",
                    "data": json.dumps({
                        "type": "FAILED",
                        "code": job.error_code,
                        "message": job.error_message
                    })
                }
                break
            
            await asyncio.sleep(1)
    
    return EventSourceResponse(event_generator())
```

---

#### Get Job Status

**GET** `/api/v1/research/jobs/{job_id}`

**Response:**
```json
{
  "job_id": "550e8400-...",
  "status": "completed",
  "progress": 100,
  "current_step": "Report generated",
  "created_at": "2024-01-29T12:00:00Z",
  "started_at": "2024-01-29T12:00:05Z",
  "completed_at": "2024-01-29T12:02:30Z",
  "duration_seconds": 145,
  "result_summary": {
    "ticker": "AAPL",
    "risks_analyzed": 25,
    "new_risks": 3,
    "critical_drifts": 2,
    "report_id": "abc123..."
  }
}
```

---

#### List User Jobs

**GET** `/api/v1/research/jobs`

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | string | all | Filter by status: `queued`, `processing`, `completed`, `failed` |
| `limit` | int | 20 | Max results (max: 100) |
| `offset` | int | 0 | Pagination offset |

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "...",
      "ticker": "AAPL",
      "analysis_type": "risk_drift",
      "status": "completed",
      "created_at": "..."
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

---

### 2. Data Access

#### Get Filings

**GET** `/api/v1/data/filings`

**Query Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `ticker` | string | Yes | Stock ticker |
| `form_type` | string | No | Filter by form type (10-K, 10-Q) |
| `limit` | int | No | Max results (default: 10) |

**Response:**
```json
{
  "ticker": "AAPL",
  "filings": [
    {
      "id": "...",
      "accession_number": "0000320193-23-000077",
      "form_type": "10-K",
      "filing_date": "2023-11-03",
      "fiscal_year": 2023,
      "has_embeddings": true,
      "risk_factor_count": 28
    }
  ]
}
```

---

#### Vector Search

**POST** `/api/v1/data/search`

**Request:**
```json
{
  "query": "What are the supply chain risks?",
  "ticker": "AAPL",
  "filing_id": null,
  "limit": 10,
  "threshold": 0.7
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "...",
      "filing_id": "...",
      "title": "Supply Chain and Manufacturing Risks",
      "content": "The Company's products are...",
      "similarity": 0.89,
      "filing_date": "2023-11-03"
    }
  ],
  "query_embedding_used": true,
  "total_results": 5
}
```

---

#### Get Report

**GET** `/api/v1/data/reports/{report_id}`

**Response:**
```json
{
  "id": "...",
  "ticker": "AAPL",
  "report_type": "risk_drift",
  "title": "AAPL Risk Drift Analysis (2024 vs 2023)",
  "markdown_content": "## Executive Summary...",
  "created_at": "...",
  "parameters": {
    "years": [2024, 2023],
    "include_sentiment": true
  },
  "summary": {
    "new_risks": 3,
    "removed_risks": 2,
    "critical_drifts": 5
  }
}
```

---

### 3. Heatmap Data

#### Get Risk Heatmap

**GET** `/api/v1/data/heatmap/{ticker}`

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `comparison_year` | int | Current - 1 | Previous year for comparison |

**Response:**
```json
{
  "heatmap_title": "AAPL Risk Profile Evolution (2024 vs 2023)",
  "generated_at": "2024-01-29T12:00:00Z",
  "ticker": "AAPL",
  "zones": {
    "critical_red": [
      {
        "risk_id": "supply_chain_concentration",
        "title": "Concentration of Supply Chain in China",
        "current_rank": 2,
        "rank_delta": 8,
        "heat_score": 90,
        "summary": "Elevated from #10 to #2, indicating increased concern"
      }
    ],
    "warning_orange": [ ... ],
    "new_blue": [ ... ],
    "stable_gray": [ ... ]
  },
  "statistics": {
    "total_risks": 28,
    "critical_count": 3,
    "warning_count": 5,
    "new_count": 4,
    "removed_count": 2
  }
}
```

---

### 4. Health & Monitoring

#### Health Check

**GET** `/api/v1/health`

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-29T12:00:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "sec_mcp": "healthy",
    "finbert": "healthy"
  }
}
```

#### Readiness Check

**GET** `/api/v1/ready`

Used by Kubernetes/Cloud Run for readiness probes.

---

### 5. Webhook Callback

When a job completes (or fails), if `webhook_url` was provided, the system will POST:

**POST** `{webhook_url}`

```json
{
  "event": "job.completed",
  "job_id": "550e8400-...",
  "status": "completed",
  "ticker": "AAPL",
  "analysis_type": "risk_drift",
  "result_summary": {
    "report_id": "...",
    "risks_analyzed": 25,
    "critical_drifts": 2
  },
  "completed_at": "2024-01-29T12:02:30Z"
}
```

For failures:
```json
{
  "event": "job.failed",
  "job_id": "550e8400-...",
  "status": "failed",
  "error_code": "MCP_CONNECTION_FAILED",
  "error_message": "Unable to fetch SEC filing: Connection timeout",
  "failed_at": "2024-01-29T12:01:00Z"
}
```

---

## Error Response Format

All error responses follow this structure:

```json
{
  "error": {
    "code": "INVALID_TICKER",
    "message": "Ticker 'XXXX' not found in SEC database",
    "details": {
      "ticker": "XXXX",
      "suggestion": "Check ticker symbol or try CIK number"
    }
  },
  "request_id": "abc123...",
  "timestamp": "2024-01-29T12:00:00Z"
}
```

---

## Rate Limiting

| Tier | Requests/Minute | Concurrent Jobs |
|------|-----------------|-----------------|
| Free | 10 | 1 |
| Pro | 60 | 5 |
| Enterprise | Unlimited | 20 |

Rate limit headers:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1706529600
```

---

## OpenAPI Specification

Full OpenAPI 3.0 spec available at:
- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **OpenAPI JSON:** `/openapi.json`