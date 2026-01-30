"""
Main FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime, UTC

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from insights.api.routes import analysis, stream
from insights.core.config import settings
from insights.api.middleware.auth import verify_jwt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("insights.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown logic.
    """
    logger.info("Starting InSights AI API...")
    # Initialize things if needed (e.g. verify MCP connection)
    yield
    logger.info("Shutting down InSights AI API...")


app = FastAPI(
    title="InSights AI",
    description="Financial Analysis Agent System API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
# Allow all for development convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
# Apply auth dependency globally or per-router. 
# For now, we'll apply it to the main research routers but leave health/ready public.
app.include_router(
    analysis.router, 
    prefix="/api/v1", 
    tags=["analysis"],
    dependencies=[Depends(verify_jwt)] if not settings.DEBUG else [] # skip in local debug
)
app.include_router(
    stream.router, 
    prefix="/api/v1", 
    tags=["streaming"],
    dependencies=[Depends(verify_jwt)] if not settings.DEBUG else []
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "service": "insights-ai"
    }


@app.get("/ready")
async def readiness_check():
    """Readiness probe for K8s/Cloud Run."""
    # Add check for DB connection here if needed
    return {"status": "ready"}


def main():
    """Entry point for the application."""
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
