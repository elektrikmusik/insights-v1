#!/bin/bash
# Initialize the InSights-ai project structure
# Usage: .agent/scripts/init-insights.sh

set -e

echo "🚀 Initializing InSights-ai project structure..."

# Create main Python package structure
mkdir -p python/insights/{adapters,services,experts,agents,core,server,workers}
mkdir -p python/insights/adapters/{mcp,db,models}
mkdir -p python/insights/services/{risk,filing,sentiment,report}
mkdir -p python/insights/agents/orchestrator
mkdir -p python/insights/server/{api,middleware}
mkdir -p python/tests/{unit,integration}
mkdir -p python/tests/unit/{services,experts,adapters}

# Create config directories
mkdir -p configs/{agents,providers,prompts/experts}

# Create __init__.py files
find python/insights -type d -exec touch {}/__init__.py \;
touch python/tests/__init__.py
touch python/tests/unit/__init__.py
touch python/tests/integration/__init__.py

# Create placeholder files
cat > python/insights/core/config.py << 'EOF'
"""Configuration loader for InSights-ai."""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    
    # LLM Providers
    OPENROUTER_API_KEY: str = ""
    SILICONFLOW_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    
    # MCP
    MCP_SERVER_URL: str = "http://localhost:8080"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # App
    APP_URL: str = "https://insights.ai"
    DEBUG: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()
EOF

cat > python/insights/core/types.py << 'EOF'
"""Shared Pydantic models for InSights-ai."""
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ExpertResult(BaseModel):
    """Result from an expert analysis."""
    expert_id: str
    findings: str
    confidence: float
    sources: List[str] = []
    timestamp: datetime = None

class SynthesizedReport(BaseModel):
    """Final synthesized report from Orchestrator."""
    company: str
    summary: str
    expert_findings: List[ExpertResult]
    recommendations: List[str]
    risk_level: str
    generated_at: datetime
EOF

echo "✅ Project structure initialized!"
echo ""
echo "Next steps:"
echo "1. Run: cd python && uv init"
echo "2. Add dependencies to pyproject.toml"
echo "3. Create .env file with API keys"
echo "4. Run: /insights-build to start implementation"
