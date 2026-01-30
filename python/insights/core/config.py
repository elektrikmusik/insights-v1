"""
Core configuration for InSights-ai.

Loads settings from environment variables with Pydantic validation.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─────────────────────────────────────────────────────────────
    # Application
    # ─────────────────────────────────────────────────────────────
    ENV: str = "development"
    DEBUG: bool = True
    APP_URL: str = "http://localhost:8000"
    LOG_LEVEL: str = "INFO"

    # ─────────────────────────────────────────────────────────────
    # Supabase
    # ─────────────────────────────────────────────────────────────
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    # ─────────────────────────────────────────────────────────────
    # LLM Providers
    # ─────────────────────────────────────────────────────────────
    # OpenRouter (primary aggregator)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL: str = "openai/gpt-4o"

    # SiliconFlow (Chinese models)
    SILICONFLOW_API_KEY: str = ""
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"

    # Google AI (embeddings)
    GOOGLE_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-004"

    # Azure OpenAI (optional enterprise)
    AZURE_OPENAI_API_KEY: str | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None

    # ─────────────────────────────────────────────────────────────
    # MCP Servers
    # ─────────────────────────────────────────────────────────────
    MCP_SEC_URL: str = "http://localhost:8080/sse"
    MCP_BRAVE_URL: str = ""
    SEC_API_USER_AGENT: str = "InSights/1.0"

    # ─────────────────────────────────────────────────────────────
    # Redis & Celery
    # ─────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str | None = None  # Falls back to REDIS_URL
    CELERY_RESULT_BACKEND: str | None = None  # Falls back to REDIS_URL

    # ─────────────────────────────────────────────────────────────
    # HuggingFace
    # ─────────────────────────────────────────────────────────────
    HF_TOKEN: str = ""  # HuggingFace API token for Inference API

    # ─────────────────────────────────────────────────────────────
    # FinBERT
    # ─────────────────────────────────────────────────────────────
    # Modes: "local" (load model), "http" (remote service), "hf_inference" (HF API)
    FINBERT_MODE: str = "hf_inference"
    FINBERT_SERVICE_URL: str = "http://localhost:8001"
    FINBERT_BATCH_SIZE: int = 16

    # ─────────────────────────────────────────────────────────────
    # Webhook
    # ─────────────────────────────────────────────────────────────
    WEBHOOK_SECRET: str = ""

    @property
    def celery_broker(self) -> str:
        """Get Celery broker URL."""
        return self.CELERY_BROKER_URL or self.REDIS_URL

    @property
    def celery_backend(self) -> str:
        """Get Celery result backend URL."""
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENV.lower() == "production"


# Global settings instance
settings = Settings()
