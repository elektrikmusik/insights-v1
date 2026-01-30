"""Tests for core configuration."""
import pytest
from pydantic import ValidationError


class TestSettings:
    """Test Settings configuration."""

    def test_settings_loads(self):
        """Settings should load with defaults."""
        from insights.core.config import Settings

        # Create with minimal required fields (empty strings OK for testing)
        settings = Settings(
            SUPABASE_URL="https://test.supabase.co",
            SUPABASE_ANON_KEY="test-key",
            OPENROUTER_API_KEY="test-key",
            GOOGLE_API_KEY="test-key",
        )

        # Environment might be 'test' or 'development' depending on runner
        assert settings.ENV in ["development", "test"]
        assert settings.DEBUG is True
        assert settings.DEFAULT_MODEL == "openai/gpt-4o"

    def test_settings_env_property(self):
        """Settings should detect production environment."""
        from insights.core.config import Settings

        settings = Settings(
            ENV="production",
            SUPABASE_URL="https://test.supabase.co",
            SUPABASE_ANON_KEY="test-key",
            OPENROUTER_API_KEY="test-key",
            GOOGLE_API_KEY="test-key",
        )

        assert settings.is_production is True

    def test_celery_defaults_to_redis(self):
        """Celery broker should default to REDIS_URL."""
        from insights.core.config import Settings

        settings = Settings(
            REDIS_URL="redis://custom:6379/1",
            SUPABASE_URL="https://test.supabase.co",
            SUPABASE_ANON_KEY="test-key",
            OPENROUTER_API_KEY="test-key",
            GOOGLE_API_KEY="test-key",
        )

        assert settings.celery_broker == "redis://custom:6379/1"
        assert settings.celery_backend == "redis://custom:6379/1"


class TestTypes:
    """Test core types."""

    def test_expert_result_creation(self):
        """ExpertResult should validate correctly."""
        from insights.core.types import ExpertResult

        result = ExpertResult(
            expert_id="risk_analyst",
            findings="Test findings",
            confidence=0.85,
            sources=["10-K"],
        )

        assert result.expert_id == "risk_analyst"
        assert result.confidence == 0.85

    def test_expert_result_confidence_bounds(self):
        """ExpertResult confidence must be 0-1."""
        from insights.core.types import ExpertResult

        with pytest.raises(ValidationError):
            ExpertResult(
                expert_id="test",
                findings="test",
                confidence=1.5,  # Invalid: > 1
            )

    def test_job_status_enum(self):
        """JobStatus enum should have correct values."""
        from insights.core.types import JobStatus

        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"

    def test_risk_zone_enum(self):
        """RiskZone enum should have correct values."""
        from insights.core.types import RiskZone

        assert RiskZone.CRITICAL_RED == "critical_red"
        assert RiskZone.WARNING_ORANGE == "warning_orange"
        assert RiskZone.NEW_BLUE == "new_blue"
        assert RiskZone.STABLE_GRAY == "stable_gray"

    def test_analyze_request_validation(self):
        """AnalyzeRequest should validate ticker."""
        from insights.core.types import AnalyzeRequest

        request = AnalyzeRequest(ticker="AAPL", years=[2024, 2023])
        assert request.ticker == "AAPL"

        with pytest.raises(ValidationError):
            AnalyzeRequest(ticker="", years=[2024])  # Empty ticker


class TestErrors:
    """Test custom exceptions."""

    def test_insights_error_base(self):
        """InsightsError should be base exception."""
        from insights.core.errors import InsightsError

        error = InsightsError("Test error")
        assert str(error) == "Test error"
        assert error.code == "InsightsError"

    def test_mcp_tool_error(self):
        """MCPToolError should include tool name."""
        from insights.core.errors import MCPToolError

        error = MCPToolError("get_filing", "Connection refused")
        assert "get_filing" in str(error)
        assert error.tool_name == "get_filing"

    def test_expert_not_found_error(self):
        """ExpertNotFoundError should include expert ID."""
        from insights.core.errors import ExpertNotFoundError

        error = ExpertNotFoundError("unknown_expert")
        assert "unknown_expert" in str(error)
        assert error.expert_id == "unknown_expert"

    def test_job_not_found_error(self):
        """JobNotFoundError should include job ID."""
        from insights.core.errors import JobNotFoundError

        error = JobNotFoundError("job-123")
        assert "job-123" in str(error)
        assert error.job_id == "job-123"
