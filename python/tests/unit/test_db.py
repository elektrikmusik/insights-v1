"""Tests for database adapter."""
from datetime import date
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from insights.adapters.db import (
    AuditLogCreate,
    CompanyCreate,
    DBManager,
    FilingCreate,
    JobCreate,
    RiskFactorCreate,
)
from insights.core.errors import DatabaseQueryError


class TestDBManagerInit:
    """Test DBManager initialization."""

    def test_uses_settings_by_default(self):
        """Should use settings for URL and key."""
        with patch("insights.adapters.db.manager.settings") as mock_settings:
            mock_settings.SUPABASE_URL = "https://test.supabase.co"
            mock_settings.SUPABASE_SERVICE_ROLE_KEY = "test-key"

            manager = DBManager()
            assert manager._url == "https://test.supabase.co"
            assert manager._key == "test-key"

    def test_accepts_custom_credentials(self):
        """Should allow custom URL and key."""
        manager = DBManager(
            supabase_url="https://custom.supabase.co",
            supabase_key="custom-key",
        )
        assert manager._url == "https://custom.supabase.co"
        assert manager._key == "custom-key"

    def test_singleton_pattern(self):
        """get_instance should return same instance."""
        # Reset singleton
        DBManager._instance = None

        instance1 = DBManager.get_instance()
        instance2 = DBManager.get_instance()

        assert instance1 is instance2

        # Cleanup
        DBManager._instance = None


class TestCompanyOperations:
    """Test company CRUD operations."""

    @pytest.fixture
    def mock_manager(self):
        """Create manager with mocked client."""
        manager = DBManager(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        manager._client = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_get_or_create_company(self, mock_manager):
        """Should upsert company and return UUID."""
        company_id = str(uuid4())
        mock_manager._client.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": company_id, "ticker": "AAPL", "cik": "0000320193"}]
        )

        result = await mock_manager.get_or_create_company(
            CompanyCreate(ticker="AAPL", cik="0000320193", name="Apple Inc.")
        )

        assert isinstance(result, UUID)
        assert str(result) == company_id
        mock_manager._client.table.assert_called_with("companies")

    @pytest.mark.asyncio
    async def test_get_or_create_company_error(self, mock_manager):
        """Should raise DatabaseQueryError on failure."""
        mock_manager._client.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[]
        )

        with pytest.raises(DatabaseQueryError):
            await mock_manager.get_or_create_company(
                CompanyCreate(ticker="AAPL", cik="0000320193")
            )

    @pytest.mark.asyncio
    async def test_get_company_by_ticker(self, mock_manager):
        """Should return company record by ticker."""
        company_id = str(uuid4())
        mock_manager._client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[{"id": company_id, "ticker": "AAPL", "cik": "0000320193", "name": "Apple Inc."}]
        )

        result = await mock_manager.get_company_by_ticker("aapl")  # lowercase to test case handling

        assert result is not None
        assert result.ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_get_company_by_ticker_not_found(self, mock_manager):
        """Should return None if not found."""
        mock_manager._client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[]
        )

        result = await mock_manager.get_company_by_ticker("NOTFOUND")

        assert result is None


class TestFilingOperations:
    """Test filing CRUD operations."""

    @pytest.fixture
    def mock_manager(self):
        """Create manager with mocked client."""
        manager = DBManager(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        manager._client = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_save_filing(self, mock_manager):
        """Should upsert filing and return UUID."""
        filing_id = str(uuid4())
        company_id = uuid4()
        mock_manager._client.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": filing_id}]
        )

        result = await mock_manager.save_filing(
            FilingCreate(
                company_id=company_id,
                accession_number="0001193125-24-012345",
                form_type="10-K",
                filing_date=date(2024, 2, 15),
                fiscal_year=2023,
            )
        )

        assert isinstance(result, UUID)
        assert str(result) == filing_id

    @pytest.mark.asyncio
    async def test_get_filings_by_company(self, mock_manager):
        """Should return list of filings."""
        company_id = uuid4()
        filing_id = str(uuid4())
        mock_manager._client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[{
                "id": filing_id,
                "company_id": str(company_id),
                "accession_number": "0001193125-24-012345",
                "form_type": "10-K",
                "filing_date": "2024-02-15",
                "fiscal_year": 2023,
            }]
        )

        result = await mock_manager.get_filings_by_company(company_id)

        assert len(result) == 1
        assert result[0].form_type == "10-K"


class TestRiskFactorOperations:
    """Test risk factor CRUD operations."""

    @pytest.fixture
    def mock_manager(self):
        """Create manager with mocked client."""
        manager = DBManager(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        manager._client = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_save_risk_factors(self, mock_manager):
        """Should batch insert risk factors."""
        filing_id = uuid4()
        factor_ids = [str(uuid4()), str(uuid4())]
        mock_manager._client.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": factor_ids[0]}, {"id": factor_ids[1]}]
        )

        factors = [
            RiskFactorCreate(
                filing_id=filing_id,
                title="Cybersecurity Risk",
                content="We face significant cybersecurity threats...",
                rank=1,
            ),
            RiskFactorCreate(
                filing_id=filing_id,
                title="Competition Risk",
                content="We operate in highly competitive markets...",
                rank=2,
            ),
        ]

        result = await mock_manager.save_risk_factors(factors)

        assert len(result) == 2
        mock_manager._client.table.assert_called_with("risk_factors")

    @pytest.mark.asyncio
    async def test_save_risk_factors_empty_list(self, mock_manager):
        """Should return empty list for empty input."""
        result = await mock_manager.save_risk_factors([])

        assert result == []
        mock_manager._client.table.assert_not_called()


class TestJobOperations:
    """Test job queue operations."""

    @pytest.fixture
    def mock_manager(self):
        """Create manager with mocked client."""
        manager = DBManager(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        manager._client = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_create_job(self, mock_manager):
        """Should create job and return UUID."""
        job_id = str(uuid4())
        mock_manager._client.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": job_id}]
        )

        result = await mock_manager.create_job(
            JobCreate(
                job_type="risk_analysis",
                request_payload={"ticker": "AAPL", "years": [2024, 2023]},
            )
        )

        assert isinstance(result, UUID)
        assert str(result) == job_id

    @pytest.mark.asyncio
    async def test_update_job_status(self, mock_manager):
        """Should update job status."""
        job_id = uuid4()
        mock_manager._client.table.return_value.update.return_value.eq.return_value.execute.return_value = MagicMock()

        await mock_manager.update_job_status(
            job_id=job_id,
            status="processing",
            progress=25,
            current_step="Fetching filings",
        )

        mock_manager._client.table.assert_called_with("job_queue")


class TestVectorSearch:
    """Test vector search operations."""

    @pytest.fixture
    def mock_manager(self):
        """Create manager with mocked client."""
        manager = DBManager(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        manager._client = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_vector_search_risk_factors(self, mock_manager):
        """Should call RPC for vector search."""
        result_id = str(uuid4())
        filing_id = str(uuid4())
        mock_manager._client.rpc.return_value.execute.return_value = MagicMock(
            data=[{
                "id": result_id,
                "filing_id": filing_id,
                "title": "Cybersecurity Risk",
                "content": "We face threats...",
                "rank": 1,
                "similarity": 0.85,
            }]
        )

        # Create a dummy 768-dim embedding
        query_embedding = [0.1] * 768

        result = await mock_manager.vector_search_risk_factors(
            query_embedding=query_embedding,
            limit=10,
            threshold=0.7,
        )

        assert len(result) == 1
        assert result[0].similarity == 0.85
        mock_manager._client.rpc.assert_called_with(
            "match_risk_factors",
            {
                "query_embedding": query_embedding,
                "match_count": 10,
                "similarity_threshold": 0.7,
            }
        )


class TestAuditLog:
    """Test audit log operations."""

    @pytest.fixture
    def mock_manager(self):
        """Create manager with mocked client."""
        manager = DBManager(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )
        manager._client = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_log_llm_usage(self, mock_manager):
        """Should insert audit log entry."""
        log_id = str(uuid4())
        mock_manager._client.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": log_id}]
        )

        result = await mock_manager.log_llm_usage(
            AuditLogCreate(
                operation="analysis",
                model_id="openai/gpt-4o",
                provider="openrouter",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=0.025,
                latency_ms=2500,
            )
        )

        assert isinstance(result, UUID)
        mock_manager._client.table.assert_called_with("audit_logs")


class TestDataModels:
    """Test Pydantic data models."""

    def test_company_create(self):
        """Should validate company creation data."""
        company = CompanyCreate(
            ticker="AAPL",
            cik="0000320193",
            name="Apple Inc.",
            sector="Technology",
        )
        assert company.ticker == "AAPL"

    def test_filing_create(self):
        """Should validate filing creation data."""
        filing = FilingCreate(
            company_id=uuid4(),
            accession_number="0001193125-24-012345",
            form_type="10-K",
            filing_date=date(2024, 2, 15),
            fiscal_year=2023,
        )
        assert filing.form_type == "10-K"

    def test_risk_factor_create(self):
        """Should validate risk factor creation data."""
        factor = RiskFactorCreate(
            filing_id=uuid4(),
            title="Competition Risk",
            content="We face intense competition...",
            rank=1,
            embedding=[0.1] * 768,
        )
        assert len(factor.embedding) == 768

    def test_job_create(self):
        """Should validate job creation data."""
        job = JobCreate(
            job_type="risk_analysis",
            request_payload={"ticker": "AAPL"},
        )
        assert job.job_type == "risk_analysis"
