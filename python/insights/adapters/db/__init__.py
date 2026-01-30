"""
Database adapters for InSights-ai.
"""
from insights.adapters.db.manager import (
    AuditLogCreate,
    # Data models
    CompanyCreate,
    CompanyRecord,
    DBManager,
    FilingCreate,
    FilingRecord,
    JobCreate,
    JobRecord,
    RiskDriftCreate,
    RiskFactorCreate,
    RiskFactorRecord,
    VectorSearchResult,
    get_db,
)

__all__ = [
    "DBManager",
    "get_db",
    "CompanyCreate",
    "CompanyRecord",
    "FilingCreate",
    "FilingRecord",
    "RiskFactorCreate",
    "RiskFactorRecord",
    "RiskDriftCreate",
    "JobCreate",
    "JobRecord",
    "AuditLogCreate",
    "VectorSearchResult",
]
