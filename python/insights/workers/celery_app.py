"""
Celery application configuration.
"""
from celery import Celery
from insights.core.config import settings

# Create Celery app
celery_app = Celery(
    "insights",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# Configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Time limits
    task_time_limit=600,        # 10 minutes hard limit
    task_soft_time_limit=540,   # 9 minutes soft limit
    
    # Retry policy
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Prefetch
    worker_prefetch_multiplier=1,  # Fair distribution
    
    # Results
    result_expires=3600,  # 1 hour
    
    # Task routing
    task_routes={
        "insights.workers.tasks.analyze_risk_drift": {"queue": "analysis"},
    },
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Autodiscover tasks
celery_app.autodiscover_tasks(["insights.workers"])
