"""
Health check endpoints
"""
from fastapi import APIRouter, Depends
from app.models import HealthResponse
from app.dependencies import get_model, get_model_config
from app.config import settings

router = APIRouter()

@router.get("/", response_model=HealthResponse)
def health_check(
    model=Depends(get_model),
    config=Depends(get_model_config)
):
    """
    Health check endpoint.

    Returns service status and confirms model is loaded.
    """
    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        model_loaded=model is not None
    )

@router.get("/ready")
def readiness_check(model=Depends(get_model)):
    """
    Kubernetes-style readiness probe.

    Returns 200 if service is ready to accept traffic.
    """
    if model is None:
        return {"status": "not ready", "reason": "model not loaded"}, 503
    return {"status": "ready"}

@router.get("/live")
def liveness_check():
    """
    Kubernetes-style liveness probe.

    Returns 200 if service is alive (even if not ready).
    """
    return {"status": "alive"}
