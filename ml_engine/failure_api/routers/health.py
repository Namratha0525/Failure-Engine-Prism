"""/health and /ready endpoints."""
from fastapi import APIRouter
from failure_api.core.ensemble import get_predictor
import time

router = APIRouter(tags=["health"])
START_TIME = time.time()

@router.get("/health")
def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - START_TIME, 1)}

@router.get("/ready")
def ready():
    try:
        p = get_predictor()
        return {"status": "ready", "models": list(p.models.keys())}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")
