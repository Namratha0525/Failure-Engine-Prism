"""/admin endpoints: model info, reload, alert reset."""
from fastapi import APIRouter
from failure_api.core.ensemble import get_predictor, init_predictor
from failure_api.core.alert_state import get_alert_state
from failure_api.core.config import settings

router = APIRouter(tags=["admin"])

@router.get("/model-info")
def model_info():
    p = get_predictor()
    return {
        "model_version": settings.model_version,
        "models_loaded": list(p.models.keys()),
        "feature_count": len(p.meta.get("feature_cols", [])),
        "rc_classes": p.meta.get("rc_classes", []),
        "ft_classes": p.meta.get("ft_classes", []),
        "thresholds": p.meta.get("thresholds", {}),
        "vote_required": p.meta.get("vote_required", 2),
    }

@router.post("/reload-model")
def reload_model():
    """Hot-reload models from disk without restarting the container."""
    init_predictor(settings.model_dir)
    return {"status": "reloaded", "model_dir": settings.model_dir}

@router.post("/reset-alerts")
def reset_alerts():
    """Reset alert suppression state."""
    get_alert_state().reset()
    return {"status": "alert state reset"}
