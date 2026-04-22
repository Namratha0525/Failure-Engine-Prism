"""
/api/v1/predict — single and batch inference endpoints.
"""
import datetime
from fastapi import APIRouter, Depends, HTTPException
from failure_api.core.ensemble import get_predictor
from failure_api.core.alert_state import get_alert_state
from failure_api.core.config import settings
from failure_api.schemas import MetricsWindow, PredictionResponse, BatchRequest, BatchResponse

router = APIRouter(tags=["predict"])


def _build_response(result: dict, suppressed: bool, action: str,
                    explanation: list | None = None) -> PredictionResponse:
    return PredictionResponse(
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        Failure="YES" if result["is_failure"] else "NO",
        Confidence=result["confidence"],
        Root_Cause_Service=result["rc_label"],
        Failure_Type=result["ft_label"],
        Propagation_Path=result["propagation"],
        Action=action,
        Suppressed=suppressed,
        model_version=settings.model_version,
        explanation=explanation,
        debug={
            "probabilities": result["probs"],
            "failure_votes": result["failure_votes"],
        } if settings.debug_mode else None,
    )


@router.post("/predict", response_model=PredictionResponse, summary="Single 30-second window prediction")
def predict_single(window: MetricsWindow):
    predictor = get_predictor()
    alert_state = get_alert_state()

    window_dict = window.model_dump(exclude={"window_start"})
    result = predictor.predict(window_dict)

    should_fire, reason = True, "NO_FAILURE"
    action = "MONITOR"
    if result["is_failure"]:
        should_fire, reason = alert_state.should_alert(
            result["ft_label"], result["ensemble_prob"])
        action = "ALERT" if should_fire else "SUPPRESSED"

    # Explanation from feature importances
    X = predictor.vectorize(window_dict)
    explanation = predictor.explain(X, top_n=5)

    return _build_response(result, not should_fire, action, explanation)


@router.post("/predict/batch", response_model=BatchResponse, summary="Batch inference on multiple windows")
def predict_batch(req: BatchRequest):
    if len(req.windows) > 500:
        raise HTTPException(status_code=422, detail="Max 500 windows per batch")

    predictor = get_predictor()
    alert_state = get_alert_state()
    predictions = []
    failures = 0
    alerts_fired = 0

    for window in req.windows:
        window_dict = window.model_dump(exclude={"window_start"})
        result = predictor.predict(window_dict)
        action = "MONITOR"
        suppressed = False

        if result["is_failure"]:
            failures += 1
            should_fire, _ = alert_state.should_alert(
                result["ft_label"], result["ensemble_prob"])
            action = "ALERT" if should_fire else "SUPPRESSED"
            suppressed = not should_fire
            if should_fire:
                alerts_fired += 1

        predictions.append(_build_response(result, suppressed, action))

    return BatchResponse(
        predictions=predictions,
        total=len(predictions),
        failures_detected=failures,
        alerts_fired=alerts_fired,
    )
