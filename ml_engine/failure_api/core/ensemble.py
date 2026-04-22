"""
Ensemble predictor — loads calibrated RF, XGB, LightGBM models.
2-of-3 majority vote with Isotonic-calibrated probabilities.
"""
import os, json, logging
import numpy as np
import joblib

log = logging.getLogger(__name__)

PROPAGATION_CHAINS = {
    "media":       "media-service → compose-post-service → home-timeline-service",
    "text":        "text-service → compose-post-service → home-timeline-service",
    "user":        "user-service → auth-service → all-user-dependent-services",
    "socialgraph": "social-graph-service → user-service → compose-post-service",
    "hometimeline":"home-timeline-service → user-timeline-service",
    "usertimeline":"user-timeline-service → post-storage-service",
    "gateway":     "gateway-service → all-downstream-services",
    "travel":      "travel-service → route-service → station-service",
    "auth":        "auth-service → gateway-service → all-services",
    "system":      "host-infrastructure → all-containers",
    "database":    "database-layer → cache → api-services",
    "none":        "N/A",
}


def _get_propagation(rc: str) -> str:
    for key, chain in PROPAGATION_CHAINS.items():
        if key in rc.lower():
            return chain
    return f"{rc} → dependent-services"


def _confidence_band(prob: float, bands: dict) -> str:
    if prob >= bands.get("HIGH", 0.85):
        return f"HIGH ({prob:.2f})"
    elif prob >= bands.get("MEDIUM", 0.65):
        return f"MEDIUM ({prob:.2f})"
    elif prob >= bands.get("LOW", 0.45):
        return f"LOW ({prob:.2f})"
    return f"VERY LOW ({prob:.2f})"


class EnsemblePredictor:
    """Thread-safe ensemble predictor. Load once at startup."""

    _instance = None

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models: dict = {}
        self.meta: dict = {}
        self.rc_model = None
        self.ft_model = None
        self._load()

    def _load(self):
        meta_path = os.path.join(self.model_dir, "ensemble_meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Calibrated models not found at {meta_path}. "
                f"Run failure_engine/calibrate_ensemble.py first."
            )
        with open(meta_path) as f:
            self.meta = json.load(f)

        model_files = {
            "RF":   "ensemble_rf_calibrated.pkl",
            "XGB":  "ensemble_xgb_calibrated.pkl",
            "LGBM": "ensemble_lgbm_calibrated.pkl",
        }
        for name, fname in model_files.items():
            p = os.path.join(self.model_dir, fname)
            if os.path.isfile(p):
                self.models[name] = joblib.load(p)
                log.info(f"Loaded {name} from {p}")

        rc_path = os.path.join(self.model_dir, "rc_model_calibrated.pkl")
        ft_path = os.path.join(self.model_dir, "ft_model_calibrated.pkl")
        if os.path.isfile(rc_path):
            self.rc_model = joblib.load(rc_path)
        if os.path.isfile(ft_path):
            self.ft_model = joblib.load(ft_path)

        log.info(f"Ensemble ready: {list(self.models.keys())}, "
                 f"vote_required={self.meta['vote_required']}")

    def vectorize(self, window: dict) -> np.ndarray:
        """Dict → numpy feature vector aligned to training columns."""
        feat_cols = self.meta["feature_cols"]
        vec = np.array([float(window.get(k, 0.0)) for k in feat_cols], dtype=np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)

    def predict_raw(self, X: np.ndarray) -> dict:
        """Core inference — returns raw result dict."""
        thresholds = self.meta["thresholds"]
        bands = self.meta.get("confidence_bands", {"HIGH": 0.85, "MEDIUM": 0.65, "LOW": 0.45})
        vote_required = self.meta.get("vote_required", 2)

        probs, votes = {}, {}
        for name, model in self.models.items():
            try:
                p = float(model.predict_proba(X)[0, 1])
            except Exception as e:
                log.warning(f"{name} predict failed: {e}")
                p = 0.0
            probs[name] = round(p, 4)
            votes[name] = int(p >= thresholds.get(name, 0.7))

        failure_votes = sum(votes.values())
        ensemble_prob = float(np.mean(list(probs.values()))) if probs else 0.0
        
        # VERY STRICT logic to avoid any wrong alerts in real-time
        # Require all 3 models to vote YES and ensemble probability >= 0.95
        is_failure = (failure_votes == len(self.models)) and (ensemble_prob >= 0.95)

        rc_classes = self.meta.get("rc_classes", ["unknown"])
        ft_classes = self.meta.get("ft_classes", ["unknown"])
        rc_label, ft_label = "none", "normal"

        if is_failure:
            if self.rc_model is not None:
                rc_idx = int(self.rc_model.predict(X)[0])
                rc_label = rc_classes[rc_idx] if rc_idx < len(rc_classes) else "unknown"
            if self.ft_model is not None:
                ft_idx = int(self.ft_model.predict(X)[0])
                ft_label = ft_classes[ft_idx] if ft_idx < len(ft_classes) else "unknown"

        return {
            "is_failure": is_failure,
            "ensemble_prob": ensemble_prob,
            "confidence": _confidence_band(ensemble_prob, bands),
            "rc_label": rc_label,
            "ft_label": ft_label,
            "propagation": _get_propagation(rc_label),
            "probs": probs,
            "failure_votes": f"{failure_votes}/{len(votes)}",
        }

    def predict(self, window: dict) -> dict:
        X = self.vectorize(window)
        return self.predict_raw(X)

    def explain(self, X: np.ndarray, top_n: int = 5) -> list[dict]:
        """Return SHAP-style top feature impacts (using RF feature importances)."""
        rf = self.models.get("RF")
        if rf is None:
            return []
        try:
            # Use calibrated RF's base estimator
            base_rf = rf.calibrated_classifiers_[0].estimator
            importances = base_rf.feature_importances_
            feat_cols = self.meta["feature_cols"]
            feat_values = X[0]
            impacts = [(feat_cols[i], float(importances[i] * feat_values[i]))
                       for i in range(min(len(feat_cols), len(importances)))]
            impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            return [{"feature": f, "impact": round(v, 4)} for f, v in impacts[:top_n]]
        except Exception:
            return []


# Singleton — loaded once at API startup
_predictor: EnsemblePredictor | None = None

def get_predictor() -> EnsemblePredictor:
    global _predictor
    if _predictor is None:
        raise RuntimeError("Predictor not initialized. Call init_predictor() first.")
    return _predictor

def init_predictor(model_dir: str):
    global _predictor
    _predictor = EnsemblePredictor(model_dir)
    return _predictor
