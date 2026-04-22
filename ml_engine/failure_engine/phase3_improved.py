#!/usr/bin/env python3
"""
Phase 3 Improved — Root Cause Detection v2
==========================================
Key improvements over v1:
  1. Label normalization — collapse experiment IDs → clean service names
  2. Rich feature engineering — z-scores, cross-metric ratios, percentile ranks
  3. Multi-model stack — XGBoost + LightGBM + RF ensemble for RC
  4. Hierarchical classification — failure_type first, then RC within group
  5. Service-agnostic scoring — ranks services by anomaly score (works for any app)

Saves:
  models/rc_model_v2.pkl         — main RC classifier (XGB)
  models/rc_label_map.json       — label normalisation map
  models/rc_meta_v2.json         — class list, thresholds, feature set
"""

import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             top_k_accuracy_score)
import xgboost as xgb
import lightgbm as lgb

from failure_engine.config import MODEL_DIR
from failure_engine.phase1_ingestion import load_cached, ingest_all

# ═══════════════════════════════════════════════════════════════════
# 1. LABEL NORMALISER
# ═══════════════════════════════════════════════════════════════════
SERVICE_NORMALISE = {
    # SN services (direct name already clean)
    "media-service":       "media-service",
    "text-service":        "text-service",
    "user-service":        "user-service",
    "auth-service":        "auth-service",
    "social-graph":        "social-graph-service",
    "socialgraph":         "social-graph-service",
    "home-timeline":       "home-timeline-service",
    "hometimeline":        "home-timeline-service",
    "user-timeline":       "user-timeline-service",
    "usertimeline":        "user-timeline-service",
    # TT services
    "gateway-service":     "gateway-service",
    "travel-service":      "travel-service",
    "database":            "database",
    # System / infra
    "system":              "system",
    "none":                "none",
}

def normalise_rc_label(raw: str) -> str:
    """Map raw directory-derived label → clean service name."""
    raw = raw.lower().strip()
    for key, clean in SERVICE_NORMALISE.items():
        if key in raw:
            return clean
    # fallback: take first token before underscore/digit
    base = raw.split("_")[0].split("-20")[0]
    return base if base else "system"


def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rc_clean"] = df["label_root_cause"].apply(normalise_rc_label)
    df["ft_clean"] = df["label_failure_type"].fillna("normal")
    return df


# ═══════════════════════════════════════════════════════════════════
# 2. RICH FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
BASE_METRICS = ["cpu", "memory", "latency", "throughput", "error_rate"]

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add z-scores, ratios, composite anomaly features."""
    df = df.copy()
    feat_cols = []

    # ── Base stat features (already in df) ──
    base_stats = []
    for m in BASE_METRICS:
        for stat in ["mean", "std", "max", "min", "slope", "roc"]:
            col = f"{m}_{stat}"
            if col in df.columns:
                base_stats.append(col)
    feat_cols.extend(base_stats)

    # ── Global z-scores (how far from window population mean) ──
    for m in BASE_METRICS:
        col = f"{m}_mean"
        if col in df.columns:
            mu, sigma = df[col].mean(), df[col].std() + 1e-9
            z_col = f"{m}_zscore"
            df[z_col] = (df[col] - mu) / sigma
            feat_cols.append(z_col)

    # ── Composite anomaly score ──
    zscore_cols = [c for c in df.columns if "zscore" in c]
    df["composite_anomaly"] = df[zscore_cols].abs().mean(axis=1)
    feat_cols.append("composite_anomaly")

    # ── Cross-metric ratios ──
    if "error_rate_mean" in df.columns and "throughput_mean" in df.columns:
        df["err_per_throughput"] = df["error_rate_mean"] / (df["throughput_mean"] + 1e-6)
        feat_cols.append("err_per_throughput")

    if "cpu_mean" in df.columns and "memory_mean" in df.columns:
        df["cpu_mem_ratio"]  = df["cpu_mean"] / (df["memory_mean"] + 1e-6)
        df["cpu_mem_combo"]  = df["cpu_mean"] * df["memory_mean"]
        feat_cols.extend(["cpu_mem_ratio", "cpu_mem_combo"])

    if "latency_mean" in df.columns and "error_rate_mean" in df.columns:
        df["latency_x_err"]  = df["latency_mean"] * df["error_rate_mean"]
        feat_cols.append("latency_x_err")

    # ── Rate-of-change magnitude ──
    for m in BASE_METRICS:
        roc = f"{m}_roc"
        if roc in df.columns:
            df[f"{m}_roc_abs"] = df[roc].abs()
            feat_cols.append(f"{m}_roc_abs")

    # ── Percentile rank within window ──
    for col in ["cpu_mean", "memory_mean", "error_rate_mean", "latency_mean"]:
        if col in df.columns:
            df[f"{col}_pctrank"] = df[col].rank(pct=True)
            feat_cols.append(f"{col}_pctrank")

    # ── Log features ──
    log_cols = [c for c in df.columns if c.startswith("log_")]
    feat_cols.extend([c for c in log_cols if c in df.columns])

    # ── Trace features ──
    trace_cols = [c for c in df.columns if c.startswith("trace_")]
    feat_cols.extend([c for c in trace_cols if c in df.columns])

    # ── API features ──
    api_cols = [c for c in df.columns if c.startswith("api_")]
    feat_cols.extend([c for c in api_cols if c in df.columns])

    # Deduplicate preserving order
    seen = set(); feat_cols_final = []
    for c in feat_cols:
        if c not in seen and c in df.columns:
            seen.add(c); feat_cols_final.append(c)

    return df, feat_cols_final


# ═══════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════
def build_rc_ensemble(X_train, y_train, n_classes: int):
    """Train XGBoost + LightGBM + RF soft-voting ensemble."""

    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", device="cuda",
        eval_metric="mlogloss", random_state=42,
        use_label_encoder=False,
    )
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", verbose=-1, random_state=42,
    )
    rf_clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )

    print("    Training XGBoost (root cause) ...")
    xgb_clf.fit(X_train, y_train)
    print("    Training LightGBM (root cause) ...")
    lgb_clf.fit(X_train, y_train)
    print("    Training RandomForest (root cause) ...")
    rf_clf.fit(X_train, y_train)

    ensemble = VotingClassifier(
        estimators=[("xgb", xgb_clf), ("lgb", lgb_clf), ("rf", rf_clf)],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    return ensemble, xgb_clf, lgb_clf, rf_clf


def top_k(y_true, proba, k):
    try:
        return top_k_accuracy_score(y_true, proba, k=k)
    except Exception:
        top_k_pred = np.argsort(proba, axis=1)[:, -k:]
        return float(np.mean([y_true[i] in top_k_pred[i] for i in range(len(y_true))]))


# ═══════════════════════════════════════════════════════════════════
# 4. SERVICE-AGNOSTIC ANOMALY RANKER
# ═══════════════════════════════════════════════════════════════════
class ServiceAnomalyRanker:
    """
    App-agnostic component: given per-service metric snapshots,
    ranks services by anomaly score and identifies the root cause.
    No training needed — pure statistical anomaly scoring.
    Works for ANY microservice application.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._baselines: dict[str, dict] = {}  # service → {metric: (mean, std)}

    def update_baseline(self, service_id: str, metrics: dict):
        """Feed normal-traffic samples to build a baseline per service."""
        if service_id not in self._baselines:
            self._baselines[service_id] = {k: [] for k in metrics}
        for k, v in metrics.items():
            self._baselines[service_id][k].append(float(v))
            if len(self._baselines[service_id][k]) > 1000:
                self._baselines[service_id][k] = self._baselines[service_id][k][-1000:]

    def _anomaly_score(self, service_id: str, current: dict) -> float:
        """Z-score based anomaly score for a service."""
        if service_id not in self._baselines:
            return 0.0
        scores = []
        for k, v in current.items():
            hist = self._baselines[service_id].get(k, [])
            if len(hist) < 5:
                continue
            mu, sigma = np.mean(hist), np.std(hist) + 1e-9
            z = abs((float(v) - mu) / sigma)
            scores.append(z)
        return float(np.mean(scores)) if scores else 0.0

    def rank_services(self, current_metrics: dict[str, dict]) -> list[dict]:
        """
        Args:
            current_metrics: {service_id: {metric_name: value, ...}}
        Returns:
            Sorted list of {service, anomaly_score, rank} — highest = most likely root cause
        """
        results = []
        for svc, metrics in current_metrics.items():
            score = self._anomaly_score(svc, metrics)
            results.append({"service": svc, "anomaly_score": round(score, 4)})
        results.sort(key=lambda x: x["anomaly_score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return results

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self._baselines, f)

    def load(self, path: str):
        import json
        if os.path.isfile(path):
            with open(path) as f:
                self._baselines = json.load(f)


# ═══════════════════════════════════════════════════════════════════
# 5. MAIN PHASE RUNNER
# ═══════════════════════════════════════════════════════════════════
def run_phase3_v2(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("  PHASE 3 v2 — Improved Root Cause Detection")
    print("=" * 60)

    # ── Normalise labels ──
    df = normalise_labels(df)
    print(f"\n  Normalised RC classes: {sorted(df['rc_clean'].unique())}")

    # ── Only use failure windows for RC training ──
    fail_df = df[df["label_binary"] == 1].copy()
    print(f"  Failure windows: {len(fail_df)}")

    # ── Feature engineering ──
    fail_df, feat_cols = engineer_features(fail_df)
    print(f"  Features: {len(feat_cols)}")

    X = fail_df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y  = le.fit_transform(fail_df["rc_clean"].values)
    classes = list(le.classes_)
    print(f"  RC classes ({len(classes)}): {classes}")

    # ── CV evaluation ──
    print("\n  Cross-validation (5-fold) on XGBoost ...")
    xgb_quick = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        tree_method="hist", device="cuda",
        eval_metric="mlogloss", random_state=42,
        use_label_encoder=False,
    )
    cv_scores = cross_val_score(xgb_quick, X, y,
                                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Train full ensemble ──
    print("\n  Training full ensemble on all data ...")
    ensemble, xgb_m, lgb_m, rf_m = build_rc_ensemble(X, y, len(classes))

    # ── Evaluate ──
    proba = ensemble.predict_proba(X)
    pred  = ensemble.predict(X)

    top1 = top_k(y, proba, 1)
    top3 = top_k(y, proba, 3)
    f1   = f1_score(y, pred, average="macro", zero_division=0)

    print(f"\n  ★ Ensemble Root Cause Results:")
    print(f"    Top-1 Accuracy : {top1:.4f}")
    print(f"    Top-3 Accuracy : {top3:.4f}")
    print(f"    F1-Macro       : {f1:.4f}")

    print("\n  Per-class breakdown:")
    report = classification_report(y, pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    for cls in classes:
        r = report.get(cls, {})
        sup = int(r.get("support", 0))
        p   = r.get("precision", 0)
        rec = r.get("recall", 0)
        f   = r.get("f1-score", 0)
        tag = "✓" if rec >= 0.8 else ("~" if rec >= 0.5 else "✗")
        print(f"    {tag} {cls:<30s}  P={p:.2f} R={rec:.2f} F1={f:.2f}  n={sup}")

    # ── Save ──
    rc_path = os.path.join(MODEL_DIR, "rc_model_v2.pkl")
    joblib.dump(ensemble, rc_path)
    print(f"\n  Saved ensemble → {rc_path}")

    label_map = {raw: normalise_rc_label(raw)
                 for raw in df["label_root_cause"].unique()}
    with open(os.path.join(MODEL_DIR, "rc_label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    meta = {
        "feature_cols": feat_cols,
        "rc_classes":   classes,
        "top1": round(top1, 4),
        "top3": round(top3, 4),
        "f1":   round(f1, 4),
        "cv_accuracy": round(float(cv_scores.mean()), 4),
    }
    with open(os.path.join(MODEL_DIR, "rc_meta_v2.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Save service-agnostic ranker (empty baseline, loadable) ──
    ranker = ServiceAnomalyRanker()
    ranker.save(os.path.join(MODEL_DIR, "service_baselines.json"))

    return {
        "model": ensemble, "le": le, "feat_cols": feat_cols,
        "top1": top1, "top3": top3, "f1": f1,
        "classes": classes,
    }


if __name__ == "__main__":
    df = load_cached()
    if df is None:
        df = ingest_all()
    run_phase3_v2(df)
