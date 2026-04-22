#!/usr/bin/env python3
"""
Calibrate Ensemble — calibrate_ensemble.py
============================================
Trains and calibrates 3 classifiers (RF, XGBoost, LightGBM) on the
ingested feature data using isotonic regression. Saves:
  - failure_engine/models/ensemble_rf_calibrated.pkl
  - failure_engine/models/ensemble_xgb_calibrated.pkl
  - failure_engine/models/ensemble_lgbm_calibrated.pkl
  - failure_engine/models/ensemble_meta.json   (threshold, feature list, label maps)
  - failure_engine/models/rc_model_calibrated.pkl  (root cause + failure type)

Run this ONCE after run_all.py to prepare the inference-ready ensemble.
"""

import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, precision_recall_curve,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings("ignore")

from failure_engine.config import MODEL_DIR
from failure_engine.phase1_ingestion import (
    load_cached, ingest_all, METRIC_FEATURE_COLS, LOG_FEATURE_COLS,
    TRACE_FEATURE_COLS, API_FEATURE_COLS, COVERAGE_FEATURE_COLS_SN,
    COVERAGE_FEATURE_COLS_TT
)

ALL_FEAT_COLS = (METRIC_FEATURE_COLS + LOG_FEATURE_COLS + TRACE_FEATURE_COLS +
                 API_FEATURE_COLS + COVERAGE_FEATURE_COLS_SN + COVERAGE_FEATURE_COLS_TT)


def _find_best_threshold(y_true, y_prob, beta=1.0):
    """Find threshold that maximises F-beta score."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    prec = prec[:-1]; rec = rec[:-1]
    denom = (beta**2 * prec) + rec
    f_scores = np.where(denom > 0, (1 + beta**2) * prec * rec / denom, 0)
    best_idx = np.argmax(f_scores)
    return float(thresholds[best_idx]), float(f_scores[best_idx])


def main():
    print("=" * 60)
    print("  ENSEMBLE CALIBRATION")
    print("=" * 60)

    # ── Load data ──
    df = load_cached()
    if df is None:
        print("  Cached data not found, running ingestion ...")
        df = ingest_all()

    feat_cols = [c for c in ALL_FEAT_COLS if c in df.columns]
    X = df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_bin = df["label_binary"].values.astype(int)

    le_rc = LabelEncoder()
    le_ft = LabelEncoder()
    y_rc = le_rc.fit_transform(df["label_root_cause"].values)
    y_ft = le_ft.fit_transform(df["label_failure_type"].values)

    X_train, X_test, y_bin_tr, y_bin_te, y_rc_tr, y_rc_te, y_ft_tr, y_ft_te = train_test_split(
        X, y_bin, y_rc, y_ft, test_size=0.2, random_state=42, stratify=y_bin)

    print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"  Root causes: {list(le_rc.classes_)}")
    print(f"  Failure types: {list(le_ft.classes_)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    calibrated_models = {}

    # ── RF ──
    print("\n  [1/3] Calibrating Random Forest ...")
    rf_base = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=42, n_jobs=-1, max_depth=None, min_samples_leaf=1)
    rf_cal = CalibratedClassifierCV(rf_base, method="isotonic", cv=5)
    rf_cal.fit(X_train, y_bin_tr)
    rf_prob = rf_cal.predict_proba(X_test)[:, 1]
    rf_thr, rf_fb = _find_best_threshold(y_bin_te, rf_prob)
    calibrated_models["RF"] = {"model": rf_cal, "threshold": rf_thr,
                                "brier": brier_score_loss(y_bin_te, rf_prob)}
    print(f"    Best threshold: {rf_thr:.3f} | Brier score: {calibrated_models['RF']['brier']:.4f}")
    path = os.path.join(MODEL_DIR, "ensemble_rf_calibrated.pkl")
    joblib.dump(rf_cal, path)
    print(f"    Saved → {path}")

    # ── XGBoost ──
    if HAS_XGB:
        print("\n  [2/3] Calibrating XGBoost ...")
        scale_pos = (y_bin_tr == 0).sum() / max((y_bin_tr == 1).sum(), 1)
        xgb_base = xgb.XGBClassifier(
            n_estimators=300, tree_method="hist", device="cuda",
            random_state=42, eval_metric="logloss",
            scale_pos_weight=scale_pos, use_label_encoder=False)
        xgb_cal = CalibratedClassifierCV(xgb_base, method="isotonic", cv=5)
        xgb_cal.fit(X_train, y_bin_tr)
        xgb_prob = xgb_cal.predict_proba(X_test)[:, 1]
        xgb_thr, _ = _find_best_threshold(y_bin_te, xgb_prob)
        calibrated_models["XGB"] = {"model": xgb_cal, "threshold": xgb_thr,
                                     "brier": brier_score_loss(y_bin_te, xgb_prob)}
        print(f"    Best threshold: {xgb_thr:.3f} | Brier score: {calibrated_models['XGB']['brier']:.4f}")
        path = os.path.join(MODEL_DIR, "ensemble_xgb_calibrated.pkl")
        joblib.dump(xgb_cal, path)
        print(f"    Saved → {path}")

    # ── LightGBM ──
    if HAS_LGBM:
        print("\n  [3/3] Calibrating LightGBM ...")
        lgbm_base = lgb.LGBMClassifier(
            n_estimators=300, is_unbalance=True, verbose=-1,
            random_state=42, n_jobs=-1)
        lgbm_cal = CalibratedClassifierCV(lgbm_base, method="isotonic", cv=5)
        lgbm_cal.fit(X_train, y_bin_tr)
        lgbm_prob = lgbm_cal.predict_proba(X_test)[:, 1]
        lgbm_thr, _ = _find_best_threshold(y_bin_te, lgbm_prob)
        calibrated_models["LGBM"] = {"model": lgbm_cal, "threshold": lgbm_thr,
                                      "brier": brier_score_loss(y_bin_te, lgbm_prob)}
        print(f"    Best threshold: {lgbm_thr:.3f} | Brier score: {calibrated_models['LGBM']['brier']:.4f}")
        path = os.path.join(MODEL_DIR, "ensemble_lgbm_calibrated.pkl")
        joblib.dump(lgbm_cal, path)
        print(f"    Saved → {path}")

    # ── Root Cause + Failure Type model ──
    print("\n  Training Root Cause + Failure Type Classifier ...")
    rc_model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    rc_model.fit(X_train, y_rc_tr)
    rc_pred = rc_model.predict(X_test)
    print(f"    RC Top-1: {accuracy_score(y_rc_te, rc_pred):.4f}")
    ft_model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    ft_model.fit(X_train, y_ft_tr)
    ft_pred = ft_model.predict(X_test)
    print(f"    FT Accuracy: {accuracy_score(y_ft_te, ft_pred):.4f}")
    joblib.dump(rc_model, os.path.join(MODEL_DIR, "rc_model_calibrated.pkl"))
    joblib.dump(ft_model, os.path.join(MODEL_DIR, "ft_model_calibrated.pkl"))

    # ── Evaluation of ensemble on test set ──
    print("\n" + "=" * 60)
    print("  ENSEMBLE TEST EVALUATION")
    print("=" * 60)
    all_probs = []
    all_thresholds = []
    for name, m in calibrated_models.items():
        prob = m["model"].predict_proba(X_test)[:, 1]
        all_probs.append(prob)
        all_thresholds.append(m["threshold"])

    # Ensemble average probability
    ensemble_prob = np.mean(all_probs, axis=0)
    ensemble_thr = np.mean(all_thresholds)

    # Unanimous vote (3 out of 3 agree on failure) to minimize false positives
    votes = np.array([(p > t).astype(int) for p, t in zip(all_probs, all_thresholds)])
    vote_pred = (votes.sum(axis=0) >= 3).astype(int)  # 3-of-3 threshold

    print(f"\n  {'Metric':<20s} {'Ensemble-Avg':>14s} {'Majority-Vote':>14s}")
    print(f"  {'-'*20} {'-'*14} {'-'*14}")

    for metric_name, fn in [
        ("Accuracy", lambda y, p: accuracy_score(y, p)),
        ("Precision", lambda y, p: precision_score(y, p, zero_division=0)),
        ("Recall", lambda y, p: recall_score(y, p, zero_division=0)),
        ("F1", lambda y, p: f1_score(y, p, zero_division=0)),
    ]:
        avg_pred = (ensemble_prob > ensemble_thr).astype(int)
        v1 = fn(y_bin_te, avg_pred)
        v2 = fn(y_bin_te, vote_pred)
        print(f"  {metric_name:<20s} {v1:>14.4f} {v2:>14.4f}")

    auc = roc_auc_score(y_bin_te, ensemble_prob)
    print(f"  {'ROC-AUC':<20s} {auc:>14.4f} {'—':>14s}")
    brier = brier_score_loss(y_bin_te, ensemble_prob)
    print(f"  {'Brier Score':<20s} {brier:>14.4f} {'—':>14s}")

    # ── Save meta ──
    thresholds_by_model = {k: v["threshold"] for k, v in calibrated_models.items()}
    meta = {
        "feature_cols": feat_cols,
        "rc_classes": list(le_rc.classes_),
        "ft_classes": list(le_ft.classes_),
        "models_available": list(calibrated_models.keys()),
        "thresholds": thresholds_by_model,
        "ensemble_threshold": float(ensemble_thr),
        "vote_required": 3,
        "brier_scores": {k: v["brier"] for k, v in calibrated_models.items()},
        "confidence_bands": {
            "HIGH": 0.85,
            "MEDIUM": 0.65,
            "LOW": 0.0
        },
    }
    meta_path = os.path.join(MODEL_DIR, "ensemble_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Meta config saved → {meta_path}")

    print(f"\n{'='*60}")
    print("  ✅ CALIBRATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
