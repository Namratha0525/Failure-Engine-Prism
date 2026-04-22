"""
Phase 2 — Metrics-Based Failure Detection
==========================================
Train RF, XGBoost, LightGBM, LSTM/GRU on metric features.
Auto-select best model via 5-fold CV F1.
"""

import os, warnings, numpy as np, pandas as pd, joblib, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
import xgboost as xgb
import lightgbm as lgb
from failure_engine.config import DEVICE, MODEL_DIR
from failure_engine.phase1_ingestion import METRIC_FEATURE_COLS

warnings.filterwarnings("ignore")

# ── LSTM Model ───────────────────────────────────────────────────────────
class MetricsLSTM(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, 1, features) — treat each window as a 1-step sequence
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def _train_lstm(X_train, y_train, X_val, y_val, epochs=100, patience=10):
    input_dim = X_train.shape[1]
    model = MetricsLSTM(input_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # handle class imbalance
    pos_w = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))

    Xt = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    Xv = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    yv = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    ds = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    best_f1, best_state, wait = 0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        # validate
        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1


def run_phase2(df: pd.DataFrame) -> dict:
    """Train & compare metrics models. Returns results dict."""
    print("\n" + "=" * 60)
    print("  PHASE 2 — Metrics-Based Failure Detection")
    print("=" * 60)

    # Prepare features
    avail_cols = [c for c in METRIC_FEATURE_COLS if c in df.columns]
    X = df[avail_cols].values.astype(np.float32)
    y = df["label_binary"].values.astype(int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Random Forest ──
    print("\n  Training Random Forest ...")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf_f1 = cross_val_score(rf, X, y, cv=cv, scoring="f1").mean()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    results["RandomForest"] = {
        "cv_f1": rf_f1,
        "accuracy": accuracy_score(y_test, rf_pred),
        "precision": precision_score(y_test, rf_pred, zero_division=0),
        "recall": recall_score(y_test, rf_pred, zero_division=0),
        "f1": f1_score(y_test, rf_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, rf_prob),
        "model": rf,
    }
    print(f"    CV F1: {rf_f1:.4f} | Test F1: {results['RandomForest']['f1']:.4f}")

    # ── XGBoost ──
    print("  Training XGBoost ...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, use_label_encoder=False, eval_metric="logloss",
        tree_method="hist", device="cuda", random_state=42,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1))
    xgb_f1 = cross_val_score(xgb_clf, X, y, cv=cv, scoring="f1").mean()
    xgb_clf.fit(X_train, y_train)
    xgb_pred = xgb_clf.predict(X_test)
    xgb_prob = xgb_clf.predict_proba(X_test)[:, 1]
    results["XGBoost"] = {
        "cv_f1": xgb_f1,
        "accuracy": accuracy_score(y_test, xgb_pred),
        "precision": precision_score(y_test, xgb_pred, zero_division=0),
        "recall": recall_score(y_test, xgb_pred, zero_division=0),
        "f1": f1_score(y_test, xgb_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, xgb_prob),
        "model": xgb_clf,
    }
    print(f"    CV F1: {xgb_f1:.4f} | Test F1: {results['XGBoost']['f1']:.4f}")

    # ── LightGBM ──
    print("  Training LightGBM ...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=200, is_unbalance=True, random_state=42,
        verbose=-1, n_jobs=-1)
    lgb_f1 = cross_val_score(lgb_clf, X, y, cv=cv, scoring="f1").mean()
    lgb_clf.fit(X_train, y_train)
    lgb_pred = lgb_clf.predict(X_test)
    lgb_prob = lgb_clf.predict_proba(X_test)[:, 1]
    results["LightGBM"] = {
        "cv_f1": lgb_f1,
        "accuracy": accuracy_score(y_test, lgb_pred),
        "precision": precision_score(y_test, lgb_pred, zero_division=0),
        "recall": recall_score(y_test, lgb_pred, zero_division=0),
        "f1": f1_score(y_test, lgb_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, lgb_prob),
        "model": lgb_clf,
    }
    print(f"    CV F1: {lgb_f1:.4f} | Test F1: {results['LightGBM']['f1']:.4f}")

    # ── LSTM ──
    print("  Training LSTM ...")
    lstm_model, lstm_cv_f1 = _train_lstm(X_train, y_train, X_test, y_test)
    lstm_model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        logits = lstm_model(Xt)
        lstm_prob = torch.sigmoid(logits).cpu().numpy()
        lstm_pred = (lstm_prob > 0.5).astype(int)
    results["LSTM"] = {
        "cv_f1": lstm_cv_f1,
        "accuracy": accuracy_score(y_test, lstm_pred),
        "precision": precision_score(y_test, lstm_pred, zero_division=0),
        "recall": recall_score(y_test, lstm_pred, zero_division=0),
        "f1": f1_score(y_test, lstm_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, lstm_prob),
        "model": lstm_model,
    }
    print(f"    Val F1: {lstm_cv_f1:.4f} | Test F1: {results['LSTM']['f1']:.4f}")

    # ── Select best ──
    best_name = max(results, key=lambda k: results[k]["cv_f1"])
    best = results[best_name]
    print(f"\n  ★ Best metrics model: {best_name} (CV F1: {best['cv_f1']:.4f})")

    # Save
    if best_name == "LSTM":
        path = os.path.join(MODEL_DIR, "metrics_best_model.pt")
        torch.save(best["model"].state_dict(), path)
    else:
        path = os.path.join(MODEL_DIR, "metrics_best_model.pkl")
        joblib.dump(best["model"], path)
    print(f"  Saved → {path}")

    # Print comparison table
    print(f"\n  {'Model':<15s} {'CV-F1':>7s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUC':>7s}")
    print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for name, r in results.items():
        print(f"  {name:<15s} {r['cv_f1']:7.4f} {r['accuracy']:7.4f} {r['precision']:7.4f} "
              f"{r['recall']:7.4f} {r['f1']:7.4f} {r['roc_auc']:7.4f}")

    return {"best_name": best_name, "best_model": best["model"], "results": results,
            "feature_cols": avail_cols, "X_test": X_test, "y_test": y_test}


if __name__ == "__main__":
    from failure_engine.phase1_ingestion import load_cached, ingest_all
    df = load_cached()
    if df is None:
        df = ingest_all()
    run_phase2(df)
