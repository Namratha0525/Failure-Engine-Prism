"""
Phase 3 — Log-Based Root Cause Detection
==========================================
Multi-class classification: predict the root-cause service.
Models: RF, LSTM, GRU, Transformer Encoder.
Select best by Top-1 / Top-3 accuracy.
"""

import os, warnings, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from failure_engine.config import DEVICE, MODEL_DIR
from failure_engine.phase1_ingestion import LOG_FEATURE_COLS, METRIC_FEATURE_COLS

warnings.filterwarnings("ignore")


def _top_k_accuracy(y_true, y_prob, k=3):
    """Compute top-k accuracy from probability matrix."""
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    correct = sum(1 for i, t in enumerate(y_true) if t in top_k[i])
    return correct / len(y_true)


# ── Sequence Models ──────────────────────────────────────────────────────
class LogSeqModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=128, layers=2,
                 dropout=0.3, cell="lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_dim, hidden, layers, batch_first=True,
                           bidirectional=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LogTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4,
                 layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :])


def _train_nn(model, X_train, y_train, X_val, y_val,
              epochs=100, patience=15, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # class weights
    classes, counts = np.unique(y_train, return_counts=True)
    weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float32).to(DEVICE)
    weights = weights / weights.sum() * len(classes)
    criterion = nn.CrossEntropyLoss(weight=weights)

    Xt = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    yt = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    Xv = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)

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
        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            preds = logits.argmax(dim=1).cpu().numpy()
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
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


def run_phase3(df) -> dict:
    """Train log-based root-cause classifiers."""
    print("\n" + "=" * 60)
    print("  PHASE 3 — Log-Based Root Cause Detection")
    print("=" * 60)

    # Features: combine log + metric features for richer signal
    log_cols = [c for c in LOG_FEATURE_COLS if c in df.columns]
    metric_cols = [c for c in METRIC_FEATURE_COLS if c in df.columns]
    feat_cols = log_cols + metric_cols
    X = df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    le = LabelEncoder()
    y = le.fit_transform(df["label_root_cause"].values)
    num_classes = len(le.classes_)
    print(f"  Classes ({num_classes}): {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    # ── Random Forest ──
    print("\n  Training RF on log features ...")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)
    results["RF"] = {
        "top1": accuracy_score(y_test, rf_pred),
        "top3": _top_k_accuracy(y_test, rf_prob, k=min(3, num_classes)),
        "f1_macro": f1_score(y_test, rf_pred, average="macro", zero_division=0),
        "model": rf, "type": "sklearn"
    }
    print(f"    Top-1: {results['RF']['top1']:.4f} | Top-3: {results['RF']['top3']:.4f}")

    # ── LSTM ──
    print("  Training LSTM ...")
    lstm = LogSeqModel(X.shape[1], num_classes, cell="lstm").to(DEVICE)
    lstm, lstm_f1 = _train_nn(lstm, X_train, y_train, X_test, y_test)
    lstm.eval()
    with torch.no_grad():
        Xv = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        logits = lstm(Xv)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    results["LSTM"] = {
        "top1": accuracy_score(y_test, preds),
        "top3": _top_k_accuracy(y_test, probs, k=min(3, num_classes)),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        "model": lstm, "type": "torch"
    }
    print(f"    Top-1: {results['LSTM']['top1']:.4f} | Top-3: {results['LSTM']['top3']:.4f}")

    # ── GRU ──
    print("  Training GRU ...")
    gru = LogSeqModel(X.shape[1], num_classes, cell="gru").to(DEVICE)
    gru, gru_f1 = _train_nn(gru, X_train, y_train, X_test, y_test)
    gru.eval()
    with torch.no_grad():
        logits = gru(Xv)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    results["GRU"] = {
        "top1": accuracy_score(y_test, preds),
        "top3": _top_k_accuracy(y_test, probs, k=min(3, num_classes)),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        "model": gru, "type": "torch"
    }
    print(f"    Top-1: {results['GRU']['top1']:.4f} | Top-3: {results['GRU']['top3']:.4f}")

    # ── Transformer ──
    print("  Training Transformer Encoder ...")
    tfm = LogTransformer(X.shape[1], num_classes).to(DEVICE)
    tfm, tfm_f1 = _train_nn(tfm, X_train, y_train, X_test, y_test)
    tfm.eval()
    with torch.no_grad():
        logits = tfm(Xv)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    results["Transformer"] = {
        "top1": accuracy_score(y_test, preds),
        "top3": _top_k_accuracy(y_test, probs, k=min(3, num_classes)),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        "model": tfm, "type": "torch"
    }
    print(f"    Top-1: {results['Transformer']['top1']:.4f} | Top-3: {results['Transformer']['top3']:.4f}")

    # ── Best ──
    best_name = max(results, key=lambda k: results[k]["top1"])
    best = results[best_name]
    print(f"\n  ★ Best log model: {best_name} (Top-1: {best['top1']:.4f})")

    path = os.path.join(MODEL_DIR, "logs_best_model.pt")
    if best["type"] == "torch":
        torch.save({"state_dict": best["model"].state_dict(),
                     "model_name": best_name,
                     "num_classes": num_classes,
                     "input_dim": X.shape[1],
                     "label_classes": list(le.classes_)}, path)
    else:
        import joblib
        joblib.dump(best["model"], path.replace(".pt", ".pkl"))
        path = path.replace(".pt", ".pkl")
    print(f"  Saved → {path}")

    # Comparison table
    print(f"\n  {'Model':<15s} {'Top-1':>7s} {'Top-3':>7s} {'F1-Mac':>7s}")
    print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*7}")
    for name, r in results.items():
        print(f"  {name:<15s} {r['top1']:7.4f} {r['top3']:7.4f} {r['f1_macro']:7.4f}")

    return {"best_name": best_name, "results": results, "label_encoder": le,
            "X_test": X_test, "y_test": y_test, "feature_cols": feat_cols}


if __name__ == "__main__":
    from failure_engine.phase1_ingestion import load_cached, ingest_all
    df = load_cached()
    if df is None:
        df = ingest_all()
    run_phase3(df)
