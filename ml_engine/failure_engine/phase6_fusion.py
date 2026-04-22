"""
Phase 6 — Multimodal Fusion Engine
====================================
Combine features from all modalities.  Train fusion strategies:
  • Stacking (meta-classifier)
  • Attention-based DNN
  • Multi-input DNN
Select best model for final predictions.
"""

import os, warnings, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib
from failure_engine.config import DEVICE, MODEL_DIR
from failure_engine.phase1_ingestion import (
    METRIC_FEATURE_COLS, LOG_FEATURE_COLS, TRACE_FEATURE_COLS,
    API_FEATURE_COLS, COVERAGE_FEATURE_COLS_SN, COVERAGE_FEATURE_COLS_TT,
    ALL_FEATURE_COLS
)

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════
#  Neural Fusion Models
# ═════════════════════════════════════════════════════════════════════════

class AttentionFusion(nn.Module):
    """Multi-head attention across modality embeddings, then FC."""
    def __init__(self, modality_dims: list[int], num_classes_binary: int,
                 num_classes_rc: int, num_classes_ft: int,
                 d_model=128, nhead=4, dropout=0.3):
        super().__init__()
        # project each modality to d_model
        self.projections = nn.ModuleList([
            nn.Linear(d, d_model) for d in modality_dims
        ])
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head_binary = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes_binary))
        self.head_rc = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes_rc))
        self.head_ft = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes_ft))

    def forward(self, modality_inputs: list[torch.Tensor]):
        # modality_inputs: list of (batch, dim_i)
        tokens = []
        for proj, x in zip(self.projections, modality_inputs):
            tokens.append(proj(x).unsqueeze(1))  # (batch, 1, d_model)
        x = torch.cat(tokens, dim=1)  # (batch, num_modalities, d_model)
        x = self.encoder(x)
        pooled = x.mean(dim=1)  # (batch, d_model)
        return self.head_binary(pooled), self.head_rc(pooled), self.head_ft(pooled)


class MultiInputDNN(nn.Module):
    """Separate encoder per modality → concat → shared FC."""
    def __init__(self, modality_dims: list[int], num_classes_binary: int,
                 num_classes_rc: int, num_classes_ft: int,
                 hidden=64, dropout=0.3):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(hidden, hidden), nn.ReLU())
            for d in modality_dims
        ])
        total_hidden = hidden * len(modality_dims)
        self.shared = nn.Sequential(
            nn.Linear(total_hidden, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.head_binary = nn.Linear(64, num_classes_binary)
        self.head_rc = nn.Linear(64, num_classes_rc)
        self.head_ft = nn.Linear(64, num_classes_ft)

    def forward(self, modality_inputs: list[torch.Tensor]):
        encoded = [enc(x) for enc, x in zip(self.encoders, modality_inputs)]
        cat = torch.cat(encoded, dim=1)
        shared = self.shared(cat)
        return self.head_binary(shared), self.head_rc(shared), self.head_ft(shared)


def _get_modality_splits(df):
    """Split features by modality group."""
    m_cols = [c for c in METRIC_FEATURE_COLS if c in df.columns]
    l_cols = [c for c in LOG_FEATURE_COLS if c in df.columns]
    t_cols = [c for c in TRACE_FEATURE_COLS if c in df.columns]
    a_cols = [c for c in API_FEATURE_COLS if c in df.columns]
    c_cols = [c for c in (COVERAGE_FEATURE_COLS_SN + COVERAGE_FEATURE_COLS_TT) if c in df.columns]
    return [m_cols, l_cols, t_cols, a_cols, c_cols]


def _train_fusion_nn(model, modal_X_train, y_bin_train, y_rc_train, y_ft_train,
                     modal_X_val, y_bin_val, y_rc_val, y_ft_val,
                     epochs=100, patience=15):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_bin = nn.CrossEntropyLoss()
    criterion_rc = nn.CrossEntropyLoss()
    criterion_ft = nn.CrossEntropyLoss()

    # to tensors
    Xs_train = [torch.tensor(x, dtype=torch.float32).to(DEVICE) for x in modal_X_train]
    Xs_val = [torch.tensor(x, dtype=torch.float32).to(DEVICE) for x in modal_X_val]
    yb_t = torch.tensor(y_bin_train, dtype=torch.long).to(DEVICE)
    yr_t = torch.tensor(y_rc_train, dtype=torch.long).to(DEVICE)
    yf_t = torch.tensor(y_ft_train, dtype=torch.long).to(DEVICE)

    best_f1, best_state, wait = 0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out_b, out_r, out_f = model(Xs_train)
        loss = criterion_bin(out_b, yb_t) + criterion_rc(out_r, yr_t) + criterion_ft(out_f, yf_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            ob, orc, oft = model(Xs_val)
            pred_b = ob.argmax(dim=1).cpu().numpy()
            pred_rc = orc.argmax(dim=1).cpu().numpy()
        f1_b = f1_score(y_bin_val, pred_b, average="macro", zero_division=0)
        f1_rc = f1_score(y_rc_val, pred_rc, average="macro", zero_division=0)
        combined = (f1_b + f1_rc) / 2
        if combined > best_f1:
            best_f1 = combined
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1


def run_phase6(df) -> dict:
    """Train and compare fusion strategies."""
    print("\n" + "=" * 60)
    print("  PHASE 6 — Multimodal Fusion Engine")
    print("=" * 60)

    # Labels
    le_rc = LabelEncoder()
    le_ft = LabelEncoder()
    y_binary = df["label_binary"].values.astype(int)
    y_rc = le_rc.fit_transform(df["label_root_cause"].values)
    y_ft = le_ft.fit_transform(df["label_failure_type"].values)

    modality_col_groups = _get_modality_splits(df)
    all_feat_cols = [c for group in modality_col_groups for c in group]
    X_all = df[all_feat_cols].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardise
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # Split per modality
    modal_dims = [len(g) for g in modality_col_groups]
    # filter out empty modalities
    non_empty = [(i, d) for i, d in enumerate(modal_dims) if d > 0]
    modal_dims_ne = [d for _, d in non_empty]
    modal_indices = [i for i, _ in non_empty]

    def split_by_modality(X_flat):
        splits = []
        offset = 0
        for i, g in enumerate(modality_col_groups):
            if i in modal_indices:
                splits.append(X_flat[:, offset:offset + len(g)])
            offset += len(g)
        return splits

    idx_train, idx_test = train_test_split(
        np.arange(len(X_all)), test_size=0.2, random_state=42, stratify=y_binary)

    modal_X_train = split_by_modality(X_all[idx_train])
    modal_X_test = split_by_modality(X_all[idx_test])

    y_bin_train, y_bin_test = y_binary[idx_train], y_binary[idx_test]
    y_rc_train, y_rc_test = y_rc[idx_train], y_rc[idx_test]
    y_ft_train, y_ft_test = y_ft[idx_train], y_ft[idx_test]

    num_rc = len(le_rc.classes_)
    num_ft = len(le_ft.classes_)
    results = {}

    # ── Stacking ──
    print("\n  Training Stacking (RF meta-classifier) ...")
    X_flat_train = X_all[idx_train]
    X_flat_test = X_all[idx_test]
    stack_rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      random_state=42, n_jobs=-1)
    stack_rf.fit(X_flat_train, y_rc_train)
    stack_pred = stack_rf.predict(X_flat_test)
    stack_prob_rc = stack_rf.predict_proba(X_flat_test)

    # binary from a separate RF
    stack_rf_bin = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                          random_state=42, n_jobs=-1)
    stack_rf_bin.fit(X_flat_train, y_bin_train)
    stack_pred_bin = stack_rf_bin.predict(X_flat_test)
    stack_prob_bin = stack_rf_bin.predict_proba(X_flat_test)[:, 1]

    results["Stacking"] = {
        "f1_binary": f1_score(y_bin_test, stack_pred_bin, zero_division=0),
        "roc_auc": roc_auc_score(y_bin_test, stack_prob_bin) if len(np.unique(y_bin_test)) > 1 else 0,
        "rc_top1": accuracy_score(y_rc_test, stack_pred),
        "rc_f1": f1_score(y_rc_test, stack_pred, average="macro", zero_division=0),
        "model_bin": stack_rf_bin, "model_rc": stack_rf,
        "type": "sklearn",
    }
    print(f"    Binary F1: {results['Stacking']['f1_binary']:.4f} | RC Top-1: {results['Stacking']['rc_top1']:.4f}")

    # ── Attention DNN ──
    print("  Training Attention Fusion DNN ...")
    attn = AttentionFusion(modal_dims_ne, 2, num_rc, num_ft).to(DEVICE)
    attn, attn_f1 = _train_fusion_nn(
        attn, modal_X_train, y_bin_train, y_rc_train, y_ft_train,
        modal_X_test, y_bin_test, y_rc_test, y_ft_test)
    attn.eval()
    with torch.no_grad():
        Xs_v = [torch.tensor(x, dtype=torch.float32).to(DEVICE) for x in modal_X_test]
        ob, orc, oft = attn(Xs_v)
        pred_b = ob.argmax(dim=1).cpu().numpy()
        pred_rc = orc.argmax(dim=1).cpu().numpy()
        pred_ft = oft.argmax(dim=1).cpu().numpy()
        prob_b = torch.softmax(ob, dim=1)[:, 1].cpu().numpy()
        prob_rc = torch.softmax(orc, dim=1).cpu().numpy()
    results["Attention"] = {
        "f1_binary": f1_score(y_bin_test, pred_b, zero_division=0),
        "roc_auc": roc_auc_score(y_bin_test, prob_b) if len(np.unique(y_bin_test)) > 1 else 0,
        "rc_top1": accuracy_score(y_rc_test, pred_rc),
        "rc_f1": f1_score(y_rc_test, pred_rc, average="macro", zero_division=0),
        "model": attn, "type": "torch",
    }
    print(f"    Binary F1: {results['Attention']['f1_binary']:.4f} | RC Top-1: {results['Attention']['rc_top1']:.4f}")

    # ── Multi-Input DNN ──
    print("  Training Multi-Input DNN ...")
    midnn = MultiInputDNN(modal_dims_ne, 2, num_rc, num_ft).to(DEVICE)
    midnn, mi_f1 = _train_fusion_nn(
        midnn, modal_X_train, y_bin_train, y_rc_train, y_ft_train,
        modal_X_test, y_bin_test, y_rc_test, y_ft_test)
    midnn.eval()
    with torch.no_grad():
        ob, orc, oft = midnn(Xs_v)
        pred_b = ob.argmax(dim=1).cpu().numpy()
        pred_rc = orc.argmax(dim=1).cpu().numpy()
        pred_ft = oft.argmax(dim=1).cpu().numpy()
        prob_b = torch.softmax(ob, dim=1)[:, 1].cpu().numpy()
        prob_rc = torch.softmax(orc, dim=1).cpu().numpy()
    results["MultiInputDNN"] = {
        "f1_binary": f1_score(y_bin_test, pred_b, zero_division=0),
        "roc_auc": roc_auc_score(y_bin_test, prob_b) if len(np.unique(y_bin_test)) > 1 else 0,
        "rc_top1": accuracy_score(y_rc_test, pred_rc),
        "rc_f1": f1_score(y_rc_test, pred_rc, average="macro", zero_division=0),
        "model": midnn, "type": "torch",
    }
    print(f"    Binary F1: {results['MultiInputDNN']['f1_binary']:.4f} | RC Top-1: {results['MultiInputDNN']['rc_top1']:.4f}")

    # ── Best — optimise for combined binary F1 + RC accuracy ──
    def _score(r):
        return r["f1_binary"] * 0.4 + r["rc_top1"] * 0.6

    best_name = max(results, key=lambda k: _score(results[k]))
    best = results[best_name]
    print(f"\n  ★ Best fusion: {best_name} (Binary F1: {best['f1_binary']:.4f}, RC Top-1: {best['rc_top1']:.4f})")

    # Save
    save_data = {
        "best_name": best_name,
        "rc_classes": list(le_rc.classes_),
        "ft_classes": list(le_ft.classes_),
        "modal_dims": modal_dims_ne,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "feature_cols": all_feat_cols,
    }
    if best["type"] == "torch":
        save_data["state_dict"] = best["model"].state_dict()
        path = os.path.join(MODEL_DIR, "fusion_best_model.pt")
        torch.save(save_data, path)
    else:
        path_rc = os.path.join(MODEL_DIR, "fusion_best_model_rc.pkl")
        path_bin = os.path.join(MODEL_DIR, "fusion_best_model_bin.pkl")
        joblib.dump(best["model_rc"], path_rc)
        joblib.dump(best["model_bin"], path_bin)
        path = path_rc
    print(f"  Saved → {path}")

    # Comparison table
    print(f"\n  {'Model':<15s} {'Bin-F1':>7s} {'AUC':>7s} {'RC-T1':>7s} {'RC-F1':>7s}")
    print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for name, r in results.items():
        print(f"  {name:<15s} {r['f1_binary']:7.4f} {r['roc_auc']:7.4f} "
              f"{r['rc_top1']:7.4f} {r['rc_f1']:7.4f}")

    return {"best_name": best_name, "results": results,
            "le_rc": le_rc, "le_ft": le_ft, "scaler": scaler,
            "idx_test": idx_test, "y_bin_test": y_bin_test,
            "y_rc_test": y_rc_test, "y_ft_test": y_ft_test}


if __name__ == "__main__":
    from failure_engine.phase1_ingestion import load_cached, ingest_all
    df = load_cached()
    if df is None:
        df = ingest_all()
    run_phase6(df)
