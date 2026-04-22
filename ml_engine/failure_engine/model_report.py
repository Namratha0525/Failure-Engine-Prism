#!/usr/bin/env python3
"""
Model Report — model_report.py
================================
Generates a full visual + text report of all models:
  - Confusion matrices (binary + root cause + failure type)
  - Precision / Recall / F1 bar charts
  - ROC curve
  - Sample predictions table
  - Per-class accuracy heatmap

Saves everything to: failure_engine/models/report/
"""

import os, sys, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

from failure_engine.config import MODEL_DIR
from failure_engine.phase1_ingestion import load_cached, ingest_all

REPORT_DIR = os.path.join(MODEL_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
BG     = "#0F172A"
CARD   = "#1E293B"
TEXT   = "#F1F5F9"
ACCENT = "#38BDF8"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": "#334155", "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": "#1E293B",
    "font.family": "DejaVu Sans", "font.size": 11,
})

# ══════════════════════════════════════════════════════════════════════
def load_data():
    df = load_cached()
    if df is None:
        df = ingest_all()
    meta = json.load(open(os.path.join(MODEL_DIR, "ensemble_meta.json")))
    feat_cols = [c for c in meta["feature_cols"] if c in df.columns]
    X = df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    le_rc = LabelEncoder(); le_ft = LabelEncoder()
    y_bin = df["label_binary"].values.astype(int)
    y_rc  = le_rc.fit_transform(df["label_root_cause"].values)
    y_ft  = le_ft.fit_transform(df["label_failure_type"].values)
    return X, y_bin, y_rc, y_ft, le_rc, le_ft, meta, df

def load_models():
    rf  = joblib.load(os.path.join(MODEL_DIR, "ensemble_rf_calibrated.pkl"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "ensemble_xgb_calibrated.pkl"))
    lgb = joblib.load(os.path.join(MODEL_DIR, "ensemble_lgbm_calibrated.pkl"))
    rc  = joblib.load(os.path.join(MODEL_DIR, "rc_model_calibrated.pkl"))
    ft  = joblib.load(os.path.join(MODEL_DIR, "ft_model_calibrated.pkl"))
    return rf, xgb, lgb, rc, ft


# ══════════════════════════════════════════════════════════════════════
# 1. METRICS SUMMARY CARD
# ══════════════════════════════════════════════════════════════════════
def plot_metrics_summary(models, X, y_bin, meta):
    print("  [1/6] Metrics summary ...")
    thresholds = meta["thresholds"]
    names = ["RF", "XGB", "LGBM"]
    model_list = list(models)

    rows = []
    for name, m in zip(names, model_list):
        prob = m.predict_proba(X)[:, 1]
        thr  = thresholds.get(name, 0.5)
        pred = (prob >= thr).astype(int)
        rows.append({
            "Model": name,
            "Accuracy":  accuracy_score(y_bin, pred),
            "Precision": precision_score(y_bin, pred, zero_division=0),
            "Recall":    recall_score(y_bin, pred, zero_division=0),
            "F1":        f1_score(y_bin, pred, zero_division=0),
            "AUC":       roc_auc_score(y_bin, prob),
        })

    # Ensemble 2-of-3
    probs_all = [m.predict_proba(X)[:,1] for m in model_list]
    votes = np.array([(p >= thresholds.get(n,0.5)).astype(int)
                      for p, n in zip(probs_all, names)])
    ens_pred = (votes.sum(0) >= 2).astype(int)
    ens_prob = np.mean(probs_all, axis=0)
    rows.append({
        "Model": "Ensemble\n(2-of-3)",
        "Accuracy":  accuracy_score(y_bin, ens_pred),
        "Precision": precision_score(y_bin, ens_pred, zero_division=0),
        "Recall":    recall_score(y_bin, ens_pred, zero_division=0),
        "F1":        f1_score(y_bin, ens_pred, zero_division=0),
        "AUC":       roc_auc_score(y_bin, ens_prob),
    })

    df_res = pd.DataFrame(rows)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    x = np.arange(len(metrics))
    colors = [BLUE, GREEN, ORANGE, PURPLE]

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
    ax.set_facecolor(CARD)
    w = 0.18
    for i, (_, row) in enumerate(df_res.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i*w - 1.5*w, vals, w, label=row["Model"],
                      color=colors[i], alpha=0.9, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT)

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0.7, 1.05); ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Binary Failure Detection — Per-Model Metrics", fontsize=14,
                 color=ACCENT, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.3)
    ax.axhline(0.9, color="#EF4444", linestyle="--", linewidth=1.2, label="90% target")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "1_metrics_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"    → {path}")
    return ens_pred, ens_prob, df_res


# ══════════════════════════════════════════════════════════════════════
# 2. BINARY CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════
def plot_confusion_binary(y_bin, ens_pred):
    print("  [2/6] Binary confusion matrix ...")
    cm = confusion_matrix(y_bin, ens_pred)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    ax.set_facecolor(CARD)
    cmap = LinearSegmentedColormap.from_list("blue_dark", ["#0F172A", "#2563EB"])
    im = ax.imshow(cm, cmap=cmap, aspect="auto")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Predicted\nNormal", "Predicted\nFailure"], fontsize=11)
    ax.set_yticklabels(["Actual\nNormal", "Actual\nFailure"], fontsize=11)
    for i in range(2):
        for j in range(2):
            label = ["TN","FP","FN","TP"][i*2+j]
            ax.text(j, i, f"{cm[i,j]}\n({label})", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color=TEXT if cm[i,j] < cm.max()*0.6 else "#0F172A")
    ax.set_title("Ensemble Confusion Matrix — Binary Detection", fontsize=13,
                 color=ACCENT, fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "2_confusion_binary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"    → {path}")


# ══════════════════════════════════════════════════════════════════════
# 3. ROC CURVE
# ══════════════════════════════════════════════════════════════════════
def plot_roc(models, names, X, y_bin, meta):
    print("  [3/6] ROC curve ...")
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(CARD)
    colors_roc = [BLUE, GREEN, ORANGE, PURPLE]

    for (name, m), col in zip(zip(names, models), colors_roc):
        prob = m.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_bin, prob)
        auc = roc_auc_score(y_bin, prob)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name}  AUC={auc:.4f}")

    # Ensemble
    probs_all = [m.predict_proba(X)[:,1] for m in models]
    ens_prob = np.mean(probs_all, axis=0)
    fpr, tpr, _ = roc_curve(y_bin, ens_prob)
    auc = roc_auc_score(y_bin, ens_prob)
    ax.plot(fpr, tpr, color="#38BDF8", lw=3, linestyle="--",
            label=f"Ensemble  AUC={auc:.4f}")

    ax.plot([0,1],[0,1], color="#475569", lw=1, linestyle=":")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#38BDF8")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Binary Failure Detection", fontsize=13,
                 color=ACCENT, fontweight="bold", pad=10)
    ax.legend(framealpha=0.3, fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "3_roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"    → {path}")


# ══════════════════════════════════════════════════════════════════════
# 4. ROOT CAUSE CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════
def plot_confusion_rootcause(rc_model, X, y_rc, le_rc):
    print("  [4/6] Root cause confusion matrix ...")
    pred_rc = rc_model.predict(X)
    labels  = list(le_rc.classes_)
    # Shorten long labels
    short = [l.split("_")[0][:18] for l in labels]
    cm = confusion_matrix(y_rc, pred_rc, labels=range(len(labels)))
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(10, n), max(8, n-2)), facecolor=BG)
    ax.set_facecolor(CARD)
    cmap = LinearSegmentedColormap.from_list("purple_dark", ["#0F172A", "#7C3AED"])
    im = ax.imshow(cm, cmap=cmap, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=8)
    # Annotate only non-zero cells
    for i in range(n):
        for j in range(n):
            if cm[i,j] > 0:
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=7,
                        color=TEXT if cm[i,j] < cm.max()*0.5 else "#0F172A",
                        fontweight="bold")
    ax.set_xlabel("Predicted Root Cause", fontsize=11)
    ax.set_ylabel("Actual Root Cause", fontsize=11)
    ax.set_title("Root Cause Prediction — Confusion Matrix", fontsize=13,
                 color=ACCENT, fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "4_confusion_rootcause.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"    → {path}")
    acc = accuracy_score(y_rc, pred_rc)
    f1  = f1_score(y_rc, pred_rc, average="macro", zero_division=0)
    print(f"    RC Accuracy={acc:.4f}  F1-macro={f1:.4f}")


# ══════════════════════════════════════════════════════════════════════
# 5. FAILURE TYPE CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════
def plot_confusion_failtype(ft_model, X, y_ft, le_ft):
    print("  [5/6] Failure type confusion matrix ...")
    pred_ft = ft_model.predict(X)
    labels  = list(le_ft.classes_)
    cm = confusion_matrix(y_ft, pred_ft, labels=range(len(labels)))
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(10, n), max(8, n-2)), facecolor=BG)
    ax.set_facecolor(CARD)
    cmap = LinearSegmentedColormap.from_list("green_dark", ["#0F172A", "#16A34A"])
    im = ax.imshow(cm, cmap=cmap, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            if cm[i,j] > 0:
                ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=8,
                        color=TEXT if cm[i,j] < cm.max()*0.5 else "#0F172A",
                        fontweight="bold")
    ax.set_xlabel("Predicted Failure Type", fontsize=11)
    ax.set_ylabel("Actual Failure Type", fontsize=11)
    ax.set_title("Failure Type Prediction — Confusion Matrix", fontsize=13,
                 color=ACCENT, fontweight="bold", pad=10)
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "5_confusion_failtype.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"    → {path}")
    acc = accuracy_score(y_ft, pred_ft)
    f1  = f1_score(y_ft, pred_ft, average="macro", zero_division=0)
    print(f"    FT Accuracy={acc:.4f}  F1-macro={f1:.4f}")


# ══════════════════════════════════════════════════════════════════════
# 6. SAMPLE PREDICTIONS TABLE
# ══════════════════════════════════════════════════════════════════════
def plot_sample_predictions(models, rc_model, ft_model, X, y_bin, y_rc, y_ft, le_rc, le_ft, meta, df):
    print("  [6/6] Sample predictions table ...")
    thresholds = meta["thresholds"]
    names = ["RF","XGB","LGBM"]

    probs_all = [m.predict_proba(X)[:,1] for m in models]
    votes = np.array([(p >= thresholds.get(n,0.5)).astype(int)
                      for p, n in zip(probs_all, names)])
    ens_pred  = (votes.sum(0) >= 2).astype(int)
    ens_prob  = np.mean(probs_all, axis=0)
    pred_rc   = rc_model.predict(X)
    pred_ft   = ft_model.predict(X)

    # Sample: 5 correct failures, 5 correct normals, any misclassified
    idx_fail_ok   = np.where((ens_pred==1) & (y_bin==1))[0][:5]
    idx_norm_ok   = np.where((ens_pred==0) & (y_bin==0))[0][:5]
    idx_wrong     = np.where(ens_pred != y_bin)[0][:3]
    sample_idx    = np.concatenate([idx_fail_ok, idx_norm_ok, idx_wrong])

    rows = []
    for i in sample_idx:
        actual_bin  = "FAILURE" if y_bin[i] else "NORMAL"
        pred_bin    = "FAILURE" if ens_pred[i] else "NORMAL"
        correct     = "✓" if y_bin[i]==ens_pred[i] else "✗"
        conf        = f"{ens_prob[i]:.2f}"
        rc_pred_lbl = le_rc.inverse_transform([pred_rc[i]])[0].split("_")[0][:16]
        rc_true_lbl = le_rc.inverse_transform([y_rc[i]])[0].split("_")[0][:16]
        ft_pred_lbl = le_ft.inverse_transform([pred_ft[i]])[0]
        ft_true_lbl = le_ft.inverse_transform([y_ft[i]])[0]
        rows.append([correct, actual_bin, pred_bin, conf, rc_true_lbl, rc_pred_lbl, ft_true_lbl, ft_pred_lbl])

    col_labels = ["OK?","Actual","Predicted","Conf","Root Cause\n(Actual)","Root Cause\n(Pred)","Fail Type\n(Actual)","Fail Type\n(Pred)"]
    fig, ax = plt.subplots(figsize=(18, max(4, len(rows)*0.55 + 1.5)), facecolor=BG)
    ax.set_facecolor(BG); ax.axis("off")
    ax.set_title("Sample Predictions (Ensemble 2-of-3 Voting)", fontsize=14,
                 color=ACCENT, fontweight="bold", pad=14)
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center",
                   bbox=[0, 0, 1, 0.92])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#334155")
        if r == 0:
            cell.set_facecolor(BLUE); cell.set_text_props(color=TEXT, fontweight="bold")
        else:
            row_data = rows[r-1]
            ok   = row_data[0]
            bg   = "#052e16" if ok=="✓" and row_data[1]=="FAILURE" else \
                   "#172554" if ok=="✓" else "#450a0a"
            cell.set_facecolor(bg); cell.set_text_props(color=TEXT)
            if c == 0:
                cell.set_text_props(color=GREEN if ok=="✓" else RED, fontsize=13)
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "6_sample_predictions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"    → {path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  MODEL REPORT GENERATOR")
    print("=" * 60)

    print("\n  Loading data ...")
    X, y_bin, y_rc, y_ft, le_rc, le_ft, meta, df = load_data()
    print(f"  Dataset: {X.shape[0]} windows × {X.shape[1]} features")
    print(f"  Failures: {y_bin.sum()}  |  Normal: {(y_bin==0).sum()}")

    print("\n  Loading models ...")
    rf, xgb, lgb, rc_model, ft_model = load_models()
    models = [rf, xgb, lgb]
    names  = ["RF", "XGB", "LGBM"]

    ens_pred, ens_prob, df_metrics = plot_metrics_summary(models, X, y_bin, meta)
    plot_confusion_binary(y_bin, ens_pred)
    plot_roc(models, names, X, y_bin, meta)
    plot_confusion_rootcause(rc_model, X, y_rc, le_rc)
    plot_confusion_failtype(ft_model, X, y_ft, le_ft)
    plot_sample_predictions(models, rc_model, ft_model,
                            X, y_bin, y_rc, y_ft, le_rc, le_ft, meta, df)

    # Text summary
    print(f"\n{'='*60}")
    print("  FINAL REPORT SUMMARY")
    print(f"{'='*60}")
    for _, row in df_metrics.iterrows():
        print(f"  {row['Model']:<16s}  Acc={row['Accuracy']:.4f}  "
              f"P={row['Precision']:.4f}  R={row['Recall']:.4f}  "
              f"F1={row['F1']:.4f}  AUC={row['AUC']:.4f}")

    rc_acc = accuracy_score(y_rc, rc_model.predict(X))
    ft_acc = accuracy_score(y_ft, ft_model.predict(X))
    print(f"\n  Root Cause Accuracy  : {rc_acc:.4f}")
    print(f"  Failure Type Accuracy: {ft_acc:.4f}")
    print(f"\n  📁 All plots saved → {REPORT_DIR}/")
    files = sorted(os.listdir(REPORT_DIR))
    for f in files:
        print(f"     {f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
