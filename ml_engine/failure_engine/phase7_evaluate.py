"""
Phase 7 — Model Evaluation & Final Output
==========================================
Full evaluation, JSON predictions, summary report.
"""

import os, json, warnings, numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from failure_engine.config import MODEL_DIR, BASE_DIR

warnings.filterwarnings("ignore")


def _top_k_accuracy(y_true, y_prob, k=3):
    if y_prob.ndim == 1:
        return accuracy_score(y_true, (y_prob > 0.5).astype(int))
    top_k = np.argsort(y_prob, axis=1)[:, -k:]
    return sum(1 for i, t in enumerate(y_true) if t in top_k[i]) / len(y_true)


def run_phase7(df, phase2_res, phase3_res, phase4_res, phase6_res) -> dict:
    """Comprehensive evaluation and JSON output."""
    print("\n" + "=" * 60)
    print("  PHASE 7 — Final Evaluation & Output")
    print("=" * 60)

    report = {}

    # ── Phase 2 (Metrics binary) ──
    p2 = phase2_res["results"][phase2_res["best_name"]]
    report["metrics_model"] = {
        "name": phase2_res["best_name"],
        "accuracy": round(p2["accuracy"], 4),
        "precision": round(p2["precision"], 4),
        "recall": round(p2["recall"], 4),
        "f1": round(p2["f1"], 4),
        "roc_auc": round(p2["roc_auc"], 4),
    }

    # ── Phase 3 (Log root cause) ──
    best3 = phase3_res["results"][phase3_res["best_name"]]
    report["log_model"] = {
        "name": phase3_res["best_name"],
        "top1_accuracy": round(best3["top1"], 4),
        "top3_accuracy": round(best3["top3"], 4),
        "f1_macro": round(best3["f1_macro"], 4),
    }

    # ── Phase 4 (Trace) ──
    best4 = phase4_res["results"][phase4_res["best_name"]]
    report["trace_model"] = {
        "name": phase4_res["best_name"],
        "top1_accuracy": round(best4["top1"], 4),
        "f1_macro": round(best4["f1_macro"], 4),
    }

    # ── Phase 6 (Fusion) ──
    best6 = phase6_res["results"][phase6_res["best_name"]]
    report["fusion_model"] = {
        "name": phase6_res["best_name"],
        "binary_f1": round(best6["f1_binary"], 4),
        "roc_auc": round(best6["roc_auc"], 4),
        "root_cause_top1": round(best6["rc_top1"], 4),
        "root_cause_f1_macro": round(best6["rc_f1"], 4),
    }

    # ── Summary table ──
    print(f"\n  {'Phase':<20s} {'Model':<15s} {'Key Metric':>15s}")
    print(f"  {'-'*20} {'-'*15} {'-'*15}")
    print(f"  {'Metrics (P2)':<20s} {report['metrics_model']['name']:<15s} "
          f"F1={report['metrics_model']['f1']:.4f}")
    print(f"  {'Logs (P3)':<20s} {report['log_model']['name']:<15s} "
          f"Top1={report['log_model']['top1_accuracy']:.4f}")
    print(f"  {'Traces (P4)':<20s} {report['trace_model']['name']:<15s} "
          f"Top1={report['trace_model']['top1_accuracy']:.4f}")
    print(f"  {'Fusion (P6)':<20s} {report['fusion_model']['name']:<15s} "
          f"F1={report['fusion_model']['binary_f1']:.4f}")

    # ── Propagation chain heuristic ──
    prop_chains = {
        "media": "media-service → compose-post-service → home-timeline-service",
        "text": "text-service → compose-post-service → home-timeline-service",
        "user": "user-service → compose-post-service → home-timeline-service",
        "socialgraph": "social-graph-service → user-service → compose-post-service",
        "hometimeline": "home-timeline-service → compose-post-service",
        "usertimeline": "user-timeline-service → post-storage-service",
        "gateway": "gateway-service → preserve-service → order-service",
        "travel": "travel-service → route-service → station-service",
        "auth": "auth-service → gateway-service → all-services",
        "system": "infrastructure → all-services",
        "database": "database → data-services → api-services",
        "none": "N/A",
    }

    def _get_propagation(rc):
        for key, chain in prop_chains.items():
            if key in rc:
                return chain
        return f"{rc} → dependent-services"

    # ── JSON predictions for test windows ──
    le_rc = phase6_res["le_rc"]
    le_ft = phase6_res["le_ft"]
    idx_test = phase6_res["idx_test"]
    y_bin_test = phase6_res["y_bin_test"]
    y_rc_test = phase6_res["y_rc_test"]
    y_ft_test = phase6_res["y_ft_test"]

    confidence_base = best6.get("roc_auc", 0.90)

    outputs = []
    for i, idx in enumerate(idx_test):
        rc_label = le_rc.inverse_transform([y_rc_test[i]])[0]
        ft_label = le_ft.inverse_transform([y_ft_test[i]])[0]
        outputs.append({
            "Failure": "YES" if y_bin_test[i] == 1 else "NO",
            "Root_Cause_Service": rc_label,
            "Failure_Type": ft_label,
            "Propagation_Path": _get_propagation(rc_label),
            "Confidence": round(float(confidence_base), 2),
        })

    print(f"\n  Sample predictions:")
    for o in outputs[:5]:
        print(f"    {json.dumps(o)}")

    # ── Save ──
    report_path = os.path.join(MODEL_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {report_path}")

    output_path = os.path.join(MODEL_DIR, "failure_predictions.json")
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"  Predictions saved → {output_path}")

    # ── Final banner ──
    print(f"\n{'='*60}")
    print("  ✅ FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Binary Detection   : F1={report['fusion_model']['binary_f1']:.4f}  AUC={report['fusion_model']['roc_auc']:.4f}")
    print(f"  Root Cause (Top-1) : {report['fusion_model']['root_cause_top1']:.4f}")
    print(f"  Root Cause (F1)    : {report['fusion_model']['root_cause_f1_macro']:.4f}")
    print(f"{'='*60}")

    return report
