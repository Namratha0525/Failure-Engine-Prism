"""
Phase 5 — API & Coverage Signal Integration
=============================================
Extract structured feature vectors from API responses and coverage data.
Already done inline in phase1_ingestion; this module provides utilities
to transform and normalise them for the fusion stage.
"""

import numpy as np
import pandas as pd
from failure_engine.phase1_ingestion import API_FEATURE_COLS, COVERAGE_FEATURE_COLS_SN, COVERAGE_FEATURE_COLS_TT


ALL_AUX_COLS = API_FEATURE_COLS + COVERAGE_FEATURE_COLS_SN + COVERAGE_FEATURE_COLS_TT


def run_phase5(df: pd.DataFrame) -> dict:
    """Extract and normalise API + coverage features."""
    print("\n" + "=" * 60)
    print("  PHASE 5 — API & Coverage Signal Integration")
    print("=" * 60)

    avail = [c for c in ALL_AUX_COLS if c in df.columns]
    X = df[avail].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardise
    mu = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mu) / std

    print(f"  API + Coverage features: {len(avail)}")
    print(f"  Non-zero features: {(X.sum(axis=0) != 0).sum()} / {len(avail)}")

    # Feature summary
    for c in avail:
        vals = df[c].values
        nz = (vals != 0).sum()
        if nz > 0:
            print(f"    {c:<35s}  mean={vals.mean():.2f}  std={vals.std():.2f}  non-zero={nz}")

    return {
        "feature_cols": avail,
        "X": X, "X_norm": X_norm,
        "mean": mu, "std": std,
    }


if __name__ == "__main__":
    from failure_engine.phase1_ingestion import load_cached, ingest_all
    df = load_cached()
    if df is None:
        df = ingest_all()
    run_phase5(df)
