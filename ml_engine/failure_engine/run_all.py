#!/usr/bin/env python3
"""
Failure Intelligence Engine — Orchestrator
============================================
Runs all 7 phases sequentially, logs timing, saves all models.
"""

import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from failure_engine.phase1_ingestion import ingest_all, load_cached
from failure_engine.phase2_metrics import run_phase2
from failure_engine.phase3_logs import run_phase3
from failure_engine.phase4_traces import run_phase4
from failure_engine.phase5_api_coverage import run_phase5
from failure_engine.phase6_fusion import run_phase6
from failure_engine.phase7_evaluate import run_phase7


def main():
    t0 = time.time()
    print("=" * 60)
    print("  MULTIMODAL FAILURE INTELLIGENCE ENGINE")
    print("=" * 60)

    # ── Phase 1 ──
    print("\n▶ PHASE 1: Data Ingestion & Alignment")
    ts = time.time()
    df = load_cached()
    if df is None:
        df = ingest_all()
    else:
        print(f"  Loaded cached data: {df.shape}")
    print(f"  Phase 1 done in {time.time()-ts:.1f}s")

    # ── Phase 2 ──
    print("\n▶ PHASE 2: Metrics-Based Failure Detection")
    ts = time.time()
    p2 = run_phase2(df)
    print(f"  Phase 2 done in {time.time()-ts:.1f}s")

    # ── Phase 3 ──
    print("\n▶ PHASE 3: Log-Based Root Cause Detection")
    ts = time.time()
    p3 = run_phase3(df)
    print(f"  Phase 3 done in {time.time()-ts:.1f}s")

    # ── Phase 4 ──
    print("\n▶ PHASE 4: Trace-Based Failure Propagation")
    ts = time.time()
    p4 = run_phase4(df)
    print(f"  Phase 4 done in {time.time()-ts:.1f}s")

    # ── Phase 5 ──
    print("\n▶ PHASE 5: API & Coverage Signal Integration")
    ts = time.time()
    p5 = run_phase5(df)
    print(f"  Phase 5 done in {time.time()-ts:.1f}s")

    # ── Phase 6 ──
    print("\n▶ PHASE 6: Multimodal Fusion Engine")
    ts = time.time()
    p6 = run_phase6(df)
    print(f"  Phase 6 done in {time.time()-ts:.1f}s")

    # ── Phase 7 ──
    print("\n▶ PHASE 7: Final Evaluation")
    ts = time.time()
    report = run_phase7(df, p2, p3, p4, p6)
    print(f"  Phase 7 done in {time.time()-ts:.1f}s")

    # ── Summary ──
    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Total pipeline time: {total:.1f}s ({total/60:.1f} min)")
    print(f"{'='*60}")

    # List all saved models
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    print(f"\n  Saved artifacts in {model_dir}/:")
    for f in sorted(os.listdir(model_dir)):
        fp = os.path.join(model_dir, f)
        sz = os.path.getsize(fp)
        print(f"    {f:<40s}  {sz/1024:.1f} KB")

    print(f"\n{'='*60}")
    print("  ✅ PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
