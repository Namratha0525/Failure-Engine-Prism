"""
Phase 1 — Data Ingestion & Alignment
=====================================
Load all 5 modalities, align to 30-second windows, produce unified feature dicts.
"""

import os, re, json, glob, warnings, traceback
import numpy as np
import pandas as pd
from collections import defaultdict
from failure_engine.config import (
    BASE_DIR, SN_ROOT, TT_ROOT, MODEL_DIR,
    SN_METRIC_FILES, TT_METRIC_NAMES, METRIC_KEYS, SN_SERVICES,
    label_binary, label_root_cause, label_failure_type, get_scenarios,
)

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════════════════

def _load_sn_metrics(scenario_dir: str) -> pd.DataFrame:
    series = {}
    for mk, fname in SN_METRIC_FILES.items():
        fp = os.path.join(scenario_dir, fname)
        if not os.path.isfile(fp):
            continue
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception:
            continue
        if "timestamp" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.dropna(subset=["timestamp", "value"], inplace=True)
        agg = df.groupby("timestamp")["value"].mean().reset_index()
        agg.rename(columns={"value": mk}, inplace=True)
        series[mk] = agg
    if not series:
        return pd.DataFrame()
    merged = None
    for k, a in series.items():
        merged = a if merged is None else pd.merge(merged, a, on="timestamp", how="outer")
    if merged is None or merged.empty:
        return pd.DataFrame()
    merged.sort_values("timestamp", inplace=True)
    merged.set_index("timestamp", inplace=True)
    for m in METRIC_KEYS:
        if m not in merged.columns:
            merged[m] = 0.0
    merged.fillna(0.0, inplace=True)
    return merged


def _load_tt_metrics(scenario_dir: str) -> pd.DataFrame:
    csvs = glob.glob(os.path.join(scenario_dir, "*.csv"))
    if not csvs:
        return pd.DataFrame()
    try:
        df = pd.read_csv(csvs[0], low_memory=False)
    except Exception:
        return pd.DataFrame()
    if "metric_name" not in df.columns or "timestamp" not in df.columns:
        return pd.DataFrame()
    rev = {v: k for k, v in TT_METRIC_NAMES.items()}
    df = df[df["metric_name"].isin(set(TT_METRIC_NAMES.values()))].copy()
    if df.empty:
        return pd.DataFrame()
    df["metric_key"] = df["metric_name"].map(rev)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df["timestamp"].dtype in ("int64", "float64"):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp", "value"], inplace=True)
    agg = df.groupby(["timestamp", "metric_key"])["value"].mean().reset_index()
    pivot = agg.pivot_table(index="timestamp", columns="metric_key", values="value", aggfunc="mean")
    pivot.sort_index(inplace=True)
    for m in METRIC_KEYS:
        if m not in pivot.columns:
            pivot[m] = 0.0
    pivot.fillna(0.0, inplace=True)
    return pivot


# ═════════════════════════════════════════════════════════════════════════
#  LOGS
# ═════════════════════════════════════════════════════════════════════════

_SN_LOG_RE = re.compile(
    r"\[(\d{4}-\w+-\d+\s[\d:.]+)\]\s+<(\w+)>:\s+\((.+?)\)\s+(.*)"
)
_TT_LOG_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}\s[\d:.]+)\s+(\w+)\s+\d+\s+---"
)
_TEMPLATE_CLEAN = [
    (re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*Z?"), "<TS>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "<UUID>"),
    (re.compile(r"\b[0-9a-f]{16,}\b", re.I), "<HEX>"),
    (re.compile(r"\b\d+\b"), "<NUM>"),
]


def _templatize(msg: str) -> str:
    for pat, rep in _TEMPLATE_CLEAN:
        msg = pat.sub(rep, msg)
    return msg


def _load_sn_logs(scenario_dir: str) -> dict:
    """Returns {service: [{timestamp, level, template}, ...]}"""
    result = {}
    for fname in os.listdir(scenario_dir):
        if not fname.endswith(".log"):
            continue
        svc = fname.replace("_.log", "").replace("_", "-").lower()
        entries = []
        fp = os.path.join(scenario_dir, fname)
        try:
            with open(fp, "r", errors="replace") as f:
                for line in f:
                    m = _SN_LOG_RE.match(line)
                    if m:
                        ts_str, level, _src, msg = m.groups()
                        try:
                            ts = pd.to_datetime(ts_str, format="%Y-%b-%d %H:%M:%S.%f")
                        except Exception:
                            ts = pd.to_datetime(ts_str, errors="coerce")
                        if pd.isna(ts):
                            continue
                        entries.append({
                            "timestamp": ts,
                            "level": level.upper(),
                            "template": _templatize(msg[:200]),
                        })
        except Exception:
            continue
        if entries:
            result[svc] = entries
    return result


def _load_tt_logs(scenario_dir: str) -> dict:
    """Returns {service: [{timestamp, level, template}, ...]}"""
    result = {}
    for item in os.listdir(scenario_dir):
        item_path = os.path.join(scenario_dir, item)
        if not os.path.isdir(item_path):
            # could be kubernetes_events or log_collection_report
            continue
        # extract service name from pod name (e.g. ts-auth-service-67c685bc55-77z9w → ts-auth-service)
        parts = item.split("-")
        svc_parts = []
        for p in parts:
            if len(p) >= 8 and re.match(r"^[0-9a-f]+$", p):
                break
            if re.match(r"^\d+$", p):
                break
            svc_parts.append(p)
        svc = "-".join(svc_parts) if svc_parts else item
        entries = []
        logs = glob.glob(os.path.join(item_path, "*.log"))
        for logf in logs:
            try:
                with open(logf, "r", errors="replace") as f:
                    for line in f:
                        if line.startswith((" ", "\t", "at ")):
                            continue  # skip stack traces
                        m = _TT_LOG_RE.match(line)
                        if m:
                            ts = pd.to_datetime(m.group(1), errors="coerce")
                            if pd.isna(ts):
                                continue
                            level = m.group(2).upper()
                            entries.append({
                                "timestamp": ts,
                                "level": level,
                                "template": _templatize(line[m.end():200]),
                            })
            except Exception:
                continue
        if entries:
            result[svc] = entries
    return result


def _aggregate_logs_to_features(log_dict: dict, window_start, window_end) -> dict:
    """Aggregate log entries within a window into features."""
    feats = {}
    total_errors = 0
    total_warns = 0
    total_logs = 0
    total_unique_templates = 0

    for svc, entries in log_dict.items():
        in_window = [e for e in entries if window_start <= e["timestamp"] < window_end]
        errs = sum(1 for e in in_window if e["level"] == "ERROR")
        warns = sum(1 for e in in_window if e["level"] in ("WARN", "WARNING"))
        infos = sum(1 for e in in_window if e["level"] == "INFO")
        templates = set(e["template"] for e in in_window)
        total_errors += errs
        total_warns += warns
        total_logs += len(in_window)
        total_unique_templates += len(templates)

    feats["log_total_count"] = total_logs
    feats["log_error_count"] = total_errors
    feats["log_warn_count"] = total_warns
    feats["log_unique_templates"] = total_unique_templates
    feats["log_error_ratio"] = total_errors / max(total_logs, 1)
    return feats


# ═════════════════════════════════════════════════════════════════════════
#  TRACES
# ═════════════════════════════════════════════════════════════════════════

def _load_sn_traces(scenario_dir: str) -> pd.DataFrame:
    """Load SN traces CSV into a DataFrame."""
    fp = os.path.join(scenario_dir, "all_traces.csv")
    if not os.path.isfile(fp):
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["duration_us"] = pd.to_numeric(df.get("duration_us", pd.Series()), errors="coerce")
    return df


def _load_tt_traces(scenario_dir: str) -> pd.DataFrame:
    """Load TT SkyWalking traces JSON into a normalised DataFrame."""
    jsons = glob.glob(os.path.join(scenario_dir, "*.json"))
    if not jsons:
        return pd.DataFrame()
    try:
        with open(jsons[0], "r") as f:
            data = json.load(f)
    except Exception:
        return pd.DataFrame()

    rows = []
    traces = data.get("traces", [])
    for t in traces:
        trace_id = t.get("summary", {}).get("trace_id", "")
        for span in t.get("spans", []):
            rows.append({
                "trace_id": trace_id,
                "span_id": span.get("spanId", ""),
                "parent_span_id": span.get("parentSpanId", ""),
                "service": span.get("serviceCode", ""),
                "operation": span.get("endpointName", ""),
                "start_time": pd.to_datetime(span.get("startTime", 0), unit="ms", errors="coerce"),
                "duration_us": int(span.get("duration", 0)) * 1000,  # ms → us
                "is_error": span.get("isError", False),
            })
    return pd.DataFrame(rows)


def _trace_graph_features(trace_df: pd.DataFrame) -> dict:
    """Compute trace-based features: per-service latency stats, call graph edges."""
    if trace_df.empty:
        return {"trace_service_count": 0, "trace_avg_duration": 0,
                "trace_error_rate": 0, "trace_edge_count": 0}

    feats = {}
    feats["trace_service_count"] = trace_df["service"].nunique() if "service" in trace_df.columns else 0
    feats["trace_avg_duration"] = float(trace_df["duration_us"].mean()) if "duration_us" in trace_df.columns else 0
    if "is_error" in trace_df.columns:
        feats["trace_error_rate"] = float(trace_df["is_error"].astype(float).mean())
    elif "http_status_code" in trace_df.columns:
        trace_df["http_status_code"] = pd.to_numeric(trace_df["http_status_code"], errors="coerce")
        feats["trace_error_rate"] = float((trace_df["http_status_code"] >= 400).mean())
    else:
        feats["trace_error_rate"] = 0
    feats["trace_edge_count"] = int(trace_df["parent_span_id"].notna().sum()) if "parent_span_id" in trace_df.columns else 0
    feats["trace_max_duration"] = float(trace_df["duration_us"].max()) if "duration_us" in trace_df.columns else 0
    feats["trace_std_duration"] = float(trace_df["duration_us"].std()) if "duration_us" in trace_df.columns else 0
    return feats


# ═════════════════════════════════════════════════════════════════════════
#  API RESPONSES
# ═════════════════════════════════════════════════════════════════════════

def _load_sn_api(scenario_dir: str) -> dict:
    feats = {}
    # response_summary.json
    summary_fp = os.path.join(scenario_dir, "response_summary.json")
    if os.path.isfile(summary_fp):
        try:
            with open(summary_fp) as f:
                s = json.load(f)
            stats = s.get("statistics", {})
            feats["api_total_requests"] = stats.get("total_requests", 0)
            feats["api_success_rate"] = stats.get("success_rate_percent", 0) / 100.0
            feats["api_error_count"] = stats.get("failed_requests", 0)
            lat = s.get("latency_statistics", {})
            feats["api_mean_latency"] = lat.get("mean_ms", 0)
            feats["api_p95_latency"] = lat.get("p95_ms", 0)
            feats["api_p99_latency"] = lat.get("p99_ms", 0)
        except Exception:
            pass
    # status_code_distribution.csv
    dist_fp = os.path.join(scenario_dir, "status_code_distribution.csv")
    if os.path.isfile(dist_fp):
        try:
            df = pd.read_csv(dist_fp)
            total = df["count"].sum() if "count" in df.columns else 1
            for _, row in df.iterrows():
                code = int(row.get("status_code", 0))
                if 400 <= code < 500:
                    feats["api_4xx_ratio"] = feats.get("api_4xx_ratio", 0) + row.get("count", 0) / max(total, 1)
                elif code >= 500:
                    feats["api_5xx_ratio"] = feats.get("api_5xx_ratio", 0) + row.get("count", 0) / max(total, 1)
        except Exception:
            pass
    return feats


def _load_tt_api(scenario_dir: str) -> dict:
    feats = {}
    # find api_responses.jsonl under date subdir
    jsonl_files = glob.glob(os.path.join(scenario_dir, "**", "api_responses.jsonl"), recursive=True)
    if not jsonl_files:
        return feats
    try:
        statuses = []
        latencies = []
        with open(jsonl_files[0], "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    statuses.append(int(obj.get("status_code", 0)))
                    lat = obj.get("latency_ms")
                    if lat is None:
                        # compute from timestamps
                        s = obj.get("request_start_time", 0)
                        e = obj.get("request_end_time", 0)
                        lat = (e - s) * 1000 if s and e else 0
                    latencies.append(float(lat))
                except Exception:
                    continue
        total = len(statuses) or 1
        feats["api_total_requests"] = total
        feats["api_success_rate"] = sum(1 for c in statuses if 200 <= c < 300) / total
        feats["api_error_count"] = sum(1 for c in statuses if c >= 400)
        feats["api_4xx_ratio"] = sum(1 for c in statuses if 400 <= c < 500) / total
        feats["api_5xx_ratio"] = sum(1 for c in statuses if c >= 500) / total
        if latencies:
            feats["api_mean_latency"] = float(np.mean(latencies))
            feats["api_p95_latency"] = float(np.percentile(latencies, 95))
            feats["api_p99_latency"] = float(np.percentile(latencies, 99))
    except Exception:
        pass
    return feats


# ═════════════════════════════════════════════════════════════════════════
#  COVERAGE
# ═════════════════════════════════════════════════════════════════════════

def _load_sn_coverage(scenario_dir: str) -> dict:
    feats = {}
    total_lines = 0
    total_exec = 0
    for svc_dir in os.listdir(scenario_dir):
        svc_path = os.path.join(scenario_dir, svc_dir)
        if not os.path.isdir(svc_path):
            continue
        summary_fp = os.path.join(svc_path, "coverage-summary.txt")
        if not os.path.isfile(summary_fp):
            continue
        try:
            with open(summary_fp) as f:
                content = f.read()
            # look for TOTAL line
            m = re.search(r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%", content)
            if m:
                lines = int(m.group(1))
                ex = int(m.group(2))
                total_lines += lines
                total_exec += ex
        except Exception:
            continue

    feats["coverage_total_lines"] = total_lines
    feats["coverage_exec_lines"] = total_exec
    feats["coverage_pct"] = total_exec / max(total_lines, 1) * 100
    return feats


def _load_tt_coverage(scenario_dir: str) -> dict:
    feats = {}
    exec_files = glob.glob(os.path.join(scenario_dir, "**", "*.exec"), recursive=True)
    total_size = sum(os.path.getsize(f) for f in exec_files) if exec_files else 0
    feats["coverage_exec_file_count"] = len(exec_files)
    feats["coverage_total_exec_size"] = total_size
    feats["coverage_avg_exec_size"] = total_size / max(len(exec_files), 1)
    return feats


# ═════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING (metrics windows)
# ═════════════════════════════════════════════════════════════════════════

def _slope(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series), dtype=float)
    try:
        return float(np.polyfit(x, series.values.astype(float), 1)[0])
    except Exception:
        return 0.0


def _rate_of_change(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    return float(series.iloc[-1] - series.iloc[0])


def _metric_window_features(window_df: pd.DataFrame) -> dict:
    feats = {}
    for mk in METRIC_KEYS:
        col = window_df[mk] if mk in window_df.columns else pd.Series([0.0])
        feats[f"{mk}_mean"] = float(col.mean()) if len(col) > 0 else 0.0
        feats[f"{mk}_std"] = float(col.std()) if len(col) > 1 else 0.0
        feats[f"{mk}_max"] = float(col.max()) if len(col) > 0 else 0.0
        feats[f"{mk}_min"] = float(col.min()) if len(col) > 0 else 0.0
        feats[f"{mk}_slope"] = _slope(col)
        feats[f"{mk}_roc"] = _rate_of_change(col)
    return feats


# ═════════════════════════════════════════════════════════════════════════
#  MAIN INGESTION
# ═════════════════════════════════════════════════════════════════════════

def _find_matching_dir(root: str, modality: str, scenario_metric_name: str) -> str | None:
    """Find the matching directory for a modality given a metric scenario name.
    Dirs share the same experiment prefix but have different modality suffixes."""
    mod_dir = os.path.join(root, modality)
    if not os.path.isdir(mod_dir):
        return None
    # extract experiment prefix (everything before _metrics_ or _logs_ etc.)
    prefix_patterns = [
        r"^(.+?)_metrics_",
        r"^(.+?)_logs_",
        r"^(.+?)_traces_",
        r"^(.+?)_openapi_",
        r"^(.+?)_coverage_",
    ]
    exp_prefix = scenario_metric_name
    for pat in prefix_patterns:
        m = re.match(pat, scenario_metric_name)
        if m:
            exp_prefix = m.group(1)
            break
    # Also try the raw name for TT
    for d in os.listdir(mod_dir):
        dp = os.path.join(mod_dir, d)
        if not os.path.isdir(dp):
            continue
        if d.startswith(exp_prefix) or exp_prefix.startswith(d) or d == scenario_metric_name:
            return dp
    return None


def ingest_all() -> pd.DataFrame:
    """
    Load all modalities from SN_data and TT_data, align to 30s windows.
    Returns a DataFrame where each row is a window with all features + labels.
    """
    all_records = []

    for source_tag, root, load_metrics, load_logs, load_traces, load_api, load_cov in [
        ("SN", SN_ROOT, _load_sn_metrics, _load_sn_logs, _load_sn_traces, _load_sn_api, _load_sn_coverage),
        ("TT", TT_ROOT, _load_tt_metrics, _load_tt_logs, _load_tt_traces, _load_tt_api, _load_tt_coverage),
    ]:
        metric_scenarios = get_scenarios(root, "metric_data")
        print(f"\n[{source_tag}] Loading {len(metric_scenarios)} scenarios ...")

        for sc_name, sc_path in metric_scenarios:
            lbl_bin = label_binary(sc_name)
            lbl_rc = label_root_cause(sc_name)
            lbl_ft = label_failure_type(sc_name)
            tag = "NORMAL" if lbl_bin == 0 else "FAILURE"
            print(f"  {sc_name[:55]:55s} [{tag}]")

            # ── Metrics ──
            metrics_df = load_metrics(sc_path)

            # ── Logs ──
            log_dir = _find_matching_dir(root, "log_data", sc_name)
            log_dict = load_logs(log_dir) if log_dir else {}

            # ── Traces ──
            trace_dir = _find_matching_dir(root, "trace_data", sc_name)
            trace_df = load_traces(trace_dir) if trace_dir else pd.DataFrame()
            trace_feats = _trace_graph_features(trace_df)

            # ── API ──
            api_dir = _find_matching_dir(root, "api_responses", sc_name)
            api_feats = load_api(api_dir) if api_dir else {}

            # ── Coverage ──
            cov_dir = _find_matching_dir(root, "coverage_data", sc_name)
            cov_feats = load_cov(cov_dir) if cov_dir else {}

            # ── Create windows from metrics ──
            if metrics_df.empty:
                continue
            windows = metrics_df.resample("30s")
            window_count = 0
            for wstart, wdf in windows:
                if wdf.empty:
                    continue
                wend = wstart + pd.Timedelta(seconds=30)
                record = {
                    "window_start": str(wstart),
                    "scenario": sc_name,
                    "source": source_tag,
                    "label_binary": lbl_bin,
                    "label_root_cause": lbl_rc,
                    "label_failure_type": lbl_ft,
                }
                # metric features
                record.update(_metric_window_features(wdf))
                # log features (aggregated for this window)
                record.update(_aggregate_logs_to_features(log_dict, wstart, wend))
                # trace features (scenario-level, same for all windows)
                record.update(trace_feats)
                # api features (scenario-level)
                record.update(api_feats)
                # coverage features (scenario-level)
                record.update(cov_feats)
                all_records.append(record)
                window_count += 1
            print(f"    → {window_count} windows")

    if not all_records:
        print("[ERROR] No data loaded.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    # fill NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)

    print(f"\n{'='*60}")
    print(f"  Total windows : {len(df)}")
    print(f"  Normal (0)    : {(df['label_binary'] == 0).sum()}")
    print(f"  Failure (1)   : {(df['label_binary'] == 1).sum()}")
    print(f"  Root causes   : {df['label_root_cause'].nunique()}")
    print(f"  Failure types : {df['label_failure_type'].nunique()}")
    print(f"  Features      : {len([c for c in df.columns if c not in ('window_start','scenario','source','label_binary','label_root_cause','label_failure_type')])}")
    print(f"{'='*60}")

    # Save for reuse
    cache_path = os.path.join(MODEL_DIR, "ingested_data.parquet")
    df.to_parquet(cache_path, index=False)
    print(f"  Cached → {cache_path}")
    return df


def load_cached() -> pd.DataFrame | None:
    """Load cached ingested data if available."""
    cache_path = os.path.join(MODEL_DIR, "ingested_data.parquet")
    if os.path.isfile(cache_path):
        return pd.read_parquet(cache_path)
    return None


# ── Helpers for downstream phases ────────────────────────────────────────
META_COLS = ["window_start", "scenario", "source", "label_binary",
             "label_root_cause", "label_failure_type"]

METRIC_FEATURE_COLS = [f"{mk}_{stat}" for mk in METRIC_KEYS
                       for stat in ["mean", "std", "max", "min", "slope", "roc"]]

LOG_FEATURE_COLS = ["log_total_count", "log_warn_count",
                    "log_unique_templates"]

TRACE_FEATURE_COLS = ["trace_service_count", "trace_avg_duration",
                      "trace_edge_count", "trace_max_duration", "trace_std_duration"]

API_FEATURE_COLS = ["api_total_requests", "api_mean_latency",
                    "api_p95_latency", "api_p99_latency"]

COVERAGE_FEATURE_COLS_SN = ["coverage_total_lines", "coverage_exec_lines", "coverage_pct"]
COVERAGE_FEATURE_COLS_TT = ["coverage_exec_file_count", "coverage_total_exec_size", "coverage_avg_exec_size"]

ALL_FEATURE_COLS = (METRIC_FEATURE_COLS + LOG_FEATURE_COLS + TRACE_FEATURE_COLS +
                    API_FEATURE_COLS + COVERAGE_FEATURE_COLS_SN + COVERAGE_FEATURE_COLS_TT)


if __name__ == "__main__":
    df = ingest_all()
    print(f"\nIngestion complete. Shape: {df.shape}")
