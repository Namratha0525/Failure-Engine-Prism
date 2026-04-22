"""
Shared configuration: paths, label maps, scenario metadata.
"""
import os, re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "failure_engine", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Dataset roots ────────────────────────────────────────────────────────
SN_ROOT = os.path.join(BASE_DIR, "SN_data")
TT_ROOT = os.path.join(BASE_DIR, "TT_data")

MODALITIES = ["metric_data", "log_data", "trace_data", "api_responses", "coverage_data"]

# ── Metrics mappings ─────────────────────────────────────────────────────
SN_METRIC_FILES = {
    "cpu": "socialnet_container_cpu.csv",
    "memory": "socialnet_container_memory.csv",
    "latency": "system_load1.csv",
    "throughput": "jaeger_spans_rate.csv",
    "error_rate": "system_network_errors.csv",
}

TT_METRIC_NAMES = {
    "cpu": "container_cpu_usage_seconds_total",
    "memory": "container_memory_working_set_bytes",
    "latency": "node_load5",
    "throughput": "http_requests_total",
    "error_rate": "container_network_receive_errors_total",
}

METRIC_KEYS = ["cpu", "memory", "latency", "throughput", "error_rate"]

# ── SN services ──────────────────────────────────────────────────────────
SN_SERVICES = [
    "compose-post-service", "home-timeline-service", "media-service",
    "nginx-web-server", "post-storage-service", "social-graph-service",
    "text-service", "unique-id-service", "url-shorten-service",
    "user-mention-service", "user-service", "user-timeline-service",
]

# ── Label extraction ─────────────────────────────────────────────────────
_ROOT_CAUSE_PATTERNS = [
    # SN patterns
    (r"Svc_Kill_(\w+)", lambda m: m.group(1).lower()),
    (r"Code_Stop_(\w+Service)", lambda m: m.group(1).lower().replace("service", "-service")),
    (r"DB_Redis_CacheLimit_(\w+)", lambda m: m.group(1).lower().replace("timeline", "-timeline").replace("graph", "-graph")),
    # TT patterns
    (r"Lv_S_KILLPOD", lambda m: "system"),
    (r"Lv_S_HTTPABORT", lambda m: "gateway-service"),
    (r"Lv_S_DNSFAIL", lambda m: "system"),
    (r"Lv_C_exception_injection", lambda m: "travel-service"),
    (r"Lv_C_security_check", lambda m: "auth-service"),
    (r"Lv_C_travel_detail_failure", lambda m: "travel-service"),
    (r"Lv_D_CONNECTION_POOL", lambda m: "database"),
    (r"Lv_D_TRANSACTION", lambda m: "database"),
    (r"Lv_D_cachelimit", lambda m: "database"),
    (r"Lv_P_CPU", lambda m: "system"),
    (r"Lv_P_DISKIO", lambda m: "system"),
    (r"Lv_P_NETLOSS", lambda m: "system"),
    # Generic performance / normal
    (r"Perf_", lambda m: "system"),
    (r"Normal", lambda m: "none"),
]

_FAILURE_TYPE_PATTERNS = [
    (r"Svc_Kill", "service_kill"),
    (r"Code_Stop", "code_stop"),
    (r"DB_Redis_CacheLimit", "cache_limit"),
    (r"Perf_CPU", "cpu_contention"),
    (r"Perf_Disk", "disk_stress"),
    (r"Perf_Network", "network_loss"),
    (r"Lv_S_KILLPOD", "pod_kill"),
    (r"Lv_S_HTTPABORT", "http_abort"),
    (r"Lv_S_DNSFAIL", "dns_failure"),
    (r"Lv_C_exception", "exception_injection"),
    (r"Lv_C_security", "security_check"),
    (r"Lv_C_travel", "travel_failure"),
    (r"Lv_D_CONNECTION", "connection_pool"),
    (r"Lv_D_TRANSACTION", "transaction_timeout"),
    (r"Lv_D_cachelimit", "cache_limit"),
    (r"Lv_P_CPU", "cpu_contention"),
    (r"Lv_P_DISKIO", "disk_stress"),
    (r"Lv_P_NETLOSS", "network_loss"),
    (r"Normal", "normal"),
]

def label_binary(dirname: str) -> int:
    return 0 if dirname.lower().startswith("normal") else 1

def label_root_cause(dirname: str) -> str:
    for pat, fn in _ROOT_CAUSE_PATTERNS:
        m = re.search(pat, dirname)
        if m:
            return fn(m)
    return "unknown"

def label_failure_type(dirname: str) -> str:
    for pat, ftype in _FAILURE_TYPE_PATTERNS:
        if re.search(pat, dirname):
            return ftype
    return "unknown"

def get_scenarios(root: str, modality: str) -> list[tuple[str, str]]:
    """Return list of (scenario_name, full_path) for a modality."""
    mod_dir = os.path.join(root, modality)
    if not os.path.isdir(mod_dir):
        return []
    return sorted(
        [(d, os.path.join(mod_dir, d)) for d in os.listdir(mod_dir)
         if os.path.isdir(os.path.join(mod_dir, d))]
    )

# ── Device ───────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── All unique root cause labels (for encoding) ─────────────────────────
def build_label_maps():
    """Build label encodings from all scenario dirs."""
    root_causes = set()
    failure_types = set()
    for root in [SN_ROOT, TT_ROOT]:
        for sc_name, _ in get_scenarios(root, "metric_data"):
            root_causes.add(label_root_cause(sc_name))
            failure_types.add(label_failure_type(sc_name))
    rc_list = sorted(root_causes)
    ft_list = sorted(failure_types)
    return (
        {v: i for i, v in enumerate(rc_list)},
        {v: i for i, v in enumerate(ft_list)},
        rc_list, ft_list
    )
