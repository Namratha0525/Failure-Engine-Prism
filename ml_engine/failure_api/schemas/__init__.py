"""Pydantic request/response schemas."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List

# ── Request ──────────────────────────────────────────────────────────────
class MetricsWindow(BaseModel):
    window_start: Optional[str] = None

    # Metric features (30-second window aggregates)
    cpu_mean: float = Field(default=0.0, ge=0)
    cpu_std: float = Field(default=0.0, ge=0)
    cpu_max: float = Field(default=0.0, ge=0)
    cpu_min: float = Field(default=0.0, ge=0)
    cpu_slope: float = 0.0
    cpu_roc: float = 0.0

    memory_mean: float = Field(default=0.0, ge=0)
    memory_std: float = Field(default=0.0, ge=0)
    memory_max: float = Field(default=0.0, ge=0)
    memory_min: float = Field(default=0.0, ge=0)
    memory_slope: float = 0.0
    memory_roc: float = 0.0

    latency_mean: float = Field(default=0.0, ge=0)
    latency_std: float = Field(default=0.0, ge=0)
    latency_max: float = Field(default=0.0, ge=0)
    latency_min: float = Field(default=0.0, ge=0)
    latency_slope: float = 0.0
    latency_roc: float = 0.0

    throughput_mean: float = Field(default=0.0, ge=0)
    throughput_std: float = Field(default=0.0, ge=0)
    throughput_max: float = Field(default=0.0, ge=0)
    throughput_min: float = Field(default=0.0, ge=0)
    throughput_slope: float = 0.0
    throughput_roc: float = 0.0

    error_rate_mean: float = Field(default=0.0, ge=0)
    error_rate_std: float = Field(default=0.0, ge=0)
    error_rate_max: float = Field(default=0.0, ge=0)
    error_rate_min: float = Field(default=0.0, ge=0)
    error_rate_slope: float = 0.0
    error_rate_roc: float = 0.0

    # Log features (optional)
    log_total_count: float = 0.0
    log_error_count: float = 0.0
    log_warn_count: float = 0.0
    log_unique_templates: float = 0.0
    log_error_ratio: float = 0.0

    # Trace features (optional)
    trace_service_count: float = 0.0
    trace_avg_duration: float = 0.0
    trace_error_rate: float = 0.0
    trace_edge_count: float = 0.0
    trace_max_duration: float = 0.0
    trace_std_duration: float = 0.0

    # API features (optional)
    api_total_requests: float = 0.0
    api_success_rate: float = 1.0
    api_error_count: float = 0.0
    api_4xx_ratio: float = 0.0
    api_5xx_ratio: float = 0.0
    api_mean_latency: float = 0.0
    api_p95_latency: float = 0.0
    api_p99_latency: float = 0.0

    # Coverage features (optional)
    coverage_total_lines: float = 0.0
    coverage_exec_lines: float = 0.0
    coverage_pct: float = 0.0
    coverage_exec_file_count: float = 0.0
    coverage_total_exec_size: float = 0.0
    coverage_avg_exec_size: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "window_start": "2026-03-20T13:30:00",
                "cpu_mean": 0.87, "cpu_roc": 0.45,
                "memory_mean": 0.92, "memory_roc": 0.21,
                "latency_mean": 4.2, "error_rate_mean": 0.18,
                "log_error_count": 47.0, "log_error_ratio": 0.31,
                "trace_error_rate": 0.22, "trace_avg_duration": 8500.0,
            }
        }


class BatchRequest(BaseModel):
    windows: List[MetricsWindow]


# ── Response ─────────────────────────────────────────────────────────────
class ExplanationItem(BaseModel):
    feature: str
    impact: float

class DebugInfo(BaseModel):
    probabilities: Dict[str, float]
    failure_votes: str

class PredictionResponse(BaseModel):
    timestamp: str
    Failure: str                      # "YES" | "NO"
    Confidence: str                   # "HIGH (0.93)" | "MEDIUM (0.72)" | "LOW (0.51)"
    Root_Cause_Service: str
    Failure_Type: str
    Propagation_Path: str
    Action: str                       # "ALERT" | "MONITOR" | "SUPPRESSED"
    Suppressed: bool
    model_version: str
    explanation: Optional[List[ExplanationItem]] = None
    debug: Optional[DebugInfo] = None

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int
    failures_detected: int
    alerts_fired: int
