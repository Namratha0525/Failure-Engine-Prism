"""
Failure Intelligence Engine — FastAPI Application
==================================================
"""
import logging, time, os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse

from failure_api.core.config import settings
from failure_api.core.ensemble import init_predictor
from failure_api.core.alert_state import init_alert_state
from failure_api.routers import health, predict, admin

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.debug_mode else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Startup / Shutdown ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Starting Failure Prediction API {settings.model_version}")
    log.info(f"Model dir: {settings.model_dir}")
    try:
        init_predictor(settings.model_dir)
        init_alert_state(settings.alert_cooldown_secs, settings.alert_max_repeat)
        log.info("Models loaded successfully ✓")
    except FileNotFoundError as e:
        log.error(f"FATAL: {e}")
        raise
    yield
    log.info("Shutdown complete.")

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Microservice Failure Prediction API",
    description=(
        "Enterprise-grade failure detection engine using calibrated RF + XGB + LightGBM ensemble. "
        "2-of-3 majority voting with alert fatigue suppression."
    ),
    version=settings.model_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_key(key: str = Depends(api_key_header)):
    if key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return key

# ── Latency middleware ────────────────────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Inference-Time-Ms"] = f"{ms:.1f}"
    if request.url.path.startswith("/api"):
        log.debug(f"{request.method} {request.url.path}  {response.status_code}  {ms:.1f}ms")
    return response

# ── Routers ───────────────────────────────────────────────────────────────
app.include_router(health.router)                                               # /health  /ready
app.include_router(predict.router, prefix="/api/v1", dependencies=[Depends(verify_key)])  # /api/v1/predict
app.include_router(admin.router,   prefix="/admin",  dependencies=[Depends(verify_key)])  # /admin/*

# ── Root ──────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({
        "name": "Failure Prediction API",
        "version": settings.model_version,
        "docs": "/docs",
        "health": "/health",
    })
