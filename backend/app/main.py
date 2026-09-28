import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import layers, memory, store
from .config import settings, validate_settings
from .routers import incidents, memory_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("incident_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for warning in validate_settings():
        logger.warning(warning)
    store.init_db()
    try:
        n = layers.backfill_local()
        if n:
            logger.info("Backfilled 3-layer memory rows for %d previously-resolved incident(s)", n)
    except Exception as e:  # noqa: BLE001
        logger.warning("layer backfill failed: %s: %s", type(e).__name__, e)
    try:
        status = await memory.ensure_banks()
        logger.info("Hindsight banks: %s", status)
    except Exception as e:  # noqa: BLE001
        logger.error("Could not reach Hindsight Cloud on startup: %s: %s", type(e).__name__, e)
    logger.info(
        "Config — groq_key=%s hindsight_key=%s project=%s url=%s",
        bool(settings.GROQ_API_KEY), bool(settings.HINDSIGHT_API_KEY),
        settings.HINDSIGHT_PROJECT, settings.HINDSIGHT_BASE_URL,
    )
    yield


app = FastAPI(
    title="Incident Response Agent",
    description="Diagnoses production incidents and gets measurably better over time via Hindsight memory.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(incidents.router)
app.include_router(memory_routes.router)


class ResetIn(BaseModel):
    confirm: bool = False


@app.post("/demo/reset")
async def demo_reset(body: ResetIn):
    """Dev-only reset of SQLite bookkeeping. Never touches Hindsight banks."""
    if not body.confirm:
        return {"ok": False, "error": "Pass {\"confirm\": true} to reset demo SQLite state."}
    store.reset_demo()
    return {"ok": True, "message": "SQLite demo state cleared. Hindsight banks untouched."}
