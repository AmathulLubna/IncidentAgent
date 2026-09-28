"""
Hindsight integration — the compulsory memory layer.

Three-tier split (application-level, one Hindsight bank per tier):

    {project}-episodic    RESOLVED incidents (as reported + human-confirmed outcome)
    {project}-semantic    distilled cross-incident patterns (from consolidation)
    {project}-procedural  promoted runbooks (symptom -> ordered fix steps -> success rate)

Retention timing (CRITICAL):
  An incident is NOT retained when it is created. The current incident must
  never be retrievable as "historical memory" during its own diagnosis, so the
  only episodic retain happens AFTER the human engineer confirms the root
  cause, the fix steps and the outcome (``POST /incidents/{id}/resolve``).

Async architecture (REQUIRED):
  The installed hindsight-client (0.10.1) exposes sync wrappers built on
  ``loop.run_until_complete`` internally. Those MUST NOT be called from
  FastAPI request handling (threadpool + event loop interactions produce
  "Timeout context manager should be used inside a task").
  All FastAPI paths therefore use the native ``a*`` coroutines
  (``aretain`` / ``arecall`` / ``acreate_bank`` / ``alist_memories``).
  The sync helpers are kept ONLY for offline scripts/REPLs.

Provenance:
  Every retain writes structured ``metadata`` + ``tags`` (incident id, service,
  timestamps, root-cause family, outcome). Live probes confirmed these
  round-trip through ``arecall``. Hindsight *rephrases* ``result.text``, so
  recall also fetches ``include_chunks=True`` to expose the original retained
  content via ``MemoryResult.raw_text`` — that raw text is what carries the
  ``[HISTORICAL INCIDENT #n RESOLVED]`` markers for memories written before
  metadata existed.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from hindsight_client import Hindsight

from .config import settings

logger = logging.getLogger("memory")

EPISODIC_BANK = f"{settings.HINDSIGHT_PROJECT}-episodic"
SEMANTIC_BANK = f"{settings.HINDSIGHT_PROJECT}-semantic"
PROCEDURAL_BANK = f"{settings.HINDSIGHT_PROJECT}-procedural"

ALL_BANKS = [EPISODIC_BANK, SEMANTIC_BANK, PROCEDURAL_BANK]

# Recall budgets: results are the rephrased summaries (+provenance), chunks are
# the original retained content (markers, ordered fix steps, raw symptoms).
RECALL_MAX_TOKENS = 2048
RECALL_MAX_CHUNK_TOKENS = 2048


@dataclass
class MemoryResult:
    """One recalled memory. ``raw_text`` is the original retained content when
    Hindsight exposed it as a chunk; ``text`` is Hindsight's rephrasing."""
    text: str
    score: float | None = None
    memory_id: str | None = None
    raw_text: str | None = None
    metadata: dict[str, str] | None = None
    tags: list[str] | None = None
    occurred_start: str | None = None


@dataclass
class RecallBatch:
    """Recall outcome that distinguishes *search failed* from *0 results*.

    Behaves like ``list[MemoryResult]`` (iter/len/bool/getitem) so existing
    call sites keep working, but ``error`` carries the real failure instead of
    silently degrading to an empty list.
    """
    results: list[MemoryResult] = field(default_factory=list)
    error: str | None = None

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __bool__(self) -> bool:
        return bool(self.results)

    def __getitem__(self, index):
        return self.results[index]


_client: Hindsight | None = None


def get_client() -> Hindsight:
    """Singleton client. Construction does no I/O, safe to call anywhere."""
    global _client
    if _client is None:
        _client = Hindsight(
            base_url=settings.HINDSIGHT_BASE_URL,
            api_key=settings.HINDSIGHT_API_KEY,
        )
    return _client


def config_summary() -> dict:
    """Safe diagnostics — never includes secret values."""
    return {
        "hindsight_url": settings.HINDSIGHT_BASE_URL,
        "hindsight_project": settings.HINDSIGHT_PROJECT,
        "hindsight_key_configured": bool(settings.HINDSIGHT_API_KEY),
        "groq_key_configured": bool(settings.GROQ_API_KEY),
        "banks": ALL_BANKS,
    }


async def ensure_banks() -> dict[str, str]:
    """Idempotently create the three banks. Returns per-bank status."""
    client = get_client()
    status: dict[str, str] = {}
    for bank_id, background in [
        (EPISODIC_BANK, "Human-resolved production incidents with confirmed outcomes."),
        (SEMANTIC_BANK, "Distilled cross-incident root-cause patterns."),
        (PROCEDURAL_BANK, "Confirmed, reusable incident runbooks."),
    ]:
        try:
            await client.acreate_bank(bank_id=bank_id, background=background)
            status[bank_id] = "ok"
        except Exception as e:  # noqa: BLE001 — bank may already exist; log and continue
            msg = str(e).lower()
            if "already exists" in msg or "conflict" in msg or "409" in msg:
                status[bank_id] = "exists"
            else:
                logger.warning("ensure_banks(%s) failed: %s: %s", bank_id, type(e).__name__, e)
                status[bank_id] = f"error: {type(e).__name__}"
    return status


# ---------------------------------------------------------------------------
# Retain (async — use from FastAPI). Only called AFTER human confirmation.
# ---------------------------------------------------------------------------

def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")[:48] or "unknown"


async def retain_episodic_resolution(
    incident_id: int,
    service: str,
    root_cause_family: str,
    fix_steps: list[str],
    outcome: str,
    *,
    symptom: str = "",
    occurred_at: str = "",
    predicted_root_cause: str | None = None,
) -> None:
    """Retain a RESOLVED incident as historical episodic memory.

    Called only after the engineer confirmed root cause + fix + outcome, so
    what is learned is *what actually happened and whether it worked* — not
    what the LLM guessed during diagnosis.
    """
    from . import layers  # local import: layers -> playbooks -> store (no cycle with memory)

    doc = next(
        d for d in layers.build_layer_docs(
            incident_id=incident_id, service=service, symptom=symptom, occurred_at=occurred_at,
            root_cause_family=root_cause_family, fix_steps=fix_steps, outcome=outcome,
            predicted_root_cause=predicted_root_cause,
        ) if d.layer == "episodic"
    )
    content = doc.content
    if outcome == "fail":
        content += (
            "\nNegative lesson: this fix did not resolve the incident — do NOT treat these "
            "steps as a proven solution; learn what was ruled out."
        )

    metadata = {
        "memory_type": "episodic",
        "incident_id": str(incident_id),
        "service": service,
        "status": "resolved",
        "outcome": outcome,
        "root_cause_family": root_cause_family or "",
        "layer": "episodic",
        "layer_key": f"episodic:{incident_id}",
        "category": doc.category,
    }
    if occurred_at:
        metadata["occurred_at"] = occurred_at
    if predicted_root_cause:
        metadata["predicted_root_cause_family"] = predicted_root_cause

    tags = [
        "historical-incident",
        "resolved",
        f"incident-{incident_id}",
        f"outcome-{outcome}",
        f"service-{_slug(service)}",
    ]
    await get_client().aretain(
        bank_id=EPISODIC_BANK,
        content=content,
        metadata=metadata,
        tags=tags,
    )


async def retain_layer_doc(doc) -> None:
    """Retain one per-incident SEMANTIC or PROCEDURAL layer document (see
    layers.py). Episodic documents go through ``retain_episodic_resolution``.
    Each layer lands in its own bank with layer-specific content."""
    bank = {"semantic": SEMANTIC_BANK, "procedural": PROCEDURAL_BANK, "episodic": EPISODIC_BANK}[doc.layer]
    await get_client().aretain(
        bank_id=bank,
        content=doc.content,
        metadata={
            "memory_type": doc.layer,
            "layer": doc.layer,
            "layer_key": doc.layer_key,
            "incident_id": str(doc.incident_id),
            "service": doc.service,
            "status": "resolved",
            "outcome": doc.outcome,
            "root_cause_family": doc.category,
        },
        tags=[f"layer-{doc.layer}", "resolved", f"incident-{doc.incident_id}",
              f"root-cause-{_slug(doc.category)}", f"service-{_slug(doc.service)}"],
    )


async def retain_semantic_pattern(
    root_cause_family: str, pattern_text: str, confidence_pct: float, window_n: int
) -> None:
    content = (
        f"[SEMANTIC PATTERN] root_cause_family={root_cause_family} confidence_pct={confidence_pct} "
        f"(observed in {confidence_pct:.0f}% of the last {window_n} relevant incidents)\n{pattern_text}"
    )
    await get_client().aretain(
        bank_id=SEMANTIC_BANK,
        content=content,
        metadata={
            "memory_type": "semantic",
            "root_cause_family": root_cause_family,
            "confidence_pct": str(confidence_pct),
            "window_n": str(window_n),
        },
        tags=["semantic-pattern", f"root-cause-{_slug(root_cause_family)}"],
    )


async def retain_procedural_runbook(
    root_cause_family: str, fix_steps: list[str], success_rate: float, attempts: int
) -> None:
    """Promote a runbook. Callers must only invoke this for human-confirmed
    SUCCESS outcomes that clear the configured thresholds."""
    steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(fix_steps))
    content = (
        f"[RUNBOOK] root_cause_family={root_cause_family} success_rate={success_rate:.0%} "
        f"(based on {attempts} confirmed attempts)\nOrdered fix steps:\n{steps_text}"
    )
    await get_client().aretain(
        bank_id=PROCEDURAL_BANK,
        content=content,
        metadata={
            "memory_type": "procedural",
            "root_cause_family": root_cause_family,
            "success_rate": f"{success_rate:.4f}",
            "attempts": str(attempts),
            "outcome": "success",
        },
        tags=["runbook", f"root-cause-{_slug(root_cause_family)}", "outcome-success"],
    )


# ---------------------------------------------------------------------------
# Recall (async). Failures are reported via RecallBatch.error — NEVER silently
# converted into "0 relevant memories".
# ---------------------------------------------------------------------------

async def _recall(bank_id: str, query: str, limit: int = 5) -> RecallBatch:
    try:
        result = await get_client().arecall(
            bank_id=bank_id,
            query=query,
            max_tokens=RECALL_MAX_TOKENS,
            include_chunks=True,          # original retained content (markers, ordered steps)
            max_chunk_tokens=RECALL_MAX_CHUNK_TOKENS,
        )
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        logger.warning("recall(%s) failed: %s", bank_id, msg)
        return RecallBatch(error=msg)

    chunks = getattr(result, "chunks", None) or {}
    out: list[MemoryResult] = []
    for m in (getattr(result, "results", None) or [])[:limit]:
        chunk_id = getattr(m, "chunk_id", None)
        raw = None
        if chunk_id and chunk_id in chunks:
            raw = getattr(chunks[chunk_id], "text", None) or None
        scores = getattr(m, "scores", None)
        out.append(MemoryResult(
            text=getattr(m, "text", "") or "",
            score=(scores.final if scores is not None else None),
            memory_id=getattr(m, "id", None),
            raw_text=raw,
            metadata=(getattr(m, "metadata", None) or None),
            tags=(getattr(m, "tags", None) or None),
            occurred_start=getattr(m, "occurred_start", None),
        ))
    return RecallBatch(results=out)


async def recall_episodic(query: str, limit: int = 5) -> RecallBatch:
    return await _recall(EPISODIC_BANK, query, limit)


async def recall_semantic(query: str, limit: int = 5) -> RecallBatch:
    return await _recall(SEMANTIC_BANK, query, limit)


async def recall_procedural(query: str, limit: int = 5) -> RecallBatch:
    return await _recall(PROCEDURAL_BANK, query, limit)


async def bank_sizes() -> dict[str, int | None]:
    """Best-effort bank sizes via alist_memories pagination total (limit=1)."""
    client = get_client()
    sizes: dict[str, int | None] = {}
    for bank_id in ALL_BANKS:
        try:
            resp = await client.alist_memories(bank_id=bank_id, limit=1)
            sizes[bank_id] = getattr(resp, "total", None)
        except Exception as e:  # noqa: BLE001
            logger.warning("bank_sizes(%s) failed: %s: %s", bank_id, type(e).__name__, e)
            sizes[bank_id] = None
    return sizes


async def health_check() -> dict[str, str]:
    """Per-bank reachability: ok / error:<type>. Never raises."""
    sizes = await bank_sizes()
    out: dict[str, str] = {}
    for bank_id in ALL_BANKS:
        out[bank_id] = "ok" if sizes.get(bank_id) is not None else "unreachable"
    # Short keys for the /health payload
    short = {
        "episodic": out[EPISODIC_BANK],
        "semantic": out[SEMANTIC_BANK],
        "procedural": out[PROCEDURAL_BANK],
    }
    return short


# ---------------------------------------------------------------------------
# Sync conveniences — scripts / REPLs ONLY. Never call from FastAPI.
# ---------------------------------------------------------------------------

def _warn_sync_context(name: str) -> None:
    import asyncio

    try:
        asyncio.get_running_loop()
        logger.warning(
            "%s (sync) called inside a running event loop — use the async variant instead", name
        )
    except RuntimeError:
        pass


def sync_recall(bank_id: str, query: str, limit: int = 5) -> RecallBatch:
    _warn_sync_context("sync_recall")
    result = get_client().recall(
        bank_id=bank_id, query=query, max_tokens=RECALL_MAX_TOKENS,
        include_chunks=True, max_chunk_tokens=RECALL_MAX_CHUNK_TOKENS,
    )
    chunks = getattr(result, "chunks", None) or {}
    out: list[MemoryResult] = []
    for m in (result.results or [])[:limit]:
        chunk_id = getattr(m, "chunk_id", None)
        raw = chunks[chunk_id].text if chunk_id and chunk_id in chunks else None
        scores = getattr(m, "scores", None)
        out.append(MemoryResult(
            text=m.text or "",
            score=(scores.final if scores is not None else None),
            memory_id=getattr(m, "id", None),
            raw_text=raw,
            metadata=(getattr(m, "metadata", None) or None),
            tags=(getattr(m, "tags", None) or None),
            occurred_start=getattr(m, "occurred_start", None),
        ))
    return RecallBatch(results=out)
