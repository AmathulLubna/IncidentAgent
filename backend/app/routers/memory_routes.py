import sqlite3

from fastapi import APIRouter

from .. import layers, memory, store, taxonomy
from ..config import settings
from ..groq_client import groq_health
from ..schemas import StatsOut

router = APIRouter(tags=["memory"])


@router.get("/stats", response_model=StatsOut)
async def get_stats():
    stats = store.get_stats()
    sizes = await memory.bank_sizes()
    local = store.layer_counts()  # fall back to local mirror when a bank size is unavailable
    return StatsOut(
        total_incidents=stats["total_incidents"],
        memory_hit_rate_last_5=stats["memory_hit_rate_last_5"],
        memory_hit_rate_series=stats["memory_hit_rate_series"],
        episodic_count=sizes.get(memory.EPISODIC_BANK) or local.get("episodic", 0),
        semantic_count=sizes.get(memory.SEMANTIC_BANK) or local.get("semantic", 0),
        procedural_count=sizes.get(memory.PROCEDURAL_BANK) or local.get("procedural", 0),
        root_cause_family_counts=stats["root_cause_family_counts"],
    )


_RECALL = {
    "episodic": memory.recall_episodic,
    "semantic": memory.recall_semantic,
    "procedural": memory.recall_procedural,
}


def _hindsight_result(layer: str, q: str, h, cats: set[str]) -> dict | None:
    """Normalise one Hindsight hit into the layer-specific explorer shape.
    Prefer the ORIGINAL retained document (raw_text) over Hindsight's rephrasing
    — the rephrasing is what made different layers look alike."""
    body = h.raw_text or h.text or ""
    text_cats = {c for c, sc in taxonomy.classify(body) if sc >= 2}
    q_tokens, b_tokens = taxonomy.tokens(q), taxonomy.tokens(body)
    if not (q_tokens & b_tokens) and not (cats & text_cats):
        return None  # fuzzy-recall junk: shares nothing with the query
    key = layers.layer_key_of(body)
    fields = layers.parse_fields(layer, body) if key else {}
    gist = next((ln[5:].strip() for ln in body.splitlines() if ln.startswith("Gist:")), None) \
        or " ".join(body.split())[:170]
    meta = h.metadata or {}
    fam = meta.get("root_cause_family") or (sorted(text_cats)[0] if text_cats else None)
    return {
        "layer": layer, "source": "hindsight", "layer_key": key,
        "incident_id": int(meta["incident_id"]) if str(meta.get("incident_id", "")).isdigit() else None,
        "category": fam, "category_label": taxonomy.label_for(fam) if fam else None,
        "service": meta.get("service"), "outcome": meta.get("outcome"),
        "title": None, "gist": gist, "fields": fields, "text": body,
        "score": h.score, "memory_id": h.memory_id, "legacy": key is None,
    }


async def _search_layer(memory_type: str, q: str, limit: int) -> dict:
    fn = _RECALL[memory_type]
    cats = {c for c, sc in taxonomy.classify(q) if sc >= 2}
    local = layers.search_local(memory_type, q, limit=limit)
    error = None
    remote: list[dict] = []
    try:
        batch = await fn(q, limit)
        error = getattr(batch, "error", None)
        if not error:
            have = {r["layer_key"] for r in local if r.get("layer_key")}
            seen_text: set[str] = set()
            for h in batch:
                item = _hindsight_result(memory_type, q, h, cats)
                if not item:
                    continue
                if item["layer_key"] and item["layer_key"] in have:
                    continue
                norm = " ".join((item["text"] or "").split()).lower()
                if norm in seen_text:
                    continue
                seen_text.add(norm)
                remote.append(item)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    results = (local + remote)[:limit]
    return {"query": q, "layer": memory_type, "info": layers.LAYER_INFO[memory_type],
            "results": results, "error": error if not results else None,
            "warning": error if results and error else None}


@router.get("/memory/layers")
async def layer_info():
    """What each layer is FOR (drives the explorer's per-layer explanation)."""
    return {"layers": layers.LAYER_INFO, "local_counts": store.layer_counts()}


@router.get("/memory/search-all")
async def search_all_layers(q: str, limit: int = 3):
    """One keyword, all three layers — each answers a different question about
    the same incident, so results are side by side but never the same text."""
    return {t: await _search_layer(t, q, limit) for t in layers.LAYERS}


@router.get("/memory/{memory_type}/search")
async def search_memory(memory_type: str, q: str, limit: int = 5):
    if memory_type not in _RECALL:
        return {"error": f"unknown memory_type '{memory_type}'", "results": []}
    return await _search_layer(memory_type, q, limit)


@router.get("/health")
async def health():
    from ..config import validate_settings

    components: dict[str, str] = {"api": "ok"}
    # SQLite
    try:
        conn = sqlite3.connect(settings.DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
        components["sqlite"] = "ok"
    except Exception:
        components["sqlite"] = "unreachable"
    # Groq (lightweight)
    try:
        g = await __import__("asyncio").to_thread(groq_health)
        components["groq"] = "ok" if g.get("status") in ("ok", "fallback_ok") else g.get("status", "unreachable")
        if g.get("status") == "missing_key":
            components["groq"] = "missing_key"
    except Exception:
        components["groq"] = "unreachable"
    # Hindsight banks
    try:
        banks = await memory.health_check()
        components["hindsight"] = "ok" if all(v == "ok" for v in banks.values()) else "degraded"
        components.update(banks)
    except Exception:
        components["hindsight"] = "unreachable"
        components.update({"episodic": "unreachable", "semantic": "unreachable", "procedural": "unreachable"})

    warnings = validate_settings()
    ok = all(v in ("ok",) for v in components.values())
    status = "ok" if (ok and not warnings) else "degraded"
    if components.get("api") != "ok" or components.get("sqlite") != "ok":
        status = "degraded"
    return {"status": status, "components": components, "warnings": warnings,
            "config": memory.config_summary()}


@router.get("/diagnostics/groq")
async def diagnostics_groq():
    """Test simple completion (no tools). Never exposes keys."""
    from ..groq_client import achat_completion

    if not settings.GROQ_API_KEY:
        return {"ok": False, "error": "GROQ_API_KEY not configured"}
    r = await achat_completion([{"role": "user", "content": "Reply with exactly: ok"}])
    if r.error:
        return {"ok": False, "error": r.error, "primary_error": r.primary_error,
                "fallback_error": r.fallback_error}
    return {"ok": True, "model_used": r.model_used,
            "reply": (getattr(r.message, "content", "") or "")[:200]}


@router.get("/diagnostics/groq-tools")
async def diagnostics_groq_tools():
    """Test structured submit_diagnosis tool call."""
    from ..agent import TOOLS
    from ..groq_client import achat_completion

    if not settings.GROQ_API_KEY:
        return {"ok": False, "error": "GROQ_API_KEY not configured"}
    messages = [
        {"role": "system", "content": "You must call submit_diagnosis with a test diagnosis."},
        {"role": "user", "content": "Test incident: auth-service 401s after deploy."},
    ]
    r = await achat_completion(messages, tools=TOOLS, tool_choice="auto")
    if r.error:
        return {"ok": False, "error": r.error}
    tcs = getattr(r.message, "tool_calls", None) or []
    return {"ok": True, "model_used": r.model_used,
            "tool_calls": [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in tcs],
            "content": getattr(r.message, "content", "")}


@router.get("/diagnostics/hindsight")
async def diagnostics_hindsight():
    """Connect → ensure banks → retain one probe → recall → sizes. No secrets."""
    try:
        banks = await memory.ensure_banks()
        probe = "diagnostic probe memory — safe to ignore"
        await memory.get_client().aretain(bank_id=memory.EPISODIC_BANK, content=probe)
        rec = await memory.recall_episodic("diagnostic probe", limit=1)
        sizes = await memory.bank_sizes()
        return {"ok": True, "banks": banks, "probe_recalled": len(rec) > 0,
                "sizes": sizes, "config": memory.config_summary()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "config": memory.config_summary()}
