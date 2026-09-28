"""Regression tests for the two production bugs (plus provenance/metric guards).

Bug A — suggested_fix_steps must survive Groq tool call -> DiagnosisResult ->
        SQLite -> FastAPI -> (frontend reads it back).
Bug B — the CURRENT incident must never be represented as historical memory,
        and a self-hit must never inflate the memory hit rate.

These are the §22 test plan scenarios:

  Test 1  test_current_incident_is_never_historical_memory
  Test 2  test_previous_incident_is_retrieved_as_historical_evidence
  Test 3  test_suggested_fix_steps_survive_end_to_end
  Test 4  test_procedural_memory_produces_steps_when_omitted
  Test 5  test_failed_resolution_is_not_promoted
  Test 6  test_predicted_and_confirmed_root_cause_are_distinct
  extras  test_unverifiable_incident_id_is_not_evidence (§9 provenance)
          test_markerless_episodic_memory_is_not_evidence (§9 provenance)
          test_recall_failure_is_reported_as_failure (§23)
          test_incident_is_not_retained_before_diagnosis (§6/§21)
          test_weak_match_is_not_presented_as_evidence (relevance floor)
          test_duplicate_runbook_content_is_one_readable_bullet (evidence dedupe)
          test_runbook_is_not_retained_twice_for_the_same_evidence (no dup runbooks)
          test_evidence_is_ranked_and_capped (evidence is readable + short)

No network: Groq is scripted and Hindsight recall/retain are stubbed.

Run:  venv/Scripts/python.exe -m pytest backend/tests -q
"""
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app import agent as agent_mod  # noqa: E402
from app import memory as memory_mod  # noqa: E402
from app import store  # noqa: E402
from app.groq_client import ChatResult  # noqa: E402

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)

ORDER_SYMPTOM = (
    "The Order API is intermittently returning HTTP 503 errors. "
    "MongoDB connection utilization: 97%. Logs: "
    '"MongoServerSelectionError: connection pool exhausted".'
)
PAYMENT_SYMPTOM = "Payment API HTTP 503 errors, high latency, MongoDB connection pool exhausted."


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store.settings, "DB_PATH", str(tmp_path / "regression.db"))
    store.init_db()
    return store


def mem(text, *, raw=None, metadata=None, score=0.91, memory_id="m-1", occurred=None):
    """One fake Hindsight recall result (same shape memory._recall produces)."""
    return memory_mod.MemoryResult(
        text=text, score=score, memory_id=memory_id, raw_text=raw,
        metadata=metadata, tags=None, occurred_start=occurred,
    )


def install_recalls(monkeypatch, *, episodic=None, semantic=None, procedural=None, errors=None):
    """Replace the three recall functions. Returns the call log [(tier, query)]."""
    errors = errors or {}
    log = []

    def _mk(tier, results):
        async def fn(query, limit=5):
            log.append((tier, query))
            if tier in errors:
                return memory_mod.RecallBatch(error=errors[tier])
            return memory_mod.RecallBatch(results=list(results or [])[:limit])
        return fn

    monkeypatch.setattr(memory_mod, "recall_episodic", _mk("episodic", episodic))
    monkeypatch.setattr(memory_mod, "recall_semantic", _mk("semantic", semantic))
    monkeypatch.setattr(memory_mod, "recall_procedural", _mk("procedural", procedural))
    return log


class _Fn:
    def __init__(self, name, args):
        self.name = name
        self.arguments = json.dumps(args)


class _TC:
    def __init__(self, name, args, tc_id):
        self.id = tc_id
        self.function = _Fn(name, args)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeGroq:
    """Scripted stand-in for agent.achat_completion. Records every message list
    it was shown so tests can assert on the exact tool output the model saw."""

    def __init__(self, script):
        self.script = [dict(s) for s in script]
        self.calls = []

    async def __call__(self, messages, tools=None, tool_choice=None):
        self.calls.append(json.loads(json.dumps(messages, default=str)))
        if not self.script:
            return ChatResult(message=_Msg(content="script exhausted"), model_used="fake-model")
        turn = self.script.pop(0)
        if "error" in turn:
            return ChatResult(error=turn["error"])
        tcs = [_TC(name, args, f"tc{i}") for i, (name, args) in enumerate(turn.get("tool_calls", []))]
        return ChatResult(
            message=_Msg(content=turn.get("content"), tool_calls=tcs or None),
            model_used="fake-model",
        )

    @property
    def last_context(self) -> str:
        return json.dumps(self.calls[-1], default=str)


def install_groq(monkeypatch, script) -> FakeGroq:
    fake = FakeGroq(script)
    monkeypatch.setattr(agent_mod, "achat_completion", fake)
    return fake


def diagnose(incident_id, service, symptom):
    return asyncio.run(agent_mod.diagnose_incident(
        incident_id, service, symptom, NOW.isoformat()
    ))


def submit_args(**overrides):
    args = {
        "diagnosis_text": (
            "Database connection pool exhaustion: MongoDB connections are not being "
            "released (97% pool utilization, MongoServerSelectionError). Confidence 91%."
        ),
        "root_cause_family_guess": "db_connection_pool_exhaustion",
        "suggested_fix_steps": [
            "Inspect current MongoDB connection utilization",
            "Verify connections are reused and released correctly",
            "Inspect maxPoolSize configuration",
            "Increase maxPoolSize only after verifying database capacity",
            "Monitor connection utilization and HTTP 503 rate",
        ],
        "primary_memory_type_used": "episodic",
        "confidence_pct": 91,
    }
    args.update(overrides)
    return args


@pytest.fixture()
def api(db, monkeypatch):
    """TestClient with Hindsight startup calls stubbed (no network)."""
    import app.main as main_mod

    async def _banks():
        return {"stub": "ok"}

    monkeypatch.setattr(memory_mod, "ensure_banks", _banks)
    with TestClient(main_mod.app) as client:
        yield client


# ---------------------------------------------------------------------------
# Test 1 — current incident cannot become historical memory
# ---------------------------------------------------------------------------

def test_current_incident_is_never_historical_memory(db, monkeypatch):
    incident_id = store.create_incident("Order API", ORDER_SYMPTOM, NOW)

    # The bank is "poisoned" with the current incident itself (the exact failure
    # mode of retaining an incident before diagnosing it): once via metadata,
    # once via raw content markers, plus an unrelated historical Payment API hit.
    self_hit_meta = mem(
        "The Order API incident is ongoing.",
        raw=f"[HISTORICAL INCIDENT #{incident_id} RESOLVED] service=Order API status=resolved outcome=success",
        metadata={"incident_id": str(incident_id), "service": "Order API", "status": "resolved"},
        memory_id="self-1",
    )
    self_hit_text = mem(
        f"Order API is reporting the same error (incident #{incident_id}).",
        raw=f"[INCIDENT #{incident_id} REPORTED] service=Order API",
        memory_id="self-2",
        score=0.99,
    )
    install_recalls(monkeypatch, episodic=[self_hit_meta, self_hit_text])

    fake = install_groq(monkeypatch, [
        {"tool_calls": [("search_episodic_memory", {"query": "Order API 503 connection pool"})]},
        # The model *claims* episodic — the claim must not survive validation.
        {"tool_calls": [("submit_diagnosis", submit_args(primary_memory_type_used="episodic"))]},
    ])

    result = diagnose(incident_id, "Order API", ORDER_SYMPTOM)

    assert result.memory_hits == [], "current incident leaked into memory_hits"
    assert result.evidence == []
    assert result.memory_hit_type == "none", "self-hit must not count as a memory hit"
    assert result.suggested_fix_steps, "diagnosis must still carry fix steps"

    # The model must never even SEE the current incident as historical evidence.
    assert "HISTORICAL INCIDENT #" not in fake.last_context
    assert "[INCIDENT #" not in fake.last_context

    # Trace shows what really happened: 2 returned, 0 accepted, 2 rejected.
    assert len(result.memory_trace) == 1
    step = result.memory_trace[0]
    assert (step.returned, step.accepted, step.rejected, step.status) == (2, 0, 2, "ok")

    # And the hit-rate metric cannot be inflated by a self-hit.
    store.save_diagnosis(incident_id, result)
    assert store.get_stats()["memory_hit_rate_last_5"] == 0.0


# ---------------------------------------------------------------------------
# Test 1b — memory claiming a non-existent incident is not evidence
# ---------------------------------------------------------------------------

def test_unverifiable_incident_id_is_not_evidence(db, monkeypatch):
    first = store.create_incident("Payment API", PAYMENT_SYMPTOM, NOW)
    store.save_resolution(
        first, "db_connection_pool_exhaustion",
        ["Correct connection reuse", "Increase pool capacity"], "success",
    )
    second = store.create_incident("Order API", ORDER_SYMPTOM, NOW)

    # A stray memory whose provenance points at an incident that does not exist
    # in incident history (SDK probe / stale artifact) — it must never be
    # presented as historical evidence, while a genuinely resolved incident is.
    probe_hit = mem(
        "The probe-service incident #999999 was resolved after a roundtrip test.",
        raw="[HISTORICAL INCIDENT #999999 RESOLVED] service=probe-service outcome=success",
        metadata={"incident_id": "999999", "service": "probe-service", "outcome": "success"},
        memory_id="probe-1",
    )
    good_hit = mem(
        f"Payment API incident #{first} resolved after fixing pool exhaustion.",
        raw=f"[HISTORICAL INCIDENT #{first} RESOLVED] service=Payment API status=resolved outcome=success",
        metadata={"incident_id": str(first), "service": "Payment API", "status": "resolved"},
        memory_id="hist-1",
    )
    install_recalls(monkeypatch, episodic=[probe_hit, good_hit])
    install_groq(monkeypatch, [
        {"tool_calls": [("search_episodic_memory", {"query": "Order API 503 connection pool"})]},
        {"tool_calls": [("submit_diagnosis", submit_args())]},
    ])

    result = diagnose(second, "Order API", ORDER_SYMPTOM)

    ids = [h.historical_incident_id for h in result.memory_hits]
    assert 999999 not in ids, "memory claiming a non-existent incident must be rejected"
    assert first in ids, "genuinely resolved incident must still be accepted"
    assert result.memory_hit_type == "episodic"
    assert result.suggested_fix_steps
    assert len(result.memory_trace) == 1
    step = result.memory_trace[0]
    assert (step.returned, step.accepted, step.rejected, step.status) == (2, 1, 1, "ok")


# ---------------------------------------------------------------------------
# Test 1c — markerless episodic memory is not evidence (semantic still is)
# ---------------------------------------------------------------------------

def test_markerless_episodic_memory_is_not_evidence(db, monkeypatch):
    first = store.create_incident("Payment API", PAYMENT_SYMPTOM, NOW)
    store.save_resolution(
        first, "db_connection_pool_exhaustion",
        ["Correct connection reuse", "Increase pool capacity"], "success",
    )
    second = store.create_incident("Order API", ORDER_SYMPTOM, NOW)

    # A stale report-style episodic memory with NO provenance whatsoever
    # (markerless content retained before the fix) — present tense, unresolved,
    # unverifiable: it must never reach the model as historical evidence.
    stale = mem(
        "The order-service is experiencing intermittent HTTP 503 errors and "
        "significant latency increases during peak traffic; restarting provides "
        "temporary relief.",
        raw=None,
        memory_id="stale-1",
    )
    good = mem(
        f"Payment API incident #{first} resolved after fixing pool exhaustion.",
        raw=f"[INCIDENT #{first} RESOLVED] service=Payment API status=resolved outcome=success",
        metadata={"incident_id": str(first), "service": "Payment API", "status": "resolved"},
        memory_id="hist-1",
    )
    # Semantic tier: cross-incident patterns legitimately carry no incident id.
    pattern = mem(
        "Pattern: pool exhaustion during peak traffic correlates with connection leaks.",
        memory_id="sem-1",
    )
    install_recalls(monkeypatch, episodic=[stale, good], semantic=[pattern])
    fake = install_groq(monkeypatch, [
        {"tool_calls": [("search_episodic_memory", {"query": "Order API 503 connection pool"})]},
        {"tool_calls": [("search_semantic_memory", {"query": "pool exhaustion peak traffic"})]},
        {"tool_calls": [("submit_diagnosis", submit_args())]},
    ])

    result = diagnose(second, "Order API", ORDER_SYMPTOM)

    ids = [h.historical_incident_id for h in result.memory_hits]
    assert first in ids, "resolved incident with provenance must be accepted"
    assert all(h.memory_id != "stale-1" for h in result.memory_hits), (
        "markerless episodic memory leaked into memory_hits"
    )
    assert any(h.memory_id == "sem-1" and h.memory_type == "semantic"
               for h in result.memory_hits), "semantic patterns must stay usable"
    epi = next(t for t in result.memory_trace if t.tier == "episodic")
    assert (epi.returned, epi.accepted, epi.rejected) == (2, 1, 1)
    # The model must never be shown the stale present-tense memory.
    assert "temporary relief" not in fake.last_context


# ---------------------------------------------------------------------------
# Test 2 — previous incident is retrieved, current one is not
# ---------------------------------------------------------------------------

def test_previous_incident_is_retrieved_as_historical_evidence(db, monkeypatch):
    first = store.create_incident("Payment API", PAYMENT_SYMPTOM, NOW)
    store.save_resolution(
        first, "db_connection_pool_exhaustion",
        ["Correct connection reuse", "Increase pool capacity"], "success",
    )
    second = store.create_incident("Order API", ORDER_SYMPTOM, NOW)

    historical = mem(
        "Payment API had HTTP 503s from connection pool exhaustion; resolved successfully.",
        raw=("[HISTORICAL INCIDENT #%d RESOLVED] service=Payment API occurred_at=%s "
             "status=resolved outcome=success\nConfirmed root cause family: "
             "db_connection_pool_exhaustion") % (first, NOW.isoformat()),
        metadata={
            "incident_id": str(first), "service": "Payment API", "status": "resolved",
            "outcome": "success", "root_cause_family": "db_connection_pool_exhaustion",
        },
        memory_id="hist-1",
    )
    # Same historical incident seen a second time as a derived observation with
    # no metadata (raw marker only) -> must dedupe, not double count.
    historical_dup = mem(
        f"The Payment API incident #{first} was resolved successfully.",
        raw=f"[HISTORICAL INCIDENT #{first} RESOLVED] service=Payment API status=resolved outcome=success",
        memory_id="hist-2",
        score=0.87,
    )
    self_hit = mem(
        f"Order API incident #{second} is the one being diagnosed.",
        raw=f"[HISTORICAL INCIDENT #{second} RESOLVED] service=Order API status=resolved outcome=success",
        metadata={"incident_id": str(second), "service": "Order API", "status": "resolved"},
        memory_id="self-3",
        score=0.95,
    )
    install_recalls(monkeypatch, episodic=[historical, historical_dup, self_hit])

    install_groq(monkeypatch, [
        {"tool_calls": [("search_episodic_memory", {"query": "Order API 503 pool exhaustion"})]},
        {"tool_calls": [("submit_diagnosis", submit_args())]},
    ])

    result = diagnose(second, "Order API", ORDER_SYMPTOM)

    ids = [h.historical_incident_id for h in result.memory_hits]
    assert first in ids, "the previous incident must appear as historical evidence"
    assert second not in ids, "the current incident must never appear as historical evidence"
    assert len(result.memory_hits) == 1, "same historical incident must dedupe to one hit"

    hit = next(h for h in result.memory_hits if h.historical_incident_id == first)
    assert hit.service == "Payment API"
    assert hit.outcome == "success"
    assert hit.root_cause_family == "db_connection_pool_exhaustion"
    assert hit.memory_id == "hist-1"
    assert hit.relevance_score == pytest.approx(0.91)

    assert result.memory_hit_type == "episodic"
    assert any("historical incident #1" in line for line in result.evidence), result.evidence
    origin = result.evidence[0].split("—", 1)[1]
    assert "historical incident #1 (Payment API), outcome=success" in origin, origin
    assert not any(h.historical_incident_id == second for h in result.memory_hits)


# ---------------------------------------------------------------------------
# Test 3 — suggested steps are present, end to end
# ---------------------------------------------------------------------------

def test_suggested_fix_steps_survive_end_to_end(api, monkeypatch):
    # One genuinely historical memory so memory_hit_type has something to reflect.
    install_recalls(monkeypatch, episodic=[
        mem("Payment API was resolved after fixing the connection lifecycle.",
            raw=("[HISTORICAL INCIDENT #999 RESOLVED] service=Payment API status=resolved "
                 "outcome=success root_cause_family=db_connection_pool_exhaustion"),
            metadata={"incident_id": "999", "service": "Payment API", "status": "resolved",
                      "outcome": "success", "root_cause_family": "db_connection_pool_exhaustion"},
            memory_id="hist-999"),
    ])
    install_groq(monkeypatch, [
        {"tool_calls": [("search_episodic_memory", {"query": "503 connection pool"})]},
        {"tool_calls": [("submit_diagnosis", submit_args())]},
    ])

    created = api.post("/incidents", json={
        "service": "Order API",
        "symptom": ORDER_SYMPTOM,
        "timestamp": NOW.isoformat(),
    })
    assert created.status_code == 200, created.text
    body = created.json()
    assert len(body["suggested_fix_steps"]) > 0, "steps missing from POST /incidents"
    expected = list(body["suggested_fix_steps"])

    # ...and they must survive persistence + a fresh read (the old bug dropped
    # them right here).
    stored = api.get(f"/incidents/{body['incident_id']}").json()
    assert stored["suggested_fix_steps"] == expected, "steps lost between POST and GET"

    listing = api.get("/incidents").json()
    assert listing[0]["suggested_fix_steps"] == expected

    # Structured diagnosis contract (§28)
    assert body["diagnosis_text"]
    assert body["confidence_pct"] == 91
    assert body["root_cause_family_guess"] == "db_connection_pool_exhaustion"
    assert body["memory_hit_type"] == "episodic"
    assert body["model_used"] == "fake-model"
    assert isinstance(body["memory_hits"], list)

    # An empty submission is rejected once, then the model is corrected.
    fake = install_groq(monkeypatch, [
        {"tool_calls": [("submit_diagnosis", submit_args(suggested_fix_steps=[]))]},
        {"tool_calls": [("submit_diagnosis", submit_args())]},
    ])
    corrected = api.post("/incidents", json={"service": "Order API", "symptom": ORDER_SYMPTOM})
    assert corrected.status_code == 200
    assert len(corrected.json()["suggested_fix_steps"]) > 0
    assert "rejected" in json.dumps(fake.calls[-1])
    assert "suggested_fix_steps" in json.dumps(fake.calls[-1])


# ---------------------------------------------------------------------------
# Test 4 — procedural memory produces steps even if the LLM omits them
# ---------------------------------------------------------------------------

def test_procedural_memory_produces_steps_when_omitted(db, monkeypatch):
    incident_id = store.create_incident("Order API", ORDER_SYMPTOM, NOW)
    runbook_raw = (
        "[RUNBOOK] root_cause_family=db_connection_pool_exhaustion success_rate=100% "
        "(based on 2 confirmed attempts)\nOrdered fix steps:\n"
        "  1. Inspect active connections\n"
        "  2. Check connection lifecycle\n"
        "  3. Inspect maxPoolSize\n"
        "  4. Verify database capacity\n"
        "  5. Restart affected instances if necessary\n"
        "  6. Monitor recovery"
    )
    install_recalls(monkeypatch, procedural=[
        mem("The prescribed fix steps for db_connection_pool_exhaustion ...",
            raw=runbook_raw, memory_id="runbook-1"),
    ])

    empty_submit = submit_args(suggested_fix_steps=[])
    install_groq(monkeypatch, [
        {"tool_calls": [("search_procedural_memory", {"query": "connection pool runbook"})]},
        {"tool_calls": [("submit_diagnosis", empty_submit)]},   # rejected (1/2)
        {"tool_calls": [("submit_diagnosis", empty_submit)]},   # rejected (2/2)
        {"tool_calls": [("submit_diagnosis", empty_submit)]},   # accepted with runbook fallback
    ])

    result = diagnose(incident_id, "Order API", ORDER_SYMPTOM)

    assert len(result.suggested_fix_steps) >= 5, result.suggested_fix_steps
    assert result.suggested_fix_steps[0] == "Inspect active connections"
    assert result.suggested_fix_steps[-1] == "Monitor recovery"
    assert all(not s.strip()[0].isdigit() for s in result.suggested_fix_steps), "numbering must be clean"
    assert result.memory_hit_type == "procedural"


# ---------------------------------------------------------------------------
# Test 5 — failed resolution is retained as evidence but never promoted
# ---------------------------------------------------------------------------

def test_failed_resolution_is_not_promoted(api, db, monkeypatch):
    # Family already has 2 successes; adding a FAIL lands the aggregate rate at
    # 2/3 = 0.667, which clears PROCEDURAL_MIN_SUCCESS_RATE=0.66 — so ONLY the
    # outcome guard can stop the failed fix from being promoted.
    for i in range(2):
        rid = store.create_incident("Payment API", PAYMENT_SYMPTOM, NOW)
        store.save_resolution(rid, "db_connection_pool_exhaustion", ["step a", "step b"], "success")

    retained_runbooks = []
    retained_episodic = []

    async def _record_runbook(*args, **kwargs):
        retained_runbooks.append((args, kwargs))

    async def _record_episodic(*args, **kwargs):
        retained_episodic.append((args, kwargs))

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(memory_mod, "retain_procedural_runbook", _record_runbook)
    monkeypatch.setattr(memory_mod, "retain_episodic_resolution", _record_episodic)
    monkeypatch.setattr(memory_mod, "bank_sizes", _noop)
    monkeypatch.setattr(memory_mod, "health_check", _noop)

    install_recalls(monkeypatch, episodic=[])
    install_groq(monkeypatch, [
        {"tool_calls": [("submit_diagnosis", submit_args())]},
    ])

    created = api.post("/incidents", json={"service": "Payment API", "symptom": PAYMENT_SYMPTOM})
    iid = created.json()["incident_id"]

    resolved = api.post(f"/incidents/{iid}/resolve", json={
        "root_cause_family": "db_connection_pool_exhaustion",
        "fix_steps": ["Increased maxPoolSize only"],
        "outcome": "fail",
    })
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["promotion"] is None, "a FAILED fix must never be promoted"
    assert retained_runbooks == [], "procedural runbook retained from a FAIL outcome"

    # The failure IS kept as episodic (negative) evidence, with outcome=fail.
    assert len(retained_episodic) == 1
    assert retained_episodic[0][0][4] == "fail"
    assert retained_episodic[0][1].get("predicted_root_cause") is not None

    # Control: the same family at the same statistics DOES promote on success.
    from app import consolidation as consolidation_mod

    ok = asyncio.run(consolidation_mod.maybe_promote_procedural(
        "db_connection_pool_exhaustion", ["step a", "step b"], "success"
    ))
    assert ok is not None and ok.get("retained") is True
    assert len(retained_runbooks) == 1


# ---------------------------------------------------------------------------
# Test 6 — confirmed root cause is separate from prediction
# ---------------------------------------------------------------------------

def test_predicted_and_confirmed_root_cause_are_distinct(api, monkeypatch):
    install_recalls(monkeypatch, episodic=[])
    install_groq(monkeypatch, [
        {"tool_calls": [("submit_diagnosis", submit_args(
            root_cause_family_guess="predicted_connection_leak",
            suggested_fix_steps=["Inspect connection lifecycle", "Verify maxPoolSize"],
        ))]},
    ])

    created = api.post("/incidents", json={"service": "Order API", "symptom": ORDER_SYMPTOM})
    iid = created.json()["incident_id"]

    after_diagnosis = api.get(f"/incidents/{iid}").json()
    assert after_diagnosis["predicted_root_cause_family"] == "predicted_connection_leak"
    assert after_diagnosis["root_cause_family"] is None, "an AI prediction must not be 'confirmed'"
    assert after_diagnosis["fix_steps"] == [], "no confirmed fix before resolution"
    assert len(after_diagnosis["suggested_fix_steps"]) == 2, "recommended fix is its own field"

    api.post(f"/incidents/{iid}/resolve", json={
        "root_cause_family": "confirmed_mongo_pool_exhaustion",
        "fix_steps": ["Corrected connection reuse", "Raised maxPoolSize to 100"],
        "outcome": "success",
    })

    after_resolution = api.get(f"/incidents/{iid}").json()
    assert after_resolution["root_cause_family"] == "confirmed_mongo_pool_exhaustion"
    assert after_resolution["predicted_root_cause_family"] == "predicted_connection_leak"
    assert after_resolution["root_cause_family"] != after_resolution["predicted_root_cause_family"]
    # Recommended vs confirmed fix stay separate too.
    assert after_resolution["suggested_fix_steps"] == ["Inspect connection lifecycle", "Verify maxPoolSize"]
    assert after_resolution["fix_steps"] == ["Corrected connection reuse", "Raised maxPoolSize to 100"]
    assert after_resolution["outcome"] == "success"


# ---------------------------------------------------------------------------
# Extra — recall failure must be reported as a failure (§23)
# ---------------------------------------------------------------------------

def test_recall_failure_is_reported_as_failure(db, monkeypatch):
    incident_id = store.create_incident("Order API", ORDER_SYMPTOM, NOW)
    install_recalls(monkeypatch, errors={"episodic": "AuthenticationError: invalid api key"})
    fake = install_groq(monkeypatch, [
        {"tool_calls": [("search_episodic_memory", {"query": "503"})]},
        {"tool_calls": [("submit_diagnosis", submit_args(primary_memory_type_used="none"))]},
    ])

    result = diagnose(incident_id, "Order API", ORDER_SYMPTOM)

    assert result.memory_status == "degraded"
    assert any("episodic memory search failed" in e for e in result.memory_errors), result.memory_errors
    assert any("AuthenticationError" in e for e in result.memory_errors)
    assert result.memory_hits == []
    assert result.memory_hit_type == "none"
    assert result.memory_trace[0].status == "failed"
    assert "AuthenticationError" in (result.memory_trace[0].error or "")
    # The model was told the search FAILED — not that there were 0 memories.
    model_saw = json.dumps(fake.calls[-1])
    assert "memory search failed" in model_saw
    assert "AuthenticationError" in model_saw
    # A memory failure must not lose the fix steps.
    assert result.suggested_fix_steps


# ---------------------------------------------------------------------------
# Extra — retention happens after resolution, never before (§6/§21)
# ---------------------------------------------------------------------------

def test_incident_is_not_retained_before_diagnosis(api, monkeypatch):
    retained = []

    def _record(name):
        async def fn(*args, **kwargs):
            retained.append((name, args, kwargs))
        return fn

    for fn_name in ("retain_episodic_resolution", "retain_semantic_pattern",
                    "retain_procedural_runbook"):
        monkeypatch.setattr(memory_mod, fn_name, _record(fn_name))

    install_recalls(monkeypatch, episodic=[])
    install_groq(monkeypatch, [
        {"tool_calls": [("search_episodic_memory", {"query": "503"})]},
        {"tool_calls": [("submit_diagnosis", submit_args())]},
    ])

    created = api.post("/incidents", json={"service": "Order API", "symptom": ORDER_SYMPTOM})
    assert created.status_code == 200
    assert retained == [], "current incident was retained into Hindsight before diagnosis"

    # After the engineer confirms the fix, retention happens.
    iid = created.json()["incident_id"]
    api.post(f"/incidents/{iid}/resolve", json={
        "root_cause_family": "db_connection_pool_exhaustion",
        "fix_steps": ["Correct connection reuse", "Monitor recovery"],
        "outcome": "success",
    })
    assert [name for name, _, _ in retained] == ["retain_episodic_resolution"]


# ---------------------------------------------------------------------------
# Extra — historically valid but topically irrelevant memories are not evidence
# ---------------------------------------------------------------------------

def test_weak_match_is_not_presented_as_evidence(db, monkeypatch):
    """The incident-#24 bug: an auth-service memory and a disk-full memory are
    BOTH resolved historical incidents, so provenance alone accepts both. Only
    the relevance floor can tell an unrelated memory away."""
    incident_id = store.create_incident(
        "media-service",
        "No space left on device writing to /tmp/uploads during video upload.",
        NOW,
    )

    strong = mem(
        "media-service ran out of disk space in /tmp/uploads while uploading video.",
        raw=("[HISTORICAL INCIDENT #901 RESOLVED] service=media-service status=resolved "
             "outcome=success root_cause_family=disk_full\n"
             "Reported symptoms (the evidence this incident was diagnosed from):\n"
             "No space left on device on /tmp/uploads"),
        metadata={"incident_id": "901", "service": "media-service", "status": "resolved",
                  "outcome": "success", "root_cause_family": "disk_full"},
        score=1.05, memory_id="disk-1",
    )
    junk = mem(
        "auth-service saw intermittent 401 responses after a deployment.",
        raw=("[HISTORICAL INCIDENT #5 RESOLVED] service=auth-service status=resolved "
             "outcome=success root_cause_family=auth_token_expiry"),
        metadata={"incident_id": "5", "service": "auth-service", "status": "resolved",
                  "outcome": "success", "root_cause_family": "auth_token_expiry"},
        score=0.11, memory_id="auth-1",
    )
    install_recalls(monkeypatch, episodic=[junk, strong])
    fake = install_groq(monkeypatch, [
        {"tool_calls": [("search_episodic_memory",
                         {"query": "No space left on device /tmp/uploads upload video"})]},
        {"tool_calls": [("submit_diagnosis", submit_args())]},
    ])

    result = diagnose(incident_id, "media-service", "No space left on device on /tmp/uploads")

    ids = [h.historical_incident_id for h in result.memory_hits]
    assert 901 in ids, "a genuinely matching historical incident must survive"
    assert 5 not in ids, "a weak cross-topic match must not be presented as evidence"

    # Trace tells the truth about WHY it is gone (not silently swallowed).
    step = result.memory_trace[0]
    assert (step.returned, step.accepted, step.rejected, step.low_relevance) == (2, 1, 0, 1)

    # The model must not be anchored on the unrelated memory either.
    assert "intermittent 401" not in fake.last_context

    # The evidence bullet is readable: no retention markers, no raw metadata.
    assert not any("historical incident #5" in line for line in result.evidence)
    line = next(l for l in result.evidence if "historical incident #901" in l)
    assert "[HISTORICAL" not in line and "status=resolved" not in line
    assert "No space left on device on /tmp/uploads" in line
    assert "relevance" in line and "different service" not in line


# ---------------------------------------------------------------------------
# Extra — one runbook retained twice is one piece of evidence, in plain words
# ---------------------------------------------------------------------------

def test_duplicate_runbook_content_is_one_readable_bullet(db, monkeypatch):
    incident_id = store.create_incident(
        "media-service", "No space left on device on /tmp/uploads", NOW,
    )
    runbook = (
        "[RUNBOOK] root_cause_family=disk_full success_rate=100% (based on 2 confirmed attempts)\n"
        "Ordered fix steps:\n  1. Check disk usage of /tmp\n  2. Clear old upload artifacts"
    )
    # Same runbook retained under two memory ids — Hindsight has no update op.
    install_recalls(monkeypatch, procedural=[
        mem("The fix for disk_full is to clear the uploads directory.",
            raw=runbook, memory_id="rb-1", score=0.95),
        mem("The fix for disk_full is to clear the uploads directory.",
            raw=runbook, memory_id="rb-2", score=0.95),
    ])
    install_groq(monkeypatch, [
        {"tool_calls": [("search_procedural_memory", {"query": "disk full remediation runbook"})]},
        {"tool_calls": [("submit_diagnosis", submit_args(
            primary_memory_type_used="procedural",
            suggested_fix_steps=["Check disk usage of /tmp", "Clear old upload artifacts"],
        ))]},
    ])

    result = diagnose(incident_id, "media-service", "No space left on device on /tmp/uploads")

    assert len(result.memory_hits) == 1, "identical content under two memory ids is one hit"

    runbook_lines = [l for l in result.evidence if l.startswith("Confirmed runbook")]
    assert len(runbook_lines) == 1, result.evidence
    line = runbook_lines[0]
    # Plain-language summary of the runbook's own recorded strength...
    assert "root cause: disk full" in line
    assert "100% success over 2 confirmed attempts" in line
    # ...instead of the raw marker dump.
    assert "[RUNBOOK]" not in line and "Ordered fix steps" not in line
    assert "— :" not in line and not line.endswith(":"), "no dangling origin"


# ---------------------------------------------------------------------------
# Extra — a runbook is retained once per evidence level, not appended twice
# ---------------------------------------------------------------------------

def test_runbook_is_not_retained_twice_for_the_same_evidence(db, monkeypatch):
    from app import consolidation as consolidation_mod

    retained = []

    async def _record(*args, **kwargs):
        retained.append(args)

    monkeypatch.setattr(memory_mod, "retain_procedural_runbook", _record)

    for _ in range(2):
        rid = store.create_incident("media-service", "No space left on device", NOW)
        store.save_resolution(rid, "disk_full", ["step a", "step b"], "success")

    first = asyncio.run(consolidation_mod.maybe_promote_procedural("disk_full", ["step a"], "success"))
    again = asyncio.run(consolidation_mod.maybe_promote_procedural("disk_full", ["step a"], "success"))

    assert first is not None and first.get("retained") is True
    assert again is None, "the same evidence level must not append a second identical runbook"
    assert len(retained) == 1

    # A NEW confirmed resolution grows the attempts count -> worth retaining.
    rid = store.create_incident("media-service", "No space left on device", NOW)
    store.save_resolution(rid, "disk_full", ["step a", "step b"], "success")
    third = asyncio.run(consolidation_mod.maybe_promote_procedural("disk_full", ["step a"], "success"))
    assert third is not None and third.get("retained") is True
    assert len(retained) == 2


# ---------------------------------------------------------------------------
# Extra — evidence list stays short and leads with the strongest match
# ---------------------------------------------------------------------------

def test_evidence_is_ranked_and_capped(monkeypatch):
    from app.agent import build_evidence_text
    from app.schemas import MemoryHit

    def hit(i, score, service):
        return MemoryHit(
            memory_type="episodic", query="disk full /tmp uploads",
            snippet=f"[HISTORICAL INCIDENT #{i} RESOLVED] service={service} status=resolved",
            historical_incident_id=i, service=service, outcome="success",
            relevance_score=score, memory_id=f"m-{i}",
        )

    hits = [hit(10, 0.40, "auth-service"), hit(11, 0.98, "media-service"),
            hit(12, 0.72, "media-service"), hit(13, None, "media-service")]

    lines = build_evidence_text(hits, current_service="media-service", limit=3)
    assert len(lines) == 3, "evidence must be capped"
    assert "historical incident #11" in lines[0], "strongest same-service match first"
    assert "historical incident #12" in lines[1]
    # Cross-service hits are demoted, unknown-score hits rank last.
    assert "historical incident #10" not in lines, "cap must cut the weakest/other-service line"
    assert any("different service" in l for l in build_evidence_text(
        [hit(10, 0.9, "auth-service")], current_service="media-service"))
