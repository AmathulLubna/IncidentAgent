"""Tests for the three improvements:

1. historical evidence is specific to each incident (topical gate + local store)
2. recommendations are category-specific problem/solution guidance
3. memory layers hold three DIFFERENT documents per resolved incident
"""
import asyncio
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app import agent as agent_mod
from app import layers, playbooks, store, taxonomy
from app import memory as memory_mod
from app.groq_client import ChatResult

from test_regression import (  # noqa: E402  (shared fakes)
    NOW, FakeGroq, install_groq, install_recalls, mem, submit_args,
)

AUTH = ("auth-service",
        "Users are receiving intermittent 401 responses after token refresh. Logs show token "
        "validation failures and refresh requests timing out.")
POOL = ("orders-api",
        'HTTP 503 errors. MongoDB connection utilization: 97%. "MongoServerSelectionError: '
        'connection pool exhausted".')
DISK = ("media-service", "No space left on device writing to /tmp/uploads during video upload.")


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store.settings, "DB_PATH", str(tmp_path / "imp.db"))
    store.init_db()
    return store


def _resolved(service, symptom, family, steps, outcome="success"):
    iid = store.create_incident(service, symptom, NOW)
    store.save_resolution(iid, family, steps, outcome)
    layers.record_resolution(
        incident_id=iid, service=service, symptom=symptom, occurred_at=NOW.isoformat(),
        root_cause_family=family, fix_steps=steps, outcome=outcome,
    )
    return iid


def _hindsight_hit(iid, service, symptom, family):
    return mem(
        f"{service} incident {iid}",
        raw=(f"[HISTORICAL INCIDENT #{iid} RESOLVED] service={service} status=resolved "
             f"outcome=success root_cause_family={family}\n{symptom}"),
        metadata={"incident_id": str(iid), "service": service, "status": "resolved",
                  "outcome": "success", "root_cause_family": family},
        score=0.9, memory_id=f"h-{iid}",
    )


# ---------------------------------------------------------------- 1. evidence

def test_each_incident_gets_its_own_historical_evidence(db, monkeypatch):
    a = _resolved(*AUTH[:1], AUTH[1], "auth_token_expiry", ["Roll back auth deploy"])
    p = _resolved(POOL[0], POOL[1], "db_connection_pool_exhaustion", ["Fix pool leak"])
    d = _resolved(DISK[0], DISK[1], "disk_full", ["Rotate logs"])
    # Fuzzy recall: Hindsight returns ALL three for ANY query (the reported bug).
    fuzzy = [_hindsight_hit(a, *AUTH, "auth_token_expiry"),
             _hindsight_hit(p, *POOL, "db_connection_pool_exhaustion"),
             _hindsight_hit(d, *DISK, "disk_full")]
    install_recalls(monkeypatch, episodic=fuzzy)

    seen = {}
    for name, (svc, sym), want in [("auth", AUTH, a), ("pool", POOL, p), ("disk", DISK, d)]:
        cur = store.create_incident(svc + "-new", sym, NOW)
        install_groq(monkeypatch, [
            {"tool_calls": [("search_episodic_memory", {"query": "incident"})]},
            {"tool_calls": [("submit_diagnosis", submit_args())]},
        ])
        res = asyncio.run(agent_mod.diagnose_incident(cur, svc + "-new", sym, NOW.isoformat()))
        ids = {h.historical_incident_id for h in res.memory_hits}
        assert ids == {want}, f"{name}: expected only #{want}, got {ids}"
        assert res.memory_trace[0].off_topic == 2
        seen[name] = ids
    assert seen["auth"] != seen["pool"] != seen["disk"]


def test_local_store_supplies_evidence_when_hindsight_returns_nothing(db, monkeypatch):
    p = _resolved(POOL[0], POOL[1], "db_connection_pool_exhaustion", ["Fix pool leak"])
    install_recalls(monkeypatch, episodic=[])
    cur = store.create_incident("checkout-api", POOL[1], NOW)
    install_groq(monkeypatch, [
        {"tool_calls": [("search_episodic_memory", {"query": "503"})]},
        {"tool_calls": [("submit_diagnosis", submit_args())]},
    ])
    res = asyncio.run(agent_mod.diagnose_incident(cur, "checkout-api", POOL[1], NOW.isoformat()))
    assert [h.historical_incident_id for h in res.memory_hits] == [p]
    assert res.memory_hits[0].source == "local"
    assert res.memory_trace[0].local_added == 1


def test_unrelated_local_history_is_not_evidence(db, monkeypatch):
    _resolved(*DISK[:1], DISK[1], "disk_full", ["Rotate logs"])
    install_recalls(monkeypatch, episodic=[])
    cur = store.create_incident("auth-service", AUTH[1], NOW)
    install_groq(monkeypatch, [
        {"tool_calls": [("search_episodic_memory", {"query": "401"})]},
        {"tool_calls": [("submit_diagnosis", submit_args(primary_memory_type_used="none"))]},
    ])
    res = asyncio.run(agent_mod.diagnose_incident(cur, "auth-service", AUTH[1], NOW.isoformat()))
    assert res.memory_hits == [] and res.memory_hit_type == "none"


def test_ungrounded_history_claims_are_removed(db):
    ctx = agent_mod.IncidentContext(5, "auth-service", AUTH[1], NOW.isoformat())
    text = ("Token refresh is failing. Same as incident #99 which was fixed by rollback. "
            "This resembles a similar past incident.")
    out = agent_mod.ground_diagnosis_text(text, [], ctx)
    assert "#99" not in out
    assert agent_mod.COLD_START_NOTE in out


# ----------------------------------------------------------- 2. recommendations

def test_recommendations_differ_per_category_and_use_the_incident(db):
    recs = {}
    for name, (svc, sym) in {"auth": AUTH, "pool": POOL, "disk": DISK}.items():
        prof = taxonomy.build_profile(svc, sym)
        recs[name] = playbooks.build_recommendations(prof, svc)[0]
    assert recs["auth"].category == "auth_token_expiry"
    assert recs["pool"].category == "db_connection_pool_exhaustion"
    assert recs["disk"].category == "disk_full"
    assert len({r.title for r in recs.values()}) == 3
    assert len({tuple(r.permanent_fix) for r in recs.values()}) == 3
    assert "media-service" in recs["disk"].problem
    assert any("deploy/media-service" in c.command for c in recs["disk"].investigate)
    assert recs["pool"].matched_signals, "signals from the incident text must be surfaced"
    for r in recs.values():
        assert r.immediate_actions and r.permanent_fix and r.verify and r.prevent and r.investigate


def test_recommendation_includes_confirmed_fix_from_matching_history(db):
    p = _resolved(POOL[0], POOL[1], "db_connection_pool_exhaustion", ["Close leaked cursors in repo layer"])
    hit = agent_mod.MemoryHit(memory_type="episodic", query="q", snippet="s", historical_incident_id=p)
    prof = taxonomy.build_profile("checkout-api", POOL[1])
    rec = playbooks.build_recommendations(prof, "checkout-api", [hit])[0]
    assert rec.proven_in_history and rec.proven_in_history[0].steps == ["Close leaked cursors in repo layer"]


def test_llm_failure_still_yields_category_specific_steps(db, monkeypatch):
    install_recalls(monkeypatch, episodic=[])
    install_groq(monkeypatch, [{"error": "boom"}])
    cur = store.create_incident(*DISK, NOW)
    res = asyncio.run(agent_mod.diagnose_incident(cur, DISK[0], DISK[1], NOW.isoformat()))
    assert res.suggested_fix_steps and res.recommendations[0].category == "disk_full"


# ------------------------------------------------------------------- 3. layers

def test_three_layers_hold_three_different_documents(db):
    docs = layers.build_layer_docs(
        incident_id=7, service="auth-service", symptom=AUTH[1], occurred_at=NOW.isoformat(),
        root_cause_family="auth_token_expiry",
        fix_steps=["Roll back auth deploy", "Flush validator key cache"], outcome="success",
    )
    by = {d.layer: d for d in docs}
    assert set(by) == {"episodic", "semantic", "procedural"}
    contents = [d.content for d in docs]
    assert len(set(contents)) == 3
    assert len({d.gist for d in docs}) == 3
    assert "Symptoms:" in by["episodic"].content and "Mechanism:" not in by["episodic"].content
    assert "Mechanism:" in by["semantic"].content and "Ordered fix steps:" not in by["semantic"].content
    assert "Ordered fix steps:" in by["procedural"].content and "Symptoms:" not in by["procedural"].content
    assert set(by["episodic"].fields) != set(by["semantic"].fields) != set(by["procedural"].fields)


def test_failed_fix_is_not_a_runbook(db):
    docs = layers.build_layer_docs(
        incident_id=8, service="orders-api", symptom=POOL[1], occurred_at="",
        root_cause_family="db_connection_pool_exhaustion", fix_steps=["Raised maxPoolSize"], outcome="fail",
    )
    assert {d.layer for d in docs} == {"episodic", "semantic"}
    assert "did NOT resolve" in docs[0].content and "ruled out" in docs[1].content


def test_layer_search_returns_layer_specific_results_per_incident(db):
    _resolved(AUTH[0], AUTH[1], "auth_token_expiry", ["Roll back auth deploy"])
    _resolved(POOL[0], POOL[1], "db_connection_pool_exhaustion", ["Fix pool leak"])
    for kw, cat in [("token refresh", "auth_token_expiry"), ("connection pool", "db_connection_pool_exhaustion")]:
        per = {t: layers.search_local(t, kw) for t in layers.LAYERS}
        assert all(per[t] for t in per), f"{kw}: every layer must answer"
        assert {r["category"] for t in per for r in per[t]} == {cat}, "keyword must not leak other incidents"
        texts = [per[t][0]["text"] for t in per]
        assert len(set(texts)) == 3, "three layers must never show the same text"
        assert "Symptoms:" in texts[0] and "Mechanism:" in texts[1] and "Ordered fix steps:" in texts[2]


def test_resolve_writes_all_three_layers(db, monkeypatch, _no_network_layer_retain):
    import app.main as main_mod

    async def _banks():
        return {}

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(memory_mod, "ensure_banks", _banks)
    monkeypatch.setattr(memory_mod, "retain_episodic_resolution", _noop)
    monkeypatch.setattr(memory_mod, "bank_sizes", _noop)
    install_recalls(monkeypatch, episodic=[])
    install_groq(monkeypatch, [{"tool_calls": [("submit_diagnosis", submit_args())]}])
    with TestClient(main_mod.app) as c:
        iid = c.post("/incidents", json={"service": POOL[0], "symptom": POOL[1]}).json()["incident_id"]
        out = c.post(f"/incidents/{iid}/resolve", json={
            "root_cause_family": "db_connection_pool_exhaustion",
            "fix_steps": ["Fix pool leak", "Monitor"], "outcome": "success"}).json()
        assert {l["layer"] for l in out["layers"]} == {"episodic", "semantic", "procedural"}
        assert {d.layer for d in _no_network_layer_retain} == {"semantic", "procedural"}
        got = c.get("/memory/search-all", params={"q": "connection pool"}).json()
        texts = [got[t]["results"][0]["text"] for t in ("episodic", "semantic", "procedural")]
        assert len(set(texts)) == 3
        rec = c.get(f"/incidents/{iid}").json()
        assert rec["recommendations"][0]["category"] == "db_connection_pool_exhaustion"
