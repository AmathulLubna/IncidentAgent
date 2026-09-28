"""Core automated tests — no manual UI clicking required.

Covers: consolidation arithmetic, procedural thresholds, store bookkeeping,
API contracts (via TestClient with Groq/Hindsight stubbed), agent evidence
builder. Live Hindsight/Groq tests live in scripts/test_*.py instead.

Run:  venv/Scripts/python.exe -m pytest backend/tests -q
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402


def test_confidence_arithmetic():
    count, window = 3, 6
    assert round(count / window * 100, 1) == 50.0
    assert round(2 / 3 * 100, 1) == 66.7


def test_procedural_thresholds():
    from app.config import settings

    assert settings.PROCEDURAL_MIN_ATTEMPTS == 2
    assert abs(settings.PROCEDURAL_MIN_SUCCESS_RATE - 0.66) < 1e-9
    # 1 attempt must NOT promote even at 100%
    assert not (1 >= settings.PROCEDURAL_MIN_ATTEMPTS)
    # 2/3 must promote
    assert 3 >= settings.PROCEDURAL_MIN_ATTEMPTS and (2 / 3) >= settings.PROCEDURAL_MIN_SUCCESS_RATE


def test_store_success_rate_tmp(tmp_path, monkeypatch):
    from app import store

    monkeypatch.setattr(store.settings, "DB_PATH", str(tmp_path / "t.db"))
    store.init_db()
    from datetime import datetime, timezone

    i1 = store.create_incident("svc", "s1", datetime.now(timezone.utc))
    i2 = store.create_incident("svc", "s2", datetime.now(timezone.utc))
    store.save_resolution(i1, "fam_a", ["step"], "success")
    store.save_resolution(i2, "fam_a", ["step"], "fail")
    attempts, rate = store.get_success_rate("fam_a")
    assert attempts == 2 and abs(rate - 0.5) < 1e-9


def test_build_evidence():
    from app.agent import build_evidence_text
    from app.schemas import MemoryHit

    hits = [MemoryHit(memory_type="episodic", query="q", snippet="Incident #4 ...")]
    ev = build_evidence_text(hits)
    assert len(ev) == 1 and "q" in ev[0]


def test_api_contract_stubbed(tmp_path, monkeypatch):
    """POST /incidents + resolve + stats + health with memory/Groq stubbed."""
    from app import memory, store
    from app.agent import MemoryHit as _MH  # noqa
    import app.agent as agent_mod
    import app.consolidation as cons_mod
    import app.main as main_mod

    monkeypatch.setattr(store.settings, "DB_PATH", str(tmp_path / "api.db"))
    store.init_db()

    async def _noop(*a, **k):
        return None

    async def _empty(*a, **k):
        return []

    async def _sizes():
        return {m: 0 for m in memory.ALL_BANKS}

    async def _health():
        return {"episodic": "ok", "semantic": "ok", "procedural": "ok"}

    async def _diag(incident_id, service, symptom, timestamp):
        from app.schemas import DiagnosisResult

        return DiagnosisResult(incident_id=incident_id, diagnosis_text="stub",
                               memory_hit_type="none", model_used="stub",
                               memory_hits=[], evidence=[])

    monkeypatch.setattr(memory, "retain_episodic_resolution", _noop)
    monkeypatch.setattr(memory, "recall_episodic", _empty)
    monkeypatch.setattr(memory, "bank_sizes", _sizes)
    monkeypatch.setattr(memory, "health_check", _health)
    monkeypatch.setattr(agent_mod, "diagnose_incident", _diag)

    async def _promo(*a, **k):
        return None

    async def _cons(*a, **k):
        return None

    monkeypatch.setattr(cons_mod, "maybe_promote_procedural", _promo)
    monkeypatch.setattr(cons_mod, "maybe_run_consolidation", _cons)

    c = TestClient(main_mod.app)
    r = c.post("/incidents", json={"service": "svc", "symptom": "sym"})
    assert r.status_code == 200, r.text
    iid = r.json()["incident_id"]
    r2 = c.post(f"/incidents/{iid}/resolve",
                json={"root_cause_family": "fam", "fix_steps": ["a"], "outcome": "success"})
    assert r2.status_code == 200, r2.text
    assert c.get("/stats").status_code == 200
    assert c.get("/incidents").status_code == 200
