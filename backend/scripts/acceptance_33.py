"""§33 live acceptance test — runs against a real uvicorn server with live
Groq + Hindsight (no stubs).

Scenario:
  1. Create + diagnose  "Payment API / HTTP 503 / MongoDB connection pool exhausted"
  2. Resolve it with a human-confirmed SUCCESS fix
  3. Create            "Order API / HTTP 503 / pool exhausted / 97% utilization /
                        high latency"
  4. Verify the whole bug-A / bug-B contract on the live response:

     * suggested_fix_steps is non-empty end-to-end (Groq -> SQLite -> FastAPI)
     * diagnosis = connection pool exhaustion
     * Payment API appears as historical evidence (memory_hits)
     * the Order API is NEVER claimed as a previous incident
     * the current incident never appears in its own memory_hits
     * unresolved entries (incident #14 style) are excluded from memory_hits
     * recall failure is reported as failure (memory_status / memory_errors)

Evidence (every request/response + assertions) is written to
docs/acceptance_33_evidence.json.

Run:  start uvicorn first, then
      venv/Scripts/python.exe scripts/acceptance_33.py [base_url]
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Groq occasionally returns unicode punctuation (e.g. ‑) that cp1252 consoles
# cannot encode — never let evidence printing crash the acceptance run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8010"
EVIDENCE_PATH = Path(__file__).resolve().parents[2] / "docs" / "acceptance_33_evidence.json"

PAYMENT_SYMPTOM = (
    "Payment API is returning HTTP 503 errors with high latency. "
    "Logs show 'MongoServerSelectionError: connection pool exhausted' and "
    "MongoDB connection utilization at maximum."
)
ORDER_SYMPTOM = (
    "Order API is intermittently returning HTTP 503 errors with high latency. "
    "MongoDB connection utilization: 97%. Logs show 'connection pool exhausted' "
    "during peak traffic."
)

CLAIM_RES = [
    re.compile(r"(?i)\b(previous|prior|earlier|past|historical)\b[^.!?]{0,120}\border\s+(?:api|service)\b"),
    re.compile(r"(?i)\border\s+(?:api|service)\b[^.!?]{0,120}\b(previous|prior|earlier|past|historical|before)\b"),
]

evidence: dict = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "base_url": BASE,
    "steps": [],
    "assertions": [],
}


def record(name: str, request: dict, response) -> dict:
    entry = {
        "name": name,
        "request": request,
        "status_code": response.status_code,
        "response": (response.json() if response.headers.get("content-type", "").startswith("application/json")
                     else response.text),
    }
    evidence["steps"].append(entry)
    return entry


def check(name: str, ok: bool, detail: str) -> bool:
    evidence["assertions"].append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    return bool(ok)


def main() -> int:
    failures = 0
    with httpx.Client(base_url=BASE, timeout=300.0) as client:
        # ------------------------------------------------------------------
        # Step 1 — create + diagnose the Payment API incident (live Groq)
        # ------------------------------------------------------------------
        print("Step 1: create Payment API incident ...")
        r = client.post("/incidents", json={"service": "Payment API", "symptom": PAYMENT_SYMPTOM})
        record("POST /incidents (Payment API)", {"service": "Payment API", "symptom": PAYMENT_SYMPTOM}, r)
        assert r.status_code == 200, r.text
        pay = r.json()
        pay_id = pay["incident_id"]
        print(f"  incident #{pay_id}, model={pay.get('model_used')}, hits={len(pay.get('memory_hits', []))}")

        failures += not check(
            "A1 suggested_fix_steps non-empty from live Groq (bug A)",
            bool(pay.get("suggested_fix_steps")),
            f"{len(pay.get('suggested_fix_steps', []))} step(s): {pay.get('suggested_fix_steps', [])[:2]}…",
        )
        failures += not check(
            "A2 current incident absent from its own memory_hits (bug B)",
            all(h.get("historical_incident_id") != pay_id for h in pay.get("memory_hits", [])),
            f"hits={[h.get('historical_incident_id') for h in pay.get('memory_hits', [])]}",
        )

        # ------------------------------------------------------------------
        # Step 2 — human-confirmed resolution (retention happens HERE)
        # ------------------------------------------------------------------
        print("Step 2: resolve Payment API with outcome=success ...")
        resolve_body = {
            "root_cause_family": "db_connection_pool_exhaustion",
            "fix_steps": [
                "Identify code paths leaking MongoDB connections",
                "Fix connection reuse so every acquired connection is released",
                "Increase maxPoolSize after verifying database capacity",
                "Monitor connection pool utilization and HTTP 503 rate",
            ],
            "outcome": "success",
        }
        r = client.post(f"/incidents/{pay_id}/resolve", json=resolve_body)
        record(f"POST /incidents/{pay_id}/resolve", resolve_body, r)
        assert r.status_code == 200, r.text
        resolved = r.json()
        failures += not check(
            "A3 episodic retention after human resolution succeeded",
            resolved.get("memory_status") == "ok" and not resolved.get("errors"),
            f"memory_status={resolved.get('memory_status')} errors={resolved.get('errors')}",
        )

        # ------------------------------------------------------------------
        # Step 3 — the ORDER API incident (must use Payment as history)
        # ------------------------------------------------------------------
        print("Step 3: create Order API incident ...")
        r = client.post("/incidents", json={"service": "Order API", "symptom": ORDER_SYMPTOM})
        record("POST /incidents (Order API)", {"service": "Order API", "symptom": ORDER_SYMPTOM}, r)
        assert r.status_code == 200, r.text
        order = r.json()
        order_id = order["incident_id"]
        hits = order.get("memory_hits", [])
        text = order.get("diagnosis_text") or ""
        print(f"  incident #{order_id}, root cause guess={order.get('root_cause_family_guess')!r}")
        print(f"  hits={[(h.get('historical_incident_id'), h.get('service')) for h in hits]}")

        # (a) bug A — steps survive the whole live pipeline
        failures += not check(
            "B1 suggested_fix_steps non-empty on the live response (bug A)",
            bool(order.get("suggested_fix_steps")),
            f"{len(order.get('suggested_fix_steps', []))} step(s): {order.get('suggested_fix_steps', [])[:2]}…",
        )

        # (b) correct root cause family
        guess = (order.get("root_cause_family_guess") or "").lower()
        failures += not check(
            "B2 diagnosis = connection pool exhaustion",
            "pool" in guess and ("exhaust" in guess or "connection" in guess or "db_" in guess),
            f"root_cause_family_guess={order.get('root_cause_family_guess')!r}",
        )

        # (c) Payment API present as historical evidence
        hist_ids = [h.get("historical_incident_id") for h in hits]
        payment_ids = {13, 15, pay_id}
        payment_hit = next(
            (h for h in hits
             if h.get("historical_incident_id") in payment_ids
             or "payment" in (h.get("service") or "").lower()),
            None,
        )
        failures += not check(
            "B3 Payment API appears as historical evidence",
            payment_hit is not None,
            f"hit={payment_hit} (memory_ids of all hits: {[h.get('memory_id') for h in hits]})",
        )

        # (d) current incident never in its own hits
        failures += not check(
            "B4 current Order incident absent from its own memory_hits (bug B)",
            order_id not in hist_ids,
            f"historical_incident_ids={hist_ids}",
        )

        # (e) unresolved / unverifiable entries excluded (incident #14 style)
        all_incidents = client.get("/incidents").json()
        record("GET /incidents (history for provenance check)", {}, client.get("/incidents"))
        status_by_id = {i["id"]: i["status"] for i in all_incidents}
        unresolved = [
            h for h in hits
            if h.get("historical_incident_id") is not None
            and status_by_id.get(h["historical_incident_id"]) != "resolved"
        ]
        # Episodic hits must ALSO be verifiable: tied to a resolved incident id
        # or carrying a resolved marker in their content — a markerless
        # present-tense "report" memory (the incident #14 artifact class)
        # satisfies neither and must not appear.
        unverifiable = [
            h for h in hits
            if h.get("memory_type") == "episodic"
            and h.get("historical_incident_id") not in status_by_id
            and "resolved" not in (h.get("snippet") or "").lower()
        ]
        failures += not check(
            "B5 memory_hits contain only RESOLVED, verifiable historical incidents (#14 excluded)",
            not unresolved and not unverifiable,
            f"unresolved={[(h.get('historical_incident_id'), status_by_id.get(h.get('historical_incident_id'))) for h in unresolved]}"
            f" unverifiable={[(h.get('memory_id'), (h.get('snippet') or '')[:60]) for h in unverifiable]}"
            f" | #14 status={status_by_id.get(14)!r}",
        )

        # (f) Order API is never presented as a previous incident
        claims = [rx.pattern for rx in CLAIM_RES if rx.search(text)]
        failures += not check(
            "B6 Order API never claimed as a previous incident",
            not claims,
            f"matched patterns={claims}" if claims else "no previous/prior/earlier Order API phrasing",
        )

        # (g) memory used successfully (not a silent failure)
        failures += not check(
            "B7 memory search succeeded (ok, no errors) — failure != 0 results",
            order.get("memory_status") == "ok" and not order.get("memory_errors"),
            f"memory_status={order.get('memory_status')!r} errors={order.get('memory_errors')}",
        )

        # (h) trace honesty: returned/accepted/rejected recorded and accounted for
        trace = order.get("memory_trace") or []
        epi = next((t for t in trace if t.get("tier") == "episodic"), None)
        accounting_ok = epi is not None and (
            epi.get("accepted", 0) + epi.get("rejected", 0) == epi.get("returned", -1)
        )
        failures += not check(
            "B8 memory_trace recorded with returned/accepted/rejected counts",
            bool(trace) and accounting_ok
            and all(t.get("status") in ("ok", "failed") for t in trace),
            json.dumps([{k: t.get(k) for k in ("tier", "status", "returned", "accepted", "rejected")}
                        for t in trace]),
        )

        # ------------------------------------------------------------------
        # Step 4 — persisted incident records read back through FastAPI
        # ------------------------------------------------------------------
        print("Step 4: read both incidents back from SQLite via FastAPI ...")
        r_pay = client.get(f"/incidents/{pay_id}")
        r_ord = client.get(f"/incidents/{order_id}")
        record(f"GET /incidents/{pay_id}", {}, r_pay)
        record(f"GET /incidents/{order_id}", {}, r_ord)
        pay_db, ord_db = r_pay.json(), r_ord.json()

        failures += not check(
            "C1 suggested_fix_steps persisted in SQLite and served by GET",
            bool(pay_db.get("suggested_fix_steps")) and bool(ord_db.get("suggested_fix_steps")),
            f"payment={len(pay_db.get('suggested_fix_steps', []))} steps, order={len(ord_db.get('suggested_fix_steps', []))} steps",
        )
        failures += not check(
            "C2 predicted vs confirmed root cause stay separate fields",
            pay_db.get("predicted_root_cause_family") == pay.get("root_cause_family_guess")
            and pay_db.get("root_cause_family") == "db_connection_pool_exhaustion"
            and pay_db.get("predicted_root_cause_family") is not None
            and pay_db.get("root_cause_family") is not None
            and pay_db.get("outcome") == "success"
            and bool(ord_db.get("suggested_fix_steps")),
            f"predicted={pay_db.get('predicted_root_cause_family')!r} "
            f"(agent guess at diagnosis={pay.get('root_cause_family_guess')!r}, "
            f"survived human resolve) "
            f"confirmed={pay_db.get('root_cause_family')!r} outcome={pay_db.get('outcome')!r}",
        )
        failures += not check(
            "C3 diagnosis saved against the right incident (status flow)",
            pay_db.get("status") == "resolved" and ord_db.get("status") == "diagnosed",
            f"payment={pay_db.get('status')} order={ord_db.get('status')}",
        )

        print("\nDiagnosis text for incident #%d:\n%s\n" % (order_id, text))

    evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
    evidence["failures"] = failures
    evidence["result"] = "PASS" if failures == 0 else f"FAIL ({failures})"
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    print(f"Result: {evidence['result']} — evidence written to {EVIDENCE_PATH}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    t0 = time.time()
    code = main()
    print(f"elapsed: {time.time() - t0:.1f}s")
    sys.exit(code)
