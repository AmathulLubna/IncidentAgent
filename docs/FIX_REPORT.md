# §34 Final Report — Bug A (`suggested_fix_steps` missing) & Bug B (current incident as historical memory)

**Status: both bugs fixed, 15/15 automated tests pass, live acceptance (§33) passes 13/13.**
No architecture was rebuilt: same FastAPI + SQLite + Groq + Hindsight design, no new
infrastructure. All changes are application-level.

---

## 1. Root causes

### Bug A — `suggested_fix_steps` missing from diagnoses (four contributing causes)

| # | Cause | Where |
|---|-------|-------|
| A1 | The `submit_diagnosis` tool schema had no required `suggested_fix_steps` parameter (no description; the prompt only mentioned steps in the runbook context), so Groq could legitimately omit it. | `backend/app/agent.py` |
| A2 | `store.save_diagnosis()` never persisted the field — there was no `suggested_fix_steps_json` column, so even a good tool call was lost at SQLite. | `backend/app/store.py` |
| A3 | The frontend read the *engineer-confirmed* field instead: `onOpen` mapped `suggested_fix_steps: i.fix_steps`, so the UI showed an empty array unless a human had already fixed it. | `frontend/src/main.jsx` |
| A4 | Internal fallback paths (validation-retry, runbook-derived steps) dropped steps on some branches instead of guaranteeing a non-empty array. | `backend/app/agent.py` |

### Bug B — the current incident presented as historical memory (four contributing causes)

| # | Cause | Where |
|---|-------|-------|
| B1 | `POST /incidents` called `retain_episodic_report()` **before** diagnosis, writing the in-flight incident into Hindsight, where the same diagnosis's `search_episodic_memory` could recall it as a "previous" incident. | `backend/app/routers/incidents.py` (old code) |
| B2 | Zero provenance validation: recalled memories were handed to the model as-is — no check that a hit's incident id ≠ current incident, or that the referenced incident was actually `resolved` in history. | `backend/app/agent.py` (old code) |
| B3 | `memory_hit_type` trusted the LLM's `primary_memory_type_used` claim, so a self-hit inflated the memory-hit-rate metric. | `backend/app/agent.py` (old code) |
| B4 | No separation between agent **predicted** and engineer-confirmed root cause/fix in storage, and `memory_hit_type`/hits were persisted without provenance. | `backend/app/store.py`, `schemas.py` |

The Phase-1 evidence trail for these causes is in `docs/PHASE1_TRACE.md`.

---

## 2. What changed (files)

### Backend — agent & memory boundary
- **`backend/app/agent.py`** — full rewrite of the diagnosis flow:
  - `historical_verdict()` provenance gate (application code, not the prompt), applied to
    every recalled result **before** it reaches the model or the UI:
    1. hit's incident id == current incident → rejected;
    2. explicit `status` other than `resolved` → rejected;
    3. incident exists in SQLite but is not `resolved` (e.g. incident #14) → rejected;
    4. incident id **does not exist** in SQLite history and has no trusted
       `status=resolved` marker → rejected (unverifiable provenance);
    5. **episodic** hit with no provenance at all (no id, no marker — the stale
       markerless *report* memories) → rejected. Semantic/procedural tiers are
       cross-incident patterns/runbooks and legitimately carry no incident id.
  - Strict `submit_diagnosis` contract: non-empty `suggested_fix_steps` required;
    invalid submissions get a corrective tool result and a bounded retry
    (`MAX_SUBMIT_REJECTIONS=2`); if the model still fails, steps are derived from a
    retrieved procedural runbook — a successful diagnosis always ends with steps.
  - `memory_hit_type` is derived from **accepted** hits only; rejections are counted in
    `memory_trace` (`returned/accepted/rejected`) and logged with reasons.
  - Recall failures surface as `memory_status='degraded'` + `memory_errors` — never
    silently reported as "0 memories".
  - Updated §12-style system prompt + tool descriptions (memory hits are stated to be
    historical, current incident never included; no destructive commands ever executed).
- **`backend/app/memory.py`** — `RecallBatch` (list-like with `.error`) distinguishes
  *search failed* from *0 results*; all `aretain` calls carry `metadata` + `tags`
  (`incident_id`, `service`, `outcome`, `root_cause_family`, `memory_type`) so provenance
  round-trips through `arecall`; raw retained content fetched with
  `arecall(include_chunks=True)` because Hindsight rephrases `result.text`;
  `retain_episodic_report()` **removed** — episodic retention happens only on resolve.

### Backend — API, storage, promotion
- **`backend/app/routers/incidents.py`** — `POST /incidents` no longer retains anything
  into Hindsight before diagnosis; `POST /incidents/{id}/resolve` retains the episodic
  resolution only after the human confirms the outcome and returns richer diagnostics
  (memory status, promotion, success rate).
- **`backend/app/store.py`** — new columns + `_migrate()` for existing DBs:
  `suggested_fix_steps_json`, `predicted_root_cause_family`, `memory_hits_json`,
  `memory_trace_json`, `memory_status`, `memory_errors_json`, plus
  `save_diagnosis(incident_id, diagnosis)` persisting steps/hits/trace honestly.
- **`backend/app/schemas.py`** — `MemoryHit` (provenance: `memory_id`,
  `historical_incident_id`, `service`, `timestamp`, `root_cause_family`, `outcome`,
  `relevance_score` — all nullable, never fabricated), `MemoryTraceStep`,
  `DiagnosisResult.memory_errors/memory_trace`, `IncidentRecord` with **separate**
  `predicted_root_cause_family` vs `root_cause_family` and
  `suggested_fix_steps` vs `fix_steps`.
- **`backend/app/consolidation.py`** — procedural promotion only on human-confirmed
  `outcome == "success"` (existing thresholds unchanged); failed fixes are never promoted.
- **`backend/app/routers/memory_routes.py`** — recall errors surfaced to the UI.

### Frontend
- **`frontend/src/main.jsx`** — incident view split into CURRENT INCIDENT / DIAGNOSIS /
  HISTORICAL EVIDENCE / RECOMMENDED ACTIONS / MEMORY USED sections; confirmed-resolution
  panel; `onOpen` now maps `suggested_fix_steps` from the diagnosis field (not
  `fix_steps`); history shows predicted vs confirmed labels.
- **`frontend/src/styles.css`** — styles for the new sections/hit/trace components.
- `npm.cmd run build` succeeds (`dist/assets/index-*.js` 252 kB).

### Docs
- `docs/PHASE1_TRACE.md` (Phase-1 trace), `docs/HINDSIGHT_MEMORY_USAGE.md` (gate rules),
  `docs/acceptance_33_evidence.json` (live evidence), `README.md` (bank table).

---

## 3. Tests (§22) — 15 passed, 0 failed

Run: `backend\venv\Scripts\python.exe -m pytest backend/tests -q` → **`15 passed`**

| §22 scenario | Test | File |
|---|---|---|
| 1. Current incident can never become historical memory | `test_current_incident_is_never_historical_memory` | `backend/tests/test_regression.py` |
| 2. Previous incident retrieved, current one not | `test_previous_incident_is_retrieved_as_historical_evidence` | 〃 |
| 3. `suggested_fix_steps` survives Groq → SQLite → API | `test_suggested_fix_steps_survive_end_to_end` | 〃 |
| 4. Procedural memory produces steps when omitted | `test_procedural_memory_produces_steps_when_omitted` | 〃 |
| 5. Failed resolution never promoted | `test_failed_resolution_is_not_promoted` | 〃 |
| 6. Predicted vs confirmed root cause distinct | `test_predicted_and_confirmed_root_cause_are_distinct` | 〃 |
| extra | `test_unverifiable_incident_id_is_not_evidence` (gate rule 4) | 〃 |
| extra | `test_markerless_episodic_memory_is_not_evidence` (gate rule 5, §9) | 〃 |
| extra | `test_recall_failure_is_reported_as_failure` (§23) | 〃 |
| extra | `test_incident_is_not_retained_before_diagnosis` (§6/§21) | 〃 |
| original unit tests (4) | confidence, thresholds, store, evidence, API contract | `backend/tests/test_core.py` |

All regression tests are network-free: Groq is scripted and Hindsight recall/retain stubbed.

---

## 4. Live acceptance (§33) — 13/13 PASS

Run: real `uvicorn` server, live Groq (`openai/gpt-oss-120b`) + live Hindsight Cloud,
driver `backend/scripts/acceptance_33.py`, full evidence in
**`docs/acceptance_33_evidence.json`** (every request/response + assertion).

Scenario: create+diagnose "Payment API / HTTP 503 / pool exhausted" → human-resolve
`SUCCESS` → create "Order API / HTTP 503 / pool exhausted / 97% utilization / high latency".

| Check | Result |
|---|---|
| A1 `suggested_fix_steps` non-empty from live Groq | PASS (5 steps) |
| A2 current incident absent from its own `memory_hits` | PASS (`hits=[13, 15, …]`, own id never present) |
| A3 episodic retention only after human resolution | PASS (`memory_status=ok, errors=[]`) |
| B1 steps non-empty on live response (bug A) | PASS (4 steps) |
| B2 diagnosis = connection pool exhaustion | PASS (`db_connection_pool_exhaustion`) |
| B3 Payment API appears as historical evidence | PASS (`#15` hit, `[HISTORICAL INCIDENT #15 RESOLVED]` markers) |
| B4 current Order incident absent from its own hits | PASS (`ids=[15, …]`) |
| B5 only RESOLVED, verifiable incidents (#14 excluded) | PASS (`#14 status='diagnosed'`, unverifiable hits empty) |
| B6 Order API never claimed as previous | PASS — diagnosis says *"mirrors historical Incident #15 on the Payment API"* |
| B7 recall succeeded (failure ≠ 0 results) | PASS (`memory_status='ok'`) |
| B8 trace honest: accepted + rejected == returned | PASS (`episodic returned=5, accepted=1, rejected=4`) |
| C1 steps persisted in SQLite, served by GET | PASS (5 + 4 steps) |
| C2 predicted vs confirmed separate fields | PASS (prediction survived human resolve in its own column) |
| C3 status flow (resolved / diagnosed) | PASS |

Server logs corroborate the gate live, e.g.:
`provenance: rejected episodic memory b48b32b6… — episodic memory carries no verifiable
incident provenance (no incident id and no resolved status marker)` and
`… rejected … — incident #14 is 'diagnosed' in incident history`.

---

## 5. Remaining issues / known limitations

1. **Stray SDK-probe memories in the live episodic bank** (8 entries, e.g. incident
   `#999999` / "probe-service"). The Hindsight SDK exposes **no per-memory delete** —
   only `clear_bank_memories` (whole bank), which would destroy 58 real memories, so
   they were left in place. They are now harmless: the provenance gate rejects them
   (non-existent incident id, or no verifiable episodic provenance) before they can
   reach the model or UI. Documented/inspectable via `backend/scripts/inspect_probe.py`.
2. **Old markerless memories are conservatively rejected.** A few genuinely-resolved
   legacy memories (e.g. `bf7685fe…`, retained by old builds without markers/metadata)
   no longer count as evidence — correct for safety, at a small recall-coverage cost.
3. **Acceptance-run artifacts in `incident_agent.db`**: incidents #15–#21 were created by
   the live runs (#19 is an unresolved Payment API left by a crashed driver run — its
   non-existence in Hindsight is itself proof that nothing is retained before resolve).
   Clear with `POST /demo/reset {"confirm": true}` if a clean demo state is wanted.
4. **Groq rate limits (429)** occur occasionally during rapid test runs; the OpenAI
   client's built-in retry handled every occurrence.
5. **Semantic bank is empty** (0 memories) — semantic distillation runs on the
   consolidation schedule and was out of scope for these fixes.
6. Everything is **uncommitted** (repo has a single `first commit`); the change set is
   exactly the `git status` output: 12 modified files + 6 new
   (`backend/tests/test_regression.py`, `backend/scripts/acceptance_33.py`,
   `backend/scripts/inspect_probe.py`, `docs/PHASE1_TRACE.md`,
   `docs/acceptance_33_evidence.json`, `docs/FIX_REPORT.md`).

### How to re-verify
```powershell
# unit + regression (15 tests, no network)
backend\venv\Scripts\python.exe -m pytest backend/tests -q
# live acceptance (start server first)
cd backend; .\venv\Scripts\python.exe -m uvicorn app.main:app --port 8010
.\venv\Scripts\python.exe scripts\acceptance_33.py
# frontend
cd frontend; npm.cmd run build
```
