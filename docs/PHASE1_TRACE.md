# MemorySRE — Phase 1 trace (root causes found before any code change)

## Bug A — missing `suggested_fix_steps`

End-to-end trace of the field:

1. Groq tool call: `submit_diagnosis` schema in `backend/app/agent.py` does **NOT**
   require `suggested_fix_steps` (only `diagnosis_text` + `primary_memory_type_used`
   are required) and the parameter has **no description**. The SYSTEM_PROMPT only
   mentions steps in the "if procedural memory returns a runbook" branch — so when
   there is no runbook, the model is never told it must still emit steps. -> often
   omitted (or emitted as a single vague string).
2. Parsed OK: `agent.py` `args.get("suggested_fix_steps", []) or []` -> `DiagnosisResult`
   (works when the model sends it).
3. **BROKEN — persistence boundary**: `store.save_diagnosis()` never writes
   `suggested_fix_steps`. There is no column for it (only `fix_steps_json`, which is
   written by `save_resolution` = the *confirmed* fix). So the recommended steps
   exist only in the POST /incidents response body and are lost afterwards.
4. FastAPI `GET /incidents` / `GET /incidents/{id}` -> `IncidentRecord.fix_steps`
   comes from `fix_steps_json` -> `[]` for any unresolved incident.
5. React `onOpen` (Overview + History) maps `suggested_fix_steps: i.fix_steps`
   -> `[]` -> IncidentPage renders "No runbook returned."
   (It also conflates the *confirmed* fix with the *recommended* fix.)
6. Fallback paths in `agent.py` (no tool_calls / max turns / LLM error) return
   `DiagnosisResult` with no steps at all, even when a procedural runbook was
   retrieved earlier in the loop.

=> Root cause A: (a) tool contract + prompt do not force steps, (b) steps are never
persisted to SQLite so any re-read loses them, (c) non-happy paths drop them.

## Bug B — current incident presented as historical memory

1. `POST /incidents` calls `memory.retain_episodic_report(...)` **before**
   `diagnose_incident(...)` — the current incident is written into the
   `incident-agent-episodic` Hindsight bank immediately.
2. The very next step recalls that bank. Hindsight returns the just-inserted
   `[INCIDENT #14 REPORTED] ...` memory (confirmed live: recall returns both raw
   "world" facts and derived observations; scores are top-ranked).
3. `_tool_dispatch()` accepts every result verbatim: no provenance, no incident id,
   no check against the current incident. `MemoryHit` only has
   `memory_type/query/snippet`.
4. The LLM is told this is a "similar past incident" and the UI renders it as
   "previous similar incident: Order API".
5. `memory_hit_type` is taken from the model's `primary_memory_type_used` claim ->
   a self-hit sets `memory_hit_type="episodic"` -> `store.get_stats()` counts it in
   the memory hit rate (metric inflation).
6. Recall failures are swallowed (`_recall` returns `[]` on exception) so
   "Hindsight is down" and "0 relevant memories" are indistinguishable.

=> Root cause B: premature retention (ordering) + zero application-level provenance
validation + a metric that trusts the LLM's claim.

## Secondary findings

- `store.save_diagnosis()` writes the agent prediction into `root_cause_family`,
  and `save_resolution()` overwrites the same column with the confirmed value ->
  predicted/confirmed are not distinct fields.
- `consolidation.maybe_promote_procedural()` is called on every resolve regardless
  of `outcome`; the threshold check uses the *aggregate* rate, so a FAIL resolution
  can still promote its own (failed) fix steps as a runbook once the family's
  running rate clears 0.66 (e.g. 2 success + 1 fail = 0.667).
- Hindsight SDK installed: **hindsight-client 0.10.1** (requirements pins
  `>=0.0.18`). Async `aretain`/`arecall`/`acreate_bank`/`alist_memories` are the
  FastAPI-safe methods (as memory.py already documents).
- Live probes established:
  * `aretain(metadata=..., tags=...)` round-trips through `arecall` (result.metadata
    and result.tags come back populated) -> real provenance source.
  * `arecall` returns `RecallResult{id, text, metadata, tags, occurred_start,
    chunk_id, scores.final, ...}`.
  * **`result.text` is an LLM rephrasing** of the retained content (ordered fix
    steps collapse to "the prescribed fix steps are ..."). The ORIGINAL retained
    content (with `[RUNBOOK] ... Ordered fix steps: 1. ...` markers and
    `[INCIDENT #13 RESOLVED] service=... outcome=success`) is available via
    `arecall(..., include_chunks=True)` -> `response.chunks[chunk_id].text`.
  * Old (pre-metadata) memories have `metadata: {}` -> provenance for them must
    fall back to chunk/raw text markers + SQLite status checks.
