# How this project uses Hindsight

Hindsight is the entire memory layer for this agent. There are no
substitutes or shadow vector stores — every piece of "the agent remembers /
learns" behavior is a Hindsight `retain` or `recall` call. This document
explains exactly which calls happen, when, and why, mapped to the three
memory tiers in the project plan.

## Three banks = three tiers

Hindsight's primitive is a **memory bank** (`retain` / `recall` / `reflect`
against a `bank_id`). This project creates three banks per deployment:

```
{project}-episodic
{project}-semantic
{project}-procedural
```

(`backend/app/memory.py::ensure_banks`, called on FastAPI startup.)

This is an application-level design choice, not a Hindsight feature — the
three-tier split is the project's differentiator, and giving each tier its
own bank keeps it visibly distinct in the Hindsight Cloud dashboard, which
matters for the "memory is visibly central" judging weight.

## Episodic memory — `retain`, written only AFTER human resolution

An incident produces exactly **one** episodic retain, and it happens in
`POST /incidents/{id}/resolve` (`retain_episodic_resolution`) — never at
ingest:

1. **On ingest: nothing is retained.** The current incident must not be
   retrievable as "historical memory" during its own diagnosis, so the agent
   recalls first and diagnoses second, with no self-reference possible.
2. **On resolution:** the reported symptoms, the engineer-confirmed root
   cause, the confirmed fix steps and the outcome (success/fail) are retained
   together with structured `metadata` (`incident_id`, `service`,
   `occurred_at`, `status=resolved`, `outcome`, `root_cause_family`) and tags
   (`historical-incident`, `incident-<id>`, `outcome-<x>`).

Failed resolutions are retained too (as negative evidence: "this fix did not
work"), which is exactly what lets a future diagnosis learn what *not* to
repeat.

## Episodic memory — `recall` and provenance

When a new incident comes in, the diagnosis agent (an LLM tool-calling
loop, `backend/app/agent.py`) *may* call `search_episodic_memory`, which runs
`client.arecall(bank_id=episodic_bank, query=..., include_chunks=True)`.

Before anything reaches the model or the UI, every result passes an
application-level provenance gate (`agent.historical_verdict`):

- a hit whose incident id equals the **current** incident id is rejected,
- a hit whose incident is not `resolved` in SQLite incident history is rejected,
- a hit whose incident id does not exist in SQLite history at all (and that
  carries no trusted `status=resolved` marker) is rejected — provenance that
  cannot be verified is not evidence,
- an **episodic** hit with no provenance whatsoever (no incident id, no
  resolved marker — e.g. a stale markerless *report* memory retained by old
  builds) is rejected; semantic/procedural hits legitimately carry no
  incident id and are unaffected,
- rejected results are counted in the memory trace, and the LLM is told how
  many were rejected and why,
- `memory_hit_type` is derived only from *accepted* hits, so a self-reference
  can never inflate the memory-hit metric.

Each accepted hit carries provenance populated only from what Hindsight
actually returned: recall result `id`, `metadata`, raw chunk text
(`include_chunks=True`, because Hindsight rephrases `result.text` and the
original `[HISTORICAL INCIDENT #n RESOLVED] service=… outcome=…` markers live
in the chunk), `occurred_start` and `scores.final`. Missing fields stay `null`.

If a recall call itself fails, the failure is surfaced as
`memory_status='degraded'` + `memory_errors` (and a failed trace step) — it is
never silently reported as "0 relevant memories".

## Semantic memory — `retain`, written on a schedule

Semantic memory is **not** written per-incident. Every
`CONSOLIDATION_EVERY_N` (default 6) resolved incidents,
`backend/app/consolidation.py::run_consolidation` runs:

1. Pull the last N resolved incidents from local bookkeeping (SQLite —
   Hindsight itself isn't the source of truth for "how many incidents have
   we resolved since last consolidation", that's ordinary app state).
2. Group by confirmed `root_cause_family`. Any family that occurred at
   least `CONSOLIDATION_MIN_OCCURRENCES` (default 3) times in the window is
   a candidate pattern.
3. Ask Groq to phrase the pattern in one sentence, grounded in 1–3 real
   example symptoms from that window (not invented).
4. Compute the confidence stat **arithmetically** from the real counts
   (`count / window_size`), not asked of the LLM — this is deliberate: the
   plan's example stat ("73% of your last 6 auth incidents...") needs to be
   true, not a plausible-sounding hallucination.
5. `retain_semantic_pattern` writes one fact per qualifying family into the
   semantic bank, with the confidence stat embedded in the text itself.

## Semantic memory — `recall`

The agent's `search_semantic_memory` tool calls
`client.recall(bank_id=semantic_bank, query=...)`. If Hindsight returns a
pattern fact, the agent is instructed to surface the embedded confidence
percentage explicitly in the diagnosis text and in the structured
`confidence_pct` field — this is the "73% of your last 6 auth incidents had
this root cause" moment from the demo script.

## Procedural memory — `retain`, written by the feedback loop

Every time a fix is confirmed (`POST /incidents/{id}/resolve`), two things
happen in `backend/app/consolidation.py::maybe_promote_procedural`:

0. The **outcome must be a human-confirmed `success`**. A `FAIL` resolution is
   retained as episodic (negative) evidence but is never promoted — even when
   the family's aggregate success rate still clears the threshold.
1. Recompute that root-cause family's success rate from local bookkeeping
   (attempts and success/fail counts — again, ordinary app state, not
   Hindsight).
2. If attempts ≥ `PROCEDURAL_MIN_ATTEMPTS` (default 2) **and** success rate
   ≥ `PROCEDURAL_MIN_SUCCESS_RATE` (default 0.66), retain a runbook fact:
   symptom family → ordered fix steps → success rate, into the procedural
   bank (with `metadata` such as `root_cause_family`, `attempts`,
   `success_rate`, `outcome=success`).

This is the literal "feedback loop" from the plan: successful fixes gain
weight (they're what gets promoted at all), and a family that stops
succeeding will simply stop clearing the bar for future promotion/refresh
— there's no separate "deprioritize" retain, the bar itself does that job.

## Procedural memory — `recall`

`search_procedural_memory` calls `client.arecall(bank_id=procedural_bank,
query=..., include_chunks=True)`. Hindsight's `result.text` is a rephrasing
("the prescribed fix steps are …"), so the agent reads the **raw chunk** to
recover the actual ordered steps. Those steps become the basis of
`suggested_fix_steps` — and if the model submits an empty array anyway,
`submit_diagnosis` is rejected with a corrective tool result (bounded retries)
and the steps are derived from the retrieved runbook as a last resort.

## What Hindsight does NOT do here

To be precise about the boundary: Hindsight does not track incident
counts, success rates, or the UI's rolling accuracy metric — those are
plain SQLite bookkeeping (`backend/app/store.py`) that *decides when* to
call Hindsight's `retain`. Hindsight is the memory itself (the facts an
on-call engineer would want recalled); the bookkeeping is the app's own
state about its own behavior. Keeping that boundary explicit was a
deliberate choice so it's obvious, in code and in the demo, which parts are
"the compulsory memory layer" versus ordinary application logic.
