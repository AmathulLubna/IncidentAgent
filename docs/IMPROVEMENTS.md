# Improvements: evidence, recommendations, memory layers

## 1. Historical evidence now matches the incident
**Cause:** Hindsight recall is fuzzy; on a small bank it returned the same memories for any query. Provenance checks (resolved? not current?) cannot tell an auth incident from a disk-full one.
**Fix:**
- `taxonomy.py` classifies every incident (15 categories), extracts signals (HTTP codes, %, error names) and scores text similarity.
- `agent.search_tier` anchors every recall query on the incident's own category + signals, then applies a **topical gate**: a memory about a different kind of problem is dropped (`off_topic` in the memory trace).
- Topic-ranked matches from local SQLite history / layer store fill gaps Hindsight misses (`local_added`).
- `ground_diagnosis_text` removes sentences citing incident ids that were not accepted evidence and adds a cold-start note when no history matched.

## 2. Developer recommendations
`playbooks.py` holds a problem → solution playbook per category: problem, likely causes, do-now, read-only investigation commands (with the service name filled in), permanent fix, verify, prevent, rollback, caution. Fixes engineers confirmed in topically matching past incidents are attached as "Confirmed in your own history". Shown in the new **Developer recommendations** card. If the LLM fails, the playbook still supplies category-specific steps.

## 3. Memory layers store three different things
`layers.py` builds three documents per resolved incident:

| Layer | Answers | Content |
|---|---|---|
| Episodic | What happened | symptoms, signals, timeline, actions, outcome |
| Semantic | Why it happens | signature, mechanism, confused-with, applies-when, evidence |
| Procedural | How to fix it | trigger, ordered steps, verify, rollback, track record (success only) |

Each goes to its own Hindsight bank and to a local `memory_layers` table, so search is layer-specific and deterministic. Explorer has an **All layers** view (one keyword, three columns). Failed fixes are kept as episodic + semantic lessons, never as a runbook.

## Upgrading an existing database
Start the backend (auto-backfills local layer rows), or run `python backend/scripts/rebuild_layers.py [--hindsight]`.
