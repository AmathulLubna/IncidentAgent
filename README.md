# Incident Response Agent

An AI agent that diagnoses production incidents and gets measurably better at
it over time — by remembering past incidents, distilling patterns across
them, and building reusable runbooks — instead of treating every incident as
a cold start. Memory is powered by **Hindsight** (Vectorize); reasoning is
powered by **Groq**.

## Architecture

```
Streamlit (UI) ⇄ FastAPI (agent loop) ⇄ Groq (LLM, tool-calling)
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     Episodic       Semantic      Procedural
     memory         memory        memory
    (Hindsight)    (Hindsight)   (Hindsight)
```

Three Hindsight memory banks, one per tier:

| Bank | Contents | Written by |
|---|---|---|
| `{project}-episodic` | Resolved incidents: symptoms + confirmed root cause + confirmed fix + outcome | On engineer-confirmed resolution (never at ingest) |
| `{project}-semantic` | Distilled cross-incident patterns + confidence stat | Periodic consolidation pass (every N resolved incidents) |
| `{project}-procedural` | Confirmed runbooks (symptom → ordered steps → success rate) | Feedback loop, once a fix clears the attempts/success-rate bar |

See [`docs/HINDSIGHT_MEMORY_USAGE.md`](docs/HINDSIGHT_MEMORY_USAGE.md) for the
full write-up of how each Hindsight operation (`retain` / `recall`) is used.

> **New:** topic-matched historical evidence, per-category developer recommendations and three distinct memory layers. See [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# then fill in GROQ_API_KEY and HINDSIGHT_API_KEY in .env
```

Hindsight Cloud: sign up at ui.hindsight.vectorize.io, apply promo
`MEMHACK99` in the billing section, then grab your API key + base URL from
the **Connect** page and put them in `.env`.

## Run

**Backend:**
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

**React frontend** (separate terminal):
```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally http://localhost:5173).
The frontend uses `VITE_API_URL` and defaults to `http://localhost:8000`.

## Seeding memory for the demo

To go straight to "a later incident flags a pattern" / "a runbook shows up"
without manually typing a dozen incidents live:

1. Fill in real incidents in `data/real_seed_incidents.json` (pull 5–8 from
   your own repos' Issues/Actions failure logs) and
   `data/postmortem_seed_incidents.json` (3–5, paraphrased from public
   postmortems, with `source_url` kept for citation).
2. (Optional) Generate synthetic filler:
   ```bash
   python data/generate_synthetic.py --per-family 4
   ```
3. With the backend running, seed everything through the live API:
   ```bash
   python data/seed_memory.py
   ```

This walks incidents through the *real* ingest → diagnose → resolve →
consolidate pipeline (not a bulk import), so the memory banks, success
rates, and accuracy metric all end up in a genuinely-reached state rather
than a faked one.

## Demo script (2–5 min video)

| Time | Beat |
|---|---|
| 0:00–0:30 | Intro — who you are, what the agent does |
| 0:30–1:00 | Incident 1: no memory hit — generic, hedged diagnosis |
| 1:00–1:30 | Incidents 2–6: watch episodic memory grow live in the sidebar |
| 1:30–2:30 | A repeat incident: semantic memory flags a pattern with a confidence % on screen |
| 2:30–3:00 | Same family recurs again: procedural memory returns a step-by-step runbook |
| 3:00–3:30 | Wrap-up — one thing that surprised you building this |

Exact incident numbers depend on your seed data mix — run `data/seed_memory.py`
against your own data first and check `/stats` + the "Explore memory" tab to
see which incident indices actually trigger each beat, then narrate the live
demo around those.

## Key design decisions / known limitations

- **Three banks, not one with tags.** Simpler to demo, and keeps the
  episodic/semantic/procedural split visible in the Hindsight dashboard
  itself, not just in application code.
- **Bookkeeping (success rates, the accuracy metric) lives in local SQLite,
  not Hindsight.** Hindsight is the memory layer; the app still needs plain
  counters to decide *when* to consolidate or promote a runbook.
- **No in-place runbook update.** Hindsight is retain-only, so a refreshed
  runbook is retained as a new fact rather than editing the old one; recall
  is expected to surface the most relevant/freshest version. Fine for a
  hackathon demo's data volume; would need de-duplication at real scale.
- **Groq function-calling fallback**: every diagnosis call tries
  `openai/gpt-oss-120b` first; on an API error or malformed tool-call
  arguments, it retries once against `qwen/qwen3-32b` before degrading to a
  safe "please triage manually" response rather than crashing.
- **Synthetic vs. real data**: honestly disclosed — see
  `data/postmortem_seed_incidents.json` / `data/real_seed_incidents.json`
  vs. `data/generate_synthetic.py`. This split should also be called out in
  the article's "lessons learned" section per the content guide.

## Build order (for reference)

- [x] Hindsight Cloud client wired up (`backend/app/memory.py`)
- [x] FastAPI backend + Groq client with fallback
- [x] Synthetic incident generator
- [x] Real/postmortem seed templates
- [x] Episodic retain/recall
- [x] Consolidation pass → semantic memory
- [x] Fix-confirmation → procedural memory promotion
- [x] Success/fail tagging + confidence scoring
- [x] React/Vite SRE control-room UI (incident feed, live memory calls, accuracy metric)
- [ ] Run through demo script end-to-end, record video *(you)*
- [ ] Write article + LinkedIn post, publish, submit *(you)*
