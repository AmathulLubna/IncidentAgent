"""
Core agent loop: diagnose the CURRENT incident by searching HISTORICAL memory
(episodic / semantic / procedural) and emit a structured, actionable diagnosis.

Two boundaries are enforced in *application code* — never left to the LLM:

1. PROVENANCE (current incident != historical memory)
   - The current incident is not retained into Hindsight until the engineer
     resolves it (see routers/incidents.py), so nothing written during this
     diagnosis can be recalled as a precedent.
   - Every recalled result still passes ``historical_verdict()`` before it
     reaches the model or the UI: a hit referencing the current incident id,
     any incident that is not resolved in incident history, an incident id
     that does not exist in history, or (episodic tier) a memory with no
     verifiable provenance at all, is rejected.
   - ``memory_hit_type`` is derived from the *accepted* historical hits, so a
     self-hit can never inflate the memory-hit metric.

2. CONTRACT (diagnosis must be actionable)
   - ``submit_diagnosis`` must carry a non-empty ``suggested_fix_steps`` array.
     Invalid submissions get a corrective tool result and another turn
     (bounded by MAX_SUBMIT_REJECTIONS); if the model still fails, steps are
     derived from a retrieved procedural runbook when one exists.
   - Recall failures are surfaced as ``memory_errors`` / ``memory_status=
     'degraded'`` instead of being silently reported as "no memories".
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from . import layers, memory, playbooks, store, taxonomy
from .config import settings
from .groq_client import achat_completion
from .schemas import DiagnosisResult, MemoryHit, MemoryTraceStep, Recommendation

logger = logging.getLogger("agent")

MAX_TURNS = 6
MAX_SUBMIT_REJECTIONS = 2
SNIPPET_LIMIT = 600        # what the UI sees of a memory
TOOL_TEXT_LIMIT = 1500     # what the model sees of a memory
TIERS = ("episodic", "semantic", "procedural")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_episodic_memory",
            "description": (
                "Search HISTORICAL past incidents (resolved before the current incident) that "
                "resemble the CURRENT incident's symptom. Returns historical incidents only — "
                "the current incident itself is never part of these results. Each result carries "
                "provenance: historical incident id, service, timestamp, confirmed root cause and "
                "outcome when known."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing the current incident's symptom/service",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_semantic_memory",
            "description": (
                "Search consolidated HISTORICAL patterns learned from multiple past incidents "
                "(e.g. 'HTTP 503 + high MongoDB connection utilization is frequently associated "
                "with connection pool exhaustion', with occurrence counts / confidence). These are "
                "generalized patterns, not individual incidents."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_procedural_memory",
            "description": (
                "Search previously SUCCESSFUL remediation runbooks (symptom -> ordered fix steps -> "
                "success rate), promoted only from human-confirmed successful resolutions. Returns "
                "the runbook's ordered steps — use them as the basis for suggested_fix_steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_diagnosis",
            "description": (
                "Submit the final diagnosis for the CURRENT incident. Must include ordered, "
                "actionable suggested_fix_steps — an empty array is rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "diagnosis_text": {
                        "type": "string",
                        "description": (
                            "3-6 sentences for the on-call engineer: root-cause hypothesis, the key "
                            "evidence from the CURRENT incident, and (if you used memory) which "
                            "historical incident/pattern supports it and why it is relevant. Never "
                            "claim a fix has already been performed."
                        ),
                    },
                    "root_cause_family_guess": {
                        "type": ["string", "null"],
                        "description": "Short slug, e.g. 'db_connection_pool_exhaustion'. null only if genuinely unknown.",
                    },
                    "suggested_fix_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Ordered remediation/verification actions for the CURRENT incident. Each "
                            "item is ONE concrete action (inspect / verify / check / change / "
                            "restart-if-needed / monitor). If a procedural runbook was retrieved, "
                            "adapt its steps in order. If no runbook exists, list safe diagnostic "
                            "and remediation steps derived from the current evidence and flag the "
                            "uncertainty in diagnosis_text. NEVER submit an empty array."
                        ),
                    },
                    "primary_memory_type_used": {
                        "type": "string",
                        "enum": ["none", "episodic", "semantic", "procedural"],
                        "description": (
                            "Which HISTORICAL memory tier most informed this diagnosis. Use 'none' "
                            "only when no historical memory was retrieved."
                        ),
                    },
                    "confidence_pct": {
                        "type": ["number", "null"],
                        "description": (
                            "0-100. Use the confidence stated by a semantic-memory pattern when one "
                            "was returned; otherwise give your own calibrated confidence."
                        ),
                    },
                },
                "required": [
                    "diagnosis_text",
                    "root_cause_family_guess",
                    "suggested_fix_steps",
                    "primary_memory_type_used",
                    "confidence_pct",
                ],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an SRE incident response agent for an engineering on-call team.

Your task is to diagnose the CURRENT incident using:
1. the current incident's evidence (service + symptoms + metrics/logs),
2. historical episodic memories (similar past incidents),
3. semantic patterns (generalized patterns learned from multiple past incidents),
4. previously successful procedural memories (runbooks).

REASONING CONTRACT
- Always separate CURRENT INCIDENT from HISTORICAL MEMORY. The current incident is
  happening now; everything the search tools return existed BEFORE it. Never
  describe the current incident as a previous incident, and never cite the current
  incident as historical evidence.
- Structure your answer conceptually as: current evidence -> diagnosis ->
  historical evidence -> recommendation. If historical memory is available:
  identify the historical incident/pattern, explain briefly why it is relevant to
  the current evidence, and adapt it — do not blindly copy it.
- Write for an on-call engineer reading this at 3am: plain language, name the
  exact past incident/pattern you relied on, and say in ONE sentence why it fits
  the current evidence (say so plainly if it came from a different service).
  Short and specific beats long and vague.
- If procedural memory exists, treat its successful steps as the primary source for
  suggested_fix_steps and adapt them to the current incident (do not just say "a
  similar incident happened before").
- If no procedural memory exists, produce safe diagnostic and remediation steps
  from the current evidence and clearly indicate the uncertainty in diagnosis_text.
- Do not claim that a fix has been performed. The human engineer confirms and
  executes the actual resolution; your output is a recommendation only.
- No chain-of-thought is needed — only a concise evidence and reasoning summary.

GROUNDING RULES (enforced again in code after you answer)
- The user message includes an INCIDENT CLASSIFICATION computed from the current evidence.
  Use it as your starting hypothesis; only override it when the evidence clearly says otherwise.
- Cite a historical incident (#id) or pattern ONLY if it appears in a search tool result in
  this conversation. Never invent incident ids, past outcomes or "we saw this before".
- Historical memories that are about a different kind of problem are removed for you. If a
  search returns 0 results, state plainly that no relevant history exists and treat the
  diagnosis as a cold-start hypothesis. Do not fill the gap with generic history.
- Your diagnosis_text and suggested_fix_steps must be specific to THIS incident's service,
  symptoms and metrics (quote the actual numbers / error names), not generic advice.

WORKFLOW
1. Read the current incident in the user message.
2. Call whichever search tools are worth checking (you do not have to call all
   three; skip tiers that cannot apply).
3. Read what comes back. If a tier returns 0 results, say so plainly rather than
   inventing a match. If a tool result carries an error, treat it as a failed
   search, not as "no memories".
4. Call submit_diagnosis exactly once with your final answer.

EVERY submit_diagnosis MUST CONTAIN
- diagnosis_text: root-cause hypothesis + confidence + supporting evidence +
  (when memory was used) which historical incident/pattern it resembles,
- root_cause_family_guess: short slug (or null),
- confidence_pct: 0-100,
- primary_memory_type_used: the tier that actually informed you ('none' only if no
  historical memory was retrieved,
- suggested_fix_steps: an ordered, non-empty array of concrete remediation or
  verification actions. Never submit an empty array — even with no runbook there
  are always safe inspection/verification steps to recommend.
"""


# ---------------------------------------------------------------------------
# Incident context + provenance parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IncidentContext:
    incident_id: int
    service: str
    symptom: str
    timestamp: str
    # Deterministic read of THIS incident (category, signals). Drives the topical
    # relevance gate so historical evidence is specific to the incident at hand.
    profile: Optional[taxonomy.Profile] = None


# Raw retained content uses `key=value` markers, e.g.
# [HISTORICAL INCIDENT #13 RESOLVED] service=Payment API occurred_at=... outcome=success
_INCIDENT_ID_RES = (
    re.compile(r"HISTORICAL\s+INCIDENT\s*#(\d+)", re.I),
    re.compile(r"\bINCIDENT\s*#(\d+)", re.I),
    re.compile(r"\bincident\s*#(\d+)", re.I),
)
_SERVICE_RE = re.compile(r"\bservice=(.+?)(?=\s+[a-z_]+=|\n|$)", re.I)
_ROOT_CAUSE_RE = re.compile(r"\broot_cause_family=(.+?)(?=\s+[a-z_]+=|\n|$)", re.I)
_OUTCOME_RE = re.compile(r"\boutcome=(success|fail)\b", re.I)
_STATUS_RE = re.compile(r"\bstatus=([a-z_]+)", re.I)
_OCCURRED_RE = re.compile(r"\b(?:occurred_at|timestamp)=(\S{6,})")
_STEP_LINE_RE = re.compile(r"^\s*\d+[.)]\s+(.+?)\s*$", re.M)
_RUNBOOK_BODY_RE = re.compile(
    r"(?:Ordered fix steps|Confirmed fix steps(?: \(engineer-verified\))?|"
    r"Attempted fix steps[^:\n]*|Fix steps)\s*:\s*\n(.*)$",
    re.S | re.I,
)


def _first_int(value) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _search_rx(rx: "re.Pattern[str]", text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = rx.search(text)
    if not m:
        return None
    value = m.group(1).strip()
    return value or None


def _clean_step(step: str) -> str:
    return re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", str(step)).strip()


def provenance_from_result(mr: memory.MemoryResult, memory_type: str) -> tuple[MemoryHit, Optional[str]]:
    """Build a MemoryHit from one recall result.

    Sources, in order of trust: Hindsight recall metadata (round-trips on
    retain), then the ORIGINAL retained content (raw chunk), then Hindsight's
    rephrased text. Returns (hit, parsed_status).
    """
    meta = mr.metadata or {}
    raw = mr.raw_text or ""
    text = mr.text or ""
    haystack = f"{raw}\n{text}"

    incident_id = None
    if memory_type == "episodic":
        incident_id = _first_int(meta.get("incident_id"))
        if incident_id is None:
            for rx in _INCIDENT_ID_RES:
                found = _search_rx(rx, haystack)
                if found:
                    incident_id = _first_int(found)
                    if incident_id is not None:
                        break

    service = (
        (meta.get("service") or "").strip()
        or _search_rx(_SERVICE_RE, raw)
        or _search_rx(_SERVICE_RE, text)
    )
    root_cause = (
        (meta.get("root_cause_family") or "").strip()
        or _search_rx(_ROOT_CAUSE_RE, raw)
        or _search_rx(_ROOT_CAUSE_RE, text)
    )
    outcome = (meta.get("outcome") or "").strip() or _search_rx(_OUTCOME_RE, haystack)
    status = (meta.get("status") or "").strip() or _search_rx(_STATUS_RE, raw)
    timestamp = (
        (meta.get("occurred_at") or "").strip()
        or _search_rx(_OCCURRED_RE, raw)
        or mr.occurred_start
    )

    snippet_source = raw or text
    snippet = snippet_source[:SNIPPET_LIMIT]
    if len(snippet_source) > SNIPPET_LIMIT:
        snippet += " …"

    hit = MemoryHit(
        memory_type=memory_type if memory_type in TIERS else "episodic",
        query="",  # filled by the caller
        snippet=snippet,
        memory_id=mr.memory_id,
        historical_incident_id=incident_id,
        service=(service or None),
        timestamp=(timestamp or None),
        root_cause_family=(root_cause or None),
        outcome=(outcome.lower() if outcome else None),
        relevance_score=mr.score,
    )
    return hit, (status.lower() if status else None)


def historical_verdict(
    hit: MemoryHit, status: Optional[str], ctx: IncidentContext, tier: str = "episodic",
) -> tuple[bool, str]:
    """Application-level provenance gate (§9): reject anything that is not a
    genuinely historical memory. Returns (is_historical, reason_if_not)."""
    incident_id = hit.historical_incident_id
    if incident_id is not None and incident_id == ctx.incident_id:
        return False, f"references the CURRENT incident (#{incident_id})"
    if status and status != "resolved":
        return False, f"incident #{incident_id} has status={status!r} (not resolved historical memory)"
    if incident_id is not None:
        record = store.get_incident(incident_id)
        if record is not None and record.get("status") != "resolved":
            return False, (
                f"incident #{incident_id} is {record.get('status')!r} in incident history "
                "(only resolved incidents are historical evidence)"
            )
        if record is None and status != "resolved":
            # Claims an incident that does not exist in incident history and has
            # no trusted `status=resolved` marker from our own retention code —
            # unverifiable provenance must not be presented as evidence.
            return False, (
                f"incident #{incident_id} does not exist in incident history "
                "(provenance cannot be verified, so it is not historical evidence)"
            )
    elif tier == "episodic" and not status:
        # Episodic memories are per-incident records: every memory written by
        # this app carries incident provenance (metadata or raw markers) and
        # every resolution marker carries status=resolved. A markerless episodic
        # hit (e.g. a stale unresolved *report* memory retained before the fix)
        # can be neither tied to a resolved incident nor verified at all, so it
        # is never presented as historical evidence. Semantic/procedural tiers
        # are cross-incident patterns/runbooks and legitimately carry no id.
        return False, (
            "episodic memory carries no verifiable incident provenance "
            "(no incident id and no resolved status marker)"
        )
    return True, ""


def _dedupe(accepted: list[tuple[MemoryHit, memory.MemoryResult]]) -> list[tuple[MemoryHit, memory.MemoryResult]]:
    """Hindsight returns raw facts AND derived observations for the same
    incident — keep one hit per historical incident / memory. Identical
    content retained twice (two memory ids, same text) is one piece of
    evidence, not two."""
    seen: set = set()
    out: list[tuple[MemoryHit, memory.MemoryResult]] = []
    for hit, mr in accepted:
        if hit.historical_incident_id is not None:
            key = ("inc", hit.historical_incident_id)
        elif hit.memory_id:
            key = ("mem", hit.memory_id)
        else:
            key = ("txt", hit.snippet[:160])
        if key in seen:
            continue
        # Second pass: same content under a different memory id.
        content_key = ("content", re.sub(r"\s+", " ", (hit.snippet or "")).strip().lower())
        if content_key[1] and content_key in seen:
            continue
        seen.add(key)
        seen.add(content_key)
        out.append((hit, mr))
    return out


def relevance_floor(results: list[memory.MemoryResult]) -> float:
    """Score threshold for ONE search.

    ``min(absolute floor, best score in this search * relative ratio)`` — the
    absolute floor removes clear junk (observed ~0.001-0.17 for unrelated
    memories vs ~0.9+ for real matches), while the relative term keeps the
    filter safe if a deployment's scores live on a smaller scale. A result
    with no score at all (``None``) is never judged by this.
    """
    known = [r.score for r in results if r.score is not None]
    if not known:
        return settings.RELEVANCE_MIN_SCORE
    return min(settings.RELEVANCE_MIN_SCORE, max(known) * settings.RELEVANCE_RELATIVE_RATIO)


# ---------------------------------------------------------------------------
# Memory tier search (with provenance validation + honest tracing)
# ---------------------------------------------------------------------------

@dataclass
class TierSearch:
    output: str                       # tool result handed back to the model
    hits: list[MemoryHit] = field(default_factory=list)
    trace: Optional[MemoryTraceStep] = None
    errors: list[str] = field(default_factory=list)
    raw_texts: list[str] = field(default_factory=list)  # original retained content (runbook steps etc.)


def _payload_result(hit: MemoryHit, mr: memory.MemoryResult) -> dict:
    payload: dict = {"memory_type": hit.memory_type}
    if hit.historical_incident_id is not None:
        payload["historical_incident_id"] = hit.historical_incident_id
    for key in ("service", "timestamp", "root_cause_family", "outcome"):
        value = getattr(hit, key)
        if value:
            payload[key if key != "timestamp" else "occurred_at"] = value
    if hit.relevance_score is not None:
        payload["relevance_score"] = hit.relevance_score
    payload["text"] = (mr.raw_text or mr.text or "")[:TOOL_TEXT_LIMIT]
    if mr.raw_text and mr.text and mr.raw_text.strip() != mr.text.strip():
        payload["hindsight_summary"] = mr.text[:400]
    return payload


_LOCAL_MAX_EXTRA = 3


def _local_supplement(
    tier: str, query: str, ctx: IncidentContext,
    accepted: list[tuple[MemoryHit, memory.MemoryResult]],
) -> list[tuple[MemoryHit, memory.MemoryResult]]:
    """Topical matches from the local SQLite history/layer store that Hindsight
    did not already return. Same provenance rules: RESOLVED incidents only, never
    the current incident."""
    profile = ctx.profile
    if profile is None:
        return []
    have_ids = {h.historical_incident_id for h, _ in accepted if h.historical_incident_id is not None}
    have_keys = {layers.layer_key_of(f"{mr.raw_text or ''} {mr.text or ''}") for _, mr in accepted}
    have_keys.discard(None)
    scored: list[tuple[float, MemoryHit, memory.MemoryResult]] = []

    if tier == "episodic":
        for rec in store.list_incidents():
            iid = rec["id"]
            if iid == ctx.incident_id or rec.get("status") != "resolved" or iid in have_ids:
                continue
            try:
                steps = json.loads(rec.get("fix_steps_json") or "[]")
            except (TypeError, ValueError):
                steps = []
            body = (
                f"{rec['service']}\n{rec['symptom']}\n{rec.get('root_cause_family') or ''}\n"
                + "\n".join(steps)
            )
            ok, reason, sim = taxonomy.topical_verdict(profile, body, rec.get("root_cause_family"))
            if not ok:
                continue
            snippet = (
                f"[HISTORICAL INCIDENT #{iid} RESOLVED] service={rec['service']} "
                f"status=resolved outcome={rec.get('outcome') or 'unknown'}\n"
                f"{rec['symptom'].strip()}\n"
                f"Confirmed root cause: {rec.get('root_cause_family')}"
            )
            hit = MemoryHit(
                memory_type="episodic", query=query, snippet=snippet[:SNIPPET_LIMIT],
                memory_id=f"local-episodic-{iid}", historical_incident_id=iid,
                service=rec["service"], timestamp=rec.get("timestamp"),
                root_cause_family=rec.get("root_cause_family"), outcome=rec.get("outcome"),
                relevance_score=round(min(1.0, sim), 3), source="local", topical_reason=reason,
            )
            scored.append((sim, hit, memory.MemoryResult(text=snippet, raw_text=snippet, memory_id=hit.memory_id)))
    else:
        for row in layers.search_local(tier, f"{ctx.service} {ctx.symptom}", limit=6, min_score=0.25):
            iid = row["incident_id"]
            if iid == ctx.incident_id or row["layer_key"] in have_keys:
                continue
            ok, reason, _ = taxonomy.topical_verdict(
                profile, f"{row['text']}", row["category"],
            )
            if not ok:
                continue
            hit = MemoryHit(
                memory_type=tier, query=query, snippet=row["text"][:SNIPPET_LIMIT],
                memory_id=row["memory_id"], service=row["service"],
                root_cause_family=row["category"], outcome=row["outcome"],
                relevance_score=row["score"], source="local", topical_reason=reason,
            )
            scored.append((row["score"], hit, memory.MemoryResult(
                text=row["text"], raw_text=row["text"], memory_id=row["memory_id"])))

    scored.sort(key=lambda t: -t[0])
    picked = scored[:_LOCAL_MAX_EXTRA]
    return _dedupe([(h, mr) for _, h, mr in picked])


async def search_tier(tier: str, query: str, ctx: IncidentContext) -> TierSearch:
    recall = {
        "episodic": memory.recall_episodic,
        "semantic": memory.recall_semantic,
        "procedural": memory.recall_procedural,
    }[tier]

    # Anchor the search on THIS incident: whatever the model typed, also carry the
    # incident's own category + strongest signals so recall cannot drift to the
    # same generic memories for every incident.
    profile = ctx.profile
    anchor = profile.query_terms() if profile else ""
    recall_query = f"{query} {anchor}".strip() if anchor and anchor.lower() not in query.lower() else query
    batch = await recall(recall_query)
    error = getattr(batch, "error", None)
    if error:
        # A failed search is NOT "0 memories" — surface it as an error.
        message = f"{tier} memory search failed: {error}"
        logger.warning(message)
        return TierSearch(
            output=json.dumps({"error": message, "historical_results": []}),
            hits=[],
            trace=MemoryTraceStep(tier=tier, query=query, status="failed", error=error),
            errors=[message],
        )

    results = list(batch)
    floor = relevance_floor(results)
    accepted: list[tuple[MemoryHit, memory.MemoryResult]] = []
    rejected = 0
    low_relevance = 0
    off_topic = 0
    for mr in results:
        hit, status = provenance_from_result(mr, tier)
        hit.query = query
        ok, reason = historical_verdict(hit, status, ctx, tier=tier)
        if not ok:
            rejected += 1
            logger.info(
                "provenance: rejected %s memory %s — %s", tier, hit.memory_id or "?", reason,
            )
            continue
        # Historically valid but topically unrelated (e.g. a connection-pool
        # runbook for a disk-full incident) — never presented as evidence.
        if hit.relevance_score is not None and hit.relevance_score < floor:
            low_relevance += 1
            logger.info(
                "relevance: dropped %s memory %s — score %.3f < floor %.3f",
                tier, hit.memory_id or "?", hit.relevance_score, floor,
            )
            continue
        # Topical gate: a resolved incident about a DIFFERENT kind of problem is
        # historically valid but is not evidence for this one (root of the
        # "same historical evidence for every incident" bug).
        if profile is not None:
            ok_topic, topic_reason, _sim = taxonomy.topical_verdict(
                profile, f"{mr.raw_text or ''}\n{mr.text or ''}", hit.root_cause_family,
            )
            if not ok_topic:
                off_topic += 1
                logger.info(
                    "topical: dropped %s memory %s — %s", tier, hit.memory_id or "?", topic_reason,
                )
                continue
            hit.topical_reason = topic_reason
        hit.source = "hindsight"
        accepted.append((hit, mr))

    accepted = _dedupe(accepted)

    # Local layer store: deterministic, topic-ranked supplement. Fills in what
    # Hindsight missed, never duplicates what it already returned.
    local_added = 0
    if profile is not None:
        for hit, mr in _local_supplement(tier, query, ctx, accepted):
            accepted.append((hit, mr))
            local_added += 1
    hits = [h for h, _ in accepted]

    dropped: list[str] = []
    if rejected:
        dropped.append(
            f"{rejected} rejected as not genuinely historical (current/unresolved incident)"
        )
    if low_relevance:
        dropped.append(
            f"{low_relevance} dropped as too weak a match (relevance < {floor:.2f})"
        )
    if off_topic:
        dropped.append(f"{off_topic} dropped as a different kind of problem")

    if hits:
        note = (
            "These results are HISTORICAL memory that existed before the current incident. "
            "The current incident is never included."
        )
        if dropped:
            note += " Also " + "; ".join(dropped) + "."
    elif dropped:
        note = "0 historical memories used for this query; " + "; ".join(dropped) + "."
    else:
        note = "0 relevant historical memories for this query."

    payload = {
        "query": query,
        "historical_results": [_payload_result(h, mr) for h, mr in accepted],
        "rejected_as_current_incident": rejected,
        "dropped_low_relevance": low_relevance,
        "dropped_off_topic": off_topic,
        "relevance_floor": round(floor, 3),
        "note": note,
    }
    return TierSearch(
        output=json.dumps(payload),
        hits=hits,
        trace=MemoryTraceStep(
            tier=tier,
            query=query,
            status="ok",
            returned=len(results),
            accepted=len(hits),
            rejected=rejected,
            low_relevance=low_relevance,
            off_topic=off_topic,
            local_added=local_added,
        ),
        raw_texts=[(mr.raw_text or mr.text or "") for _, mr in accepted],
    )


# ---------------------------------------------------------------------------
# submit_diagnosis validation + step coercion
# ---------------------------------------------------------------------------

def coerce_fix_steps(value) -> list[str]:
    """Normalize whatever the model sent into a clean ordered step list."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        pieces: list[str] = []
        for line in re.split(r"\n+", text):
            line = line.strip()
            if not line:
                continue
            # split inline numbered lists: "1. do X 2. do Y"
            parts = re.split(r"(?<=\w)\s+(?=\d{1,2}[.)]\s)", line)
            pieces.extend(p for p in (p.strip() for p in parts) if p)
        return [_clean_step(p) for p in pieces if _clean_step(p)]
    if isinstance(value, list):
        steps = [_clean_step(v) for v in value]
        return [s for s in steps if s]
    return []


def validate_submit(args: dict) -> tuple[bool, str]:
    """The submit_diagnosis contract. Returns (ok, corrective_instruction)."""
    if not str(args.get("diagnosis_text") or "").strip():
        return False, (
            "diagnosis_text must be a non-empty string containing the root-cause hypothesis, "
            "confidence and the evidence from the CURRENT incident."
        )
    steps = coerce_fix_steps(args.get("suggested_fix_steps"))
    if not steps:
        return False, (
            "suggested_fix_steps must be a non-empty array of concrete remediation/verification "
            "actions for the CURRENT incident (one action per item, ordered). If a procedural "
            "runbook was retrieved, adapt its steps; otherwise list safe diagnostic and "
            "remediation steps from the current evidence. An empty array is not acceptable."
        )
    if len(steps) == 1 and len(steps[0]) > 240:
        return False, (
            "suggested_fix_steps must be structured ordered actions, not one long paragraph. "
            "Split it into discrete steps."
        )
    return True, ""


def steps_from_runbook(raw_texts: list[str]) -> list[str]:
    """Last-resort fallback: extract ordered steps from retrieved procedural
    memory (the raw retained runbook content). Only ever derived from what
    Hindsight actually returned."""
    for raw in raw_texts:
        if not raw:
            continue
        body_match = _RUNBOOK_BODY_RE.search(raw)
        body = body_match.group(1) if body_match else raw
        found = [_clean_step(m) for m in _STEP_LINE_RE.findall(body)]
        found = [s for s in found if s]
        if len(found) >= 1:
            return found
    return []


def derive_memory_hit_type(claimed: Optional[str], hits: list[MemoryHit]) -> str:
    """memory_hit_type reflects ACTUAL accepted historical hits — never the
    model's claim alone (a self-hit must not inflate the hit metric)."""
    types = {h.memory_type for h in hits}
    if not types:
        return "none"
    if claimed in types:
        return claimed
    for tier in ("procedural", "episodic", "semantic"):
        if tier in types:
            return tier
    return "none"


# ---------------------------------------------------------------------------
# "Why this recommendation" evidence lines (structured, human-readable)
# ---------------------------------------------------------------------------

_SNIPPET_HEAD = 140        # cap on the plain-language detail per line
_QUERY_HEAD = 48           # cap on the echoed search query
_EVIDENCE_LABEL = {
    "episodic": "Similar past incident",
    "semantic": "Recurring pattern",
    "procedural": "Confirmed runbook",
}
# Retention markers (`[HISTORICAL INCIDENT #4 RESOLVED]`) and metadata runs
# (`service=x timestamp=y status=resolved`) are plumbing, not the story.
_MARKER_RE = re.compile(r"^\[[^\]]{0,80}\]\s*")
# Metadata runs can hold multi-word values (`service=Payment API occurred_at=…`),
# so strip a leading `key=` token up to the next `key=` on the same line.
_KEYVAL_LEAD_RE = re.compile(r"^[a-z_]+=.+?(?=\s+[a-z_][a-z0-9_]*=|$)")
_RUNBOOK_RE = re.compile(
    r"root_cause_family=(?P<family>\S+?)\s+success_rate=(?P<rate>[\d.]+)%"
    r".*?\(based on\s+(?P<attempts>\d+)\s+confirmed attempt",
    re.I | re.S,
)
_PATTERN_RE = re.compile(
    r"root_cause_family=(?P<family>\S+?)\s+confidence_pct=(?P<conf>[\d.]+)", re.I
)


def _clip(text, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _prose_lines(snippet: str) -> list[str]:
    """Memory text stripped of retention markers and metadata, so the line
    reads as a sentence instead of `[INCIDENT #4 REPORTED] service=... timestamp=...`.
    Section headers ("Reported symptoms (...):", "Ordered fix steps:") are
    dropped too — the line should say something, not introduce something."""
    lines: list[str] = []
    for raw_line in (snippet or "").splitlines():
        line = _MARKER_RE.sub("", raw_line.strip())
        while True:
            m = _KEYVAL_LEAD_RE.match(line)
            if not m:
                break
            line = line[m.end():].strip()
        if not line or line.startswith("(") or line.endswith(":"):
            continue
        lines.append(line)
    return lines


def _evidence_origin(hit: MemoryHit) -> str:
    """The 'where did this come from' segment, connector included, so the line
    never ends up with a dangling `— :`. Never invented — real hit fields only."""
    if hit.historical_incident_id is not None:
        origin = f"historical incident #{hit.historical_incident_id}"
        if hit.service:
            origin += f" ({hit.service})"
        if hit.outcome:
            origin += f", outcome={hit.outcome}"
            origin += ", fix worked" if hit.outcome == "success" else ", fix did not work"
        return f" — {origin}"
    family = hit.root_cause_family
    if not family:
        # Provenance didn't carry it; the memory text itself states it.
        m = _RUNBOOK_RE.search(hit.snippet or "") or _PATTERN_RE.search(hit.snippet or "")
        family = m.group("family") if m else None
    if family:
        return f" (root cause: {family.replace('_', ' ')})"
    if hit.service:
        return f" — {hit.service}"
    return ""


def _evidence_detail(hit: MemoryHit) -> str:
    """One short sentence of substance for the bullet. Runbooks and patterns
    report their own recorded strength; incidents quote their symptom line;
    marker-only memories fall back to what their provenance establishes."""
    text = hit.snippet or ""
    if hit.memory_type == "procedural":
        m = _RUNBOOK_RE.search(text)
        if m:
            rate, attempts = float(m.group("rate")), int(m.group("attempts"))
            attempts_word = "attempt" if attempts == 1 else "attempts"
            return _clip(
                f"{rate:g}% success over {attempts} confirmed {attempts_word}",
                _SNIPPET_HEAD,
            )
    if hit.memory_type == "semantic":
        m = _PATTERN_RE.search(text)
        if m:
            return _clip(
                f"seen in {float(m.group('conf')):g}% of recent incidents",
                _SNIPPET_HEAD,
            )
    prose = _prose_lines(text)
    if prose:
        return _clip(prose[0], _SNIPPET_HEAD)
    # Marker-only memory: the origin segment already states incident, service
    # and outcome — don't repeat it. Anything else, name what provenance knows.
    if hit.historical_incident_id is not None:
        return ""
    return _clip(f"past memory from {hit.service}" if hit.service else "", _SNIPPET_HEAD)


def build_evidence_text(
    hits: list[MemoryHit],
    current_service: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[str]:
    """Structured 'why this recommendation' lines for the UI. No chain-of-thought.

    Ranked strongest-match first (cross-service hits demoted) and capped, so an
    on-call engineer reads a few clear bullets rather than a raw retrieval dump.
    Every claim in a line comes from a real memory field.
    """
    if limit is None:
        limit = settings.EVIDENCE_MAX_LINES

    def _rank(h: MemoryHit):
        mismatch = bool(
            current_service and h.service
            and h.service.strip().lower() != current_service.strip().lower()
        )
        return (mismatch, h.relevance_score is None, -(h.relevance_score or 0.0))

    lines: list[str] = []
    for h in sorted(hits, key=_rank):
        label = _EVIDENCE_LABEL.get(h.memory_type, "Historical memory")
        query = _clip(h.query or "this incident's symptoms", _QUERY_HEAD)
        origin = _evidence_origin(h)
        detail = _evidence_detail(h)

        line = f"{label} for {query!r}"
        if origin:
            line += origin
        if detail:
            line += f": {detail}"

        notes: list[str] = []
        if current_service and h.service and h.service.strip().lower() != current_service.strip().lower():
            notes.append(f"different service ({h.service})")
        if h.relevance_score is not None:
            notes.append(f"relevance {h.relevance_score:.2f}")
        if notes:
            line += " · " + " · ".join(notes)

        lines.append(line)
        if limit and len(lines) >= limit:
            break
    return lines



# ---------------------------------------------------------------------------
# Hallucination guard: keep the diagnosis text honest about history
# ---------------------------------------------------------------------------

_INC_REF_RE = re.compile(r"incident\s*#\s*(\d+)", re.I)
_HISTORY_CLAIM_RE = re.compile(
    r"similar (?:past |previous |earlier )?(?:incident|case|pattern)|historical (?:incident|pattern|memory|evidence)|"
    r"(?:seen|happened|occurred|resolved) (?:this |it )?(?:before|previously)|past incidents?|previous incidents?|"
    r"earlier incidents?|matches? (?:a |an )?(?:known|past|previous)",
    re.I,
)
_NO_HISTORY_RE = re.compile(
    r"\bno (?:matching |relevant |similar |usable )?(?:historical|past|previous|prior)|cold[- ]start|"
    r"not (?:found|available) in (?:historical )?memory|0 (?:relevant )?(?:historical )?(?:memories|results)|"
    r"no (?:relevant )?(?:memory|memories|precedent)",
    re.I,
)
COLD_START_NOTE = (
    "Note: no relevant resolved incident was found in memory for this problem type, so this is a "
    "cold-start hypothesis built from the current evidence only."
)


def ground_diagnosis_text(text: str, hits: list[MemoryHit], ctx: IncidentContext) -> str:
    """Remove claims the evidence cannot support.

    * sentences citing an incident id that is not among the ACCEPTED historical
      hits (or that cite the CURRENT incident as history) are dropped;
    * if no historical hit was accepted but the text still leans on 'similar past
      incidents', a cold-start note is appended so the engineer is not misled.
    """
    if not text:
        return text
    allowed = {h.historical_incident_id for h in hits if h.historical_incident_id is not None}
    kept: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        ids = {int(i) for i in _INC_REF_RE.findall(sent)}
        ungrounded = {i for i in ids if i not in allowed and i != ctx.incident_id}
        current_as_history = (
            ctx.incident_id in ids and bool(_HISTORY_CLAIM_RE.search(sent))
        )
        if ungrounded or current_as_history:
            logger.info("grounding: dropped sentence citing %s", sorted(ungrounded or ids))
            continue
        kept.append(sent)
    out = " ".join(kept).strip()
    if not out:
        out = "Diagnosis based on the current incident evidence."
    if not hits and _HISTORY_CLAIM_RE.search(out) and not _NO_HISTORY_RE.search(out):
        out = f"{out} {COLD_START_NOTE}"
    return out

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _finalize(
    ctx: IncidentContext,
    *,
    diagnosis_text: str,
    raw_steps,
    claimed_type: Optional[str],
    confidence_pct,
    root_cause_family_guess,
    hits: list[MemoryHit],
    trace: list[MemoryTraceStep],
    errors: list[str],
    runbook_texts: list[str],
    model_used: str,
) -> DiagnosisResult:
    profile = ctx.profile or taxonomy.build_profile(ctx.service, ctx.symptom)
    recommendations: list[Recommendation] = playbooks.build_recommendations(profile, ctx.service, hits)
    steps = coerce_fix_steps(raw_steps)
    if not steps:
        steps = steps_from_runbook(runbook_texts)
        if steps:
            logger.warning(
                "submit_diagnosis had no usable suggested_fix_steps; using %d step(s) "
                "retrieved from procedural memory", len(steps),
            )
    if not steps:
        # Last resort: category playbook — specific to the incident type, never empty.
        steps = playbooks.steps_from_recommendations(recommendations)
    return DiagnosisResult(
        incident_id=ctx.incident_id,
        diagnosis_text=ground_diagnosis_text(diagnosis_text or "No diagnosis produced.", hits, ctx),
        memory_hit_type=derive_memory_hit_type(claimed_type, hits),
        confidence_pct=confidence_pct,
        root_cause_family_guess=root_cause_family_guess or profile.primary,
        suggested_fix_steps=steps,
        memory_hits=hits,
        model_used=model_used,
        memory_status="degraded" if errors else "ok",
        memory_errors=errors,
        memory_trace=trace,
        evidence=build_evidence_text(hits, current_service=ctx.service),
        incident_category=profile.primary,
        incident_category_label=taxonomy.label_for(profile.primary) if profile.primary else None,
        incident_signals=profile.signals,
        recommendations=recommendations,
    )


async def diagnose_incident(incident_id: int, service: str, symptom: str, timestamp: str) -> DiagnosisResult:
    profile = taxonomy.build_profile(service, symptom)
    ctx = IncidentContext(incident_id=incident_id, service=service, symptom=symptom,
                          timestamp=timestamp, profile=profile)
    if profile.primary:
        classification = (
            f"INCIDENT CLASSIFICATION (deterministic, from the current evidence)\n"
            f"Most likely category: {profile.primary} ({taxonomy.label_for(profile.primary)}), "
            f"classifier confidence {profile.confidence_pct:.0f}%\n"
            + (f"Also possible: {profile.secondary}\n" if profile.secondary else "")
            + (f"Signals found: {'; '.join(profile.signals)}\n" if profile.signals else "")
        )
    else:
        classification = (
            "INCIDENT CLASSIFICATION (deterministic): no known failure category matched "
            "confidently — treat as unclassified and be explicit about uncertainty.\n"
        )
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"CURRENT INCIDENT (happening now — NOT historical memory)\n"
                f"Incident #{incident_id}\nService: {service}\nReported at: {timestamp}\n"
                f"Current evidence — symptom / error trace:\n{symptom}\n\n"
                f"{classification}\n"
                f"Search historical memory for precedents, then call submit_diagnosis with a "
                f"root-cause hypothesis, confidence and non-empty suggested_fix_steps."
            ),
        },
    ]

    hits: list[MemoryHit] = []
    trace: list[MemoryTraceStep] = []
    errors: list[str] = []
    runbook_texts: list[str] = []
    model_used = "unknown"
    submit_rejections = 0
    nudge_count = 0

    for _ in range(MAX_TURNS):
        result = await achat_completion(messages, tools=TOOLS, tool_choice="auto")
        if result.error:
            # LLM failure is reported through diagnosis_text/model_used — it is
            # NOT a memory failure, so memory_status/memory_errors stay honest.
            logger.error("Groq call failed during diagnosis of incident #%s: %s", incident_id, result.error)
            return _finalize(
                ctx,
                diagnosis_text=(
                    "Automated diagnosis is temporarily unavailable (LLM error after fallback). "
                    "Please triage manually; this incident has still been recorded."
                ),
                raw_steps=None,
                claimed_type=None,
                confidence_pct=None,
                root_cause_family_guess=None,
                hits=hits,
                trace=trace,
                errors=errors,
                runbook_texts=runbook_texts,
                model_used="none (error)",
            )

        model_used = result.model_used or "unknown"
        message = result.message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            # The model answered in prose instead of calling submit_diagnosis.
            # Nudge it once back to the structured tool, then accept what we get.
            if nudge_count < 1:
                nudge_count += 1
                messages.append({
                    "role": "user",
                    "content": (
                        "You must deliver the final answer by calling submit_diagnosis exactly once "
                        "with diagnosis_text, root_cause_family_guess, confidence_pct, "
                        "primary_memory_type_used and a NON-EMPTY suggested_fix_steps array."
                    ),
                })
                continue
            # Still honour the fix-step contract on the free-text fallback.
            return _finalize(
                ctx,
                diagnosis_text=message.content or "No diagnosis produced.",
                raw_steps=None,
                claimed_type=None,
                confidence_pct=None,
                root_cause_family_guess=None,
                hits=hits,
                trace=trace,
                errors=errors,
                runbook_texts=runbook_texts,
                model_used=model_used,
            )

        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tool_calls
        ]})

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if tc.function.name == "submit_diagnosis":
                ok, corrective = validate_submit(args)
                if not ok and submit_rejections < MAX_SUBMIT_REJECTIONS:
                    submit_rejections += 1
                    logger.info("submit_diagnosis rejected (%d/%d): %s",
                                submit_rejections, MAX_SUBMIT_REJECTIONS, corrective)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({
                            "error": "submit_diagnosis rejected by validation",
                            "reason": corrective,
                            "instruction": "Call submit_diagnosis again with the corrected fields.",
                        }),
                    })
                    continue
                if not ok:
                    logger.error("submit_diagnosis still invalid after %d rejections; "
                                 "falling back to procedural runbook steps", submit_rejections)
                return _finalize(
                    ctx,
                    diagnosis_text=str(args.get("diagnosis_text") or message.content or ""),
                    raw_steps=args.get("suggested_fix_steps"),
                    claimed_type=args.get("primary_memory_type_used"),
                    confidence_pct=args.get("confidence_pct"),
                    root_cause_family_guess=args.get("root_cause_family_guess"),
                    hits=hits,
                    trace=trace,
                    errors=errors,
                    runbook_texts=runbook_texts,
                    model_used=model_used,
                )

            if tc.function.name not in ("search_episodic_memory", "search_semantic_memory",
                                        "search_procedural_memory"):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"error": f"unknown tool {tc.function.name}"}),
                })
                continue

            tier = tc.function.name.removeprefix("search_").removesuffix("_memory")
            query = str(args.get("query") or symptom)
            search = await search_tier(tier, query, ctx)
            hits.extend(search.hits)
            if search.trace:
                trace.append(search.trace)
            errors.extend(search.errors)
            if tier == "procedural":
                runbook_texts.extend(search.raw_texts)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": search.output})

    return _finalize(
        ctx,
        diagnosis_text="Diagnosis inconclusive after multiple memory lookups — please triage manually.",
        raw_steps=None,
        claimed_type=None,
        confidence_pct=None,
        root_cause_family_guess=None,
        hits=hits,
        trace=trace,
        errors=errors,
        runbook_texts=runbook_texts,
        model_used=model_used,
    )
