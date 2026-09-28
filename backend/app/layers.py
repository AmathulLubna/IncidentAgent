"""
Memory layers — three DIFFERENT memories per resolved incident.

Each layer answers a different question, so each stores different content:

    EPISODIC    WHAT HAPPENED   the event: symptoms, signals, timeline, what was
                                tried, outcome (a case file).
    SEMANTIC    WHY IT HAPPENS  the causal knowledge: the signature that identifies
                                this kind of failure, the mechanism behind it, what
                                it is commonly confused with, when it applies.
    PROCEDURAL  HOW TO FIX IT   the runbook: trigger, ordered steps, verification,
                                rollback and track record.

Before this module all layers ended up holding (or being displayed as) the same
incident text. Now ``build_layer_docs()`` derives three distinct documents from
one resolution, every document is also mirrored into a local SQLite table so
layer search is deterministic and layer-specific even when Hindsight recall is
fuzzy, and ``search_local()`` ranks each layer by what that layer is FOR.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from . import playbooks, store, taxonomy

LAYERS = ("episodic", "semantic", "procedural")

LAYER_INFO: dict[str, dict] = {
    "episodic": {
        "label": "Episodic",
        "function": "WHAT HAPPENED",
        "purpose": "Case file of one resolved incident: the symptoms, signals, what was tried and how it ended.",
        "fields": ["Service", "When", "Symptoms", "Signals observed", "Root cause (confirmed)", "Actions taken", "Outcome"],
    },
    "semantic": {
        "label": "Semantic",
        "function": "WHY IT HAPPENS",
        "purpose": "Causal knowledge: how to recognise this kind of failure, the mechanism behind it and what it is confused with.",
        "fields": ["Signature", "Mechanism", "Confused with", "Applies when", "Evidence"],
    },
    "procedural": {
        "label": "Procedural",
        "function": "HOW TO FIX IT",
        "purpose": "Reusable runbook: when to use it, ordered steps, how to verify, how to roll back and its track record.",
        "fields": ["Trigger", "Steps", "Verify", "Rollback", "Track record"],
    },
}


@dataclass
class LayerDoc:
    layer: str
    incident_id: int
    category: str
    service: str
    outcome: str
    title: str
    content: str            # full retained document
    gist: str               # one-line summary for cards/evidence
    fields: dict            # structured view for the UI

    @property
    def layer_key(self) -> str:
        return f"{self.layer}:{self.incident_id}"


def _clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _numbered(steps: list[str]) -> str:
    return "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps)) or "  (none recorded)"


def _family_evidence(category: str) -> tuple[int, int]:
    """(resolved incidents in this category, successes) from local history."""
    total = ok = 0
    for rec in store.list_incidents():
        if rec.get("status") != "resolved":
            continue
        fam = taxonomy.normalize_family(rec.get("root_cause_family"))
        if fam == category:
            total += 1
            ok += 1 if rec.get("outcome") == "success" else 0
    return total, ok


def build_layer_docs(
    *, incident_id: int, service: str, symptom: str, occurred_at: str,
    root_cause_family: str, fix_steps: list[str], outcome: str,
    predicted_root_cause: Optional[str] = None,
) -> list[LayerDoc]:
    """One resolved incident -> three different documents (procedural only for
    a confirmed SUCCESS: a failed fix is a lesson, not a runbook)."""
    category = taxonomy.category_of_text(f"{service}\n{symptom}", root_cause_family) \
        or taxonomy.normalize_family(root_cause_family) or "unclassified"
    label = taxonomy.label_for(category)
    pb = playbooks.playbook_for(category)
    signals = taxonomy.extract_signals(symptom)
    sig_text = "; ".join(signals) if signals else _clip(symptom, 120)
    head_common = f"service={service} status=resolved outcome={outcome} root_cause_family={category}"
    docs: list[LayerDoc] = []

    # ---------------- EPISODIC: what happened ----------------
    steps_label = "Actions taken (engineer-confirmed, worked)" if outcome == "success" \
        else "Actions attempted (engineer-confirmed, did NOT resolve it)"
    ep_lines = [
        # marker kept identical to the historical format: agent.provenance_from_result parses it
        f"[HISTORICAL INCIDENT #{incident_id} RESOLVED] service={service} occurred_at={occurred_at} "
        f"status=resolved outcome={outcome} root_cause_family={root_cause_family} layer_key=episodic:{incident_id}",
        f"Gist: {service} — {_clip(symptom, 140)} → {label}; outcome {outcome.upper()}.",
        f"Service: {service}",
        f"When: {occurred_at}",
        f"Symptoms: {symptom.strip()}",
        f"Signals observed: {sig_text}",
        f"Root cause (confirmed): {root_cause_family}",
    ]
    if predicted_root_cause and taxonomy.normalize_family(predicted_root_cause) != category:
        ep_lines.append(f"Agent predicted (unconfirmed, differed): {predicted_root_cause}")
    ep_lines += [f"{steps_label}:", _numbered(fix_steps), f"Outcome: {outcome.upper()}"]
    docs.append(LayerDoc(
        "episodic", incident_id, category, service, outcome,
        title=f"Incident #{incident_id} · {service}",
        content="\n".join(ep_lines),
        gist=_clip(f"{service}: {symptom}", 160),
        fields={
            "Service": service, "When": occurred_at, "Symptoms": symptom.strip(),
            "Signals observed": sig_text, "Root cause (confirmed)": root_cause_family,
            "Actions taken": fix_steps, "Outcome": outcome.upper(),
        },
    ))

    # ---------------- SEMANTIC: why it happens ----------------
    confused = [taxonomy.label_for(c) for c in pb.confused_with] or ["no common look-alikes recorded"]
    total, ok = _family_evidence(category)
    evidence = f"Confirmed in incident #{incident_id} ({service}, outcome {outcome}); " \
               f"{max(total, 1)} resolved incident(s) in this category, {ok} fixed successfully."
    signature = f"{sig_text}. Typical indicators of {label.lower()}: {pb.applies_when}"
    if outcome != "success":
        evidence += " Negative lesson: the attempted fix did not work — treat it as ruled out, not proven."
    sem_lines = [
        f"[SEMANTIC INSIGHT #{incident_id}] {head_common} layer_key=semantic:{incident_id}",
        f"Gist: {label} — {_clip(pb.mechanism, 170)}",
        f"Signature: {signature}",
        f"Mechanism: {pb.mechanism}",
        f"Confused with: {', '.join(confused)}",
        f"Applies when: {pb.applies_when}",
        f"Evidence: {evidence}",
    ]
    docs.append(LayerDoc(
        "semantic", incident_id, category, service, outcome,
        title=f"Pattern · {label}",
        content="\n".join(sem_lines),
        gist=_clip(f"{label}: {pb.mechanism}", 170),
        fields={
            "Signature": signature, "Mechanism": pb.mechanism,
            "Confused with": ", ".join(confused), "Applies when": pb.applies_when,
            "Evidence": evidence,
        },
    ))

    # ---------------- PROCEDURAL: how to fix (success only) ----------------
    if outcome == "success" and fix_steps:
        attempts, rate = store.get_success_rate(root_cause_family)
        track = (f"{rate:.0%} success over {attempts} confirmed attempt(s) for this root cause"
                 if attempts else "first confirmed success")
        verify = "; ".join(pb.verify) or "Error rate and latency return to baseline."
        rollback = pb.rollback or "Revert the last change made in the steps above."
        trigger = f"{label}: {pb.applies_when}"
        proc_lines = [
            f"[PROCEDURAL RUNBOOK #{incident_id}] {head_common} layer_key=procedural:{incident_id}",
            f"Gist: To fix {label.lower()} on {service}: {_clip(fix_steps[0], 130)}",
            f"Trigger: {trigger}",
            "Ordered fix steps:",
            _numbered(fix_steps),
            f"Verify: {verify}",
            f"Rollback: {rollback}",
            f"Track record: {track}",
        ]
        docs.append(LayerDoc(
            "procedural", incident_id, category, service, outcome,
            title=f"Runbook · {label}",
            content="\n".join(proc_lines),
            gist=_clip(f"Fix {label.lower()}: " + " → ".join(_clip(s, 45) for s in fix_steps[:3]), 170),
            fields={
                "Trigger": trigger, "Steps": fix_steps, "Verify": verify,
                "Rollback": rollback, "Track record": track,
            },
        ))
    return docs


def parse_fields(layer: str, raw: str) -> dict:
    """Best-effort structured view of a retained layer document (used for
    Hindsight results, whose original text we wrote in this format)."""
    labels = LAYER_INFO.get(layer, {}).get("fields", [])
    out: dict = {}
    current: Optional[str] = None
    for line in (raw or "").splitlines():
        m = re.match(r"^\s*(Steps|Ordered fix steps|Actions (?:taken|attempted)[^:]*)\s*:\s*$", line)
        if m:
            current = "Steps" if layer == "procedural" else "Actions taken"
            out[current] = []
            continue
        if current and re.match(r"^\s*\d+[.)]\s+", line):
            out[current].append(re.sub(r"^\s*\d+[.)]\s+", "", line).strip())
            continue
        current = None if not re.match(r"^\s+\S", line) else current
        for label in labels:
            if line.startswith(label + ":"):
                out[label] = line[len(label) + 1:].strip()
    return out


def layer_key_of(text: str) -> Optional[str]:
    m = re.search(r"layer_key=([a-z]+:\d+)", text or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Persistence + local, layer-aware search
# ---------------------------------------------------------------------------

def save_docs(docs: list[LayerDoc]) -> None:
    for d in docs:
        store.upsert_layer(
            layer=d.layer, incident_id=d.incident_id, category=d.category, service=d.service,
            outcome=d.outcome, title=d.title, gist=d.gist, content=d.content,
            fields_json=json.dumps(d.fields),
        )


def record_resolution(**kwargs) -> list[LayerDoc]:
    docs = build_layer_docs(**kwargs)
    save_docs(docs)
    return docs


def backfill_local() -> int:
    """Create layer rows for resolved incidents that predate this feature.
    Local only (no network). Returns number of incidents backfilled."""
    n = 0
    existing = store.layer_incident_ids()
    for rec in store.list_incidents():
        if rec.get("status") != "resolved" or rec["id"] in existing:
            continue
        try:
            steps = json.loads(rec.get("fix_steps_json") or "[]")
        except (TypeError, ValueError):
            steps = []
        record_resolution(
            incident_id=rec["id"], service=rec["service"], symptom=rec["symptom"],
            occurred_at=rec.get("timestamp") or "",
            root_cause_family=rec.get("root_cause_family") or "unknown",
            fix_steps=steps, outcome=rec.get("outcome") or "success",
            predicted_root_cause=rec.get("predicted_root_cause_family"),
        )
        n += 1
    return n


def _row_score(layer: str, q_tokens_text: str, q_cats: set[str], row: dict) -> float:
    """Rank a stored layer row for a query by what THAT layer is for."""
    fields = json.loads(row.get("fields_json") or "{}")
    if layer == "episodic":
        body = " ".join([row["service"], str(fields.get("Symptoms", "")),
                         str(fields.get("Signals observed", "")), str(fields.get("Root cause (confirmed)", ""))])
    elif layer == "semantic":
        body = " ".join([str(fields.get("Signature", "")), str(fields.get("Mechanism", "")),
                         str(fields.get("Applies when", "")), taxonomy.label_for(row["category"])])
    else:
        body = " ".join([str(fields.get("Trigger", "")), " ".join(fields.get("Steps", []) or []),
                         taxonomy.label_for(row["category"])])
    sim = taxonomy.similarity(q_tokens_text, body)
    # short keyword queries: plain substring hits count too ("token refresh")
    qt = taxonomy.tokens(q_tokens_text)
    bt = taxonomy.tokens(body)
    kw = (len(qt & bt) / len(qt)) if qt else 0.0
    score = max(sim, 0.8 * kw)
    if row["category"] in q_cats:
        score += 0.35
    return round(min(score, 1.0), 4)


def search_local(layer: str, query: str, limit: int = 8, min_score: float = 0.2) -> list[dict]:
    cats = {c for c, s in taxonomy.classify(query) if s >= 2}
    normalized = taxonomy.normalize_family(query.strip().replace(" ", "_"))
    if normalized in taxonomy.CATEGORY_LABELS:
        cats.add(normalized)
    scored = []
    for row in store.list_layer_rows(layer):
        s = _row_score(layer, query, cats, row)
        if s >= min_score:
            scored.append((s, row))
    scored.sort(key=lambda t: (-t[0], -t[1]["incident_id"]))

    out, seen_gist = [], set()
    for score, row in scored:
        # semantic/procedural rows of the same category read identically —
        # keep the best-scoring one per (category, gist) so results stay distinct.
        f = json.loads(row.get("fields_json") or "{}")
        key = (row["layer"], row["incident_id"]) if layer == "episodic" else \
            (row["category"], json.dumps(f.get("Signature") or f.get("Steps"), sort_keys=True))
        if key in seen_gist:
            continue
        seen_gist.add(key)
        out.append({
            "layer": layer, "source": "local", "layer_key": f"{layer}:{row['incident_id']}",
            "incident_id": row["incident_id"], "category": row["category"],
            "category_label": taxonomy.label_for(row["category"]), "service": row["service"],
            "outcome": row["outcome"], "title": row["title"], "gist": row["gist"],
            "fields": json.loads(row.get("fields_json") or "{}"),
            "text": row["content"], "score": score, "memory_id": f"local-{layer}-{row['incident_id']}",
        })
        if len(out) >= limit:
            break
    return out
