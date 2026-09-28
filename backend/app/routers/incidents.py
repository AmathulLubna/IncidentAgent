import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from .. import consolidation, layers, memory, store
from .. import taxonomy
from ..agent import diagnose_incident
from ..schemas import (
    DiagnosisResult,
    IncidentIn,
    IncidentRecord,
    MemoryHit,
    MemoryTraceStep,
    Recommendation,
    ResolveIn,
)

logger = logging.getLogger("incidents")

router = APIRouter(prefix="/incidents", tags=["incidents"])


@router.post("", response_model=DiagnosisResult)
async def create_incident(incident: IncidentIn):
    ts = incident.timestamp or datetime.now(timezone.utc)
    incident_id = store.create_incident(incident.service, incident.symptom, ts)

    # NOTE: nothing is retained into Hindsight here — deliberately.
    # The current incident must NEVER be retrievable as "historical memory"
    # while it is being diagnosed. Episodic retention happens only after the
    # engineer confirms the resolution (see resolve_incident below).
    diagnosis = await diagnose_incident(
        incident_id, incident.service, incident.symptom, ts.isoformat()
    )

    store.save_diagnosis(incident_id, diagnosis)
    return diagnosis


@router.post("/{incident_id}/resolve")
async def resolve_incident(incident_id: int, resolution: ResolveIn):
    record = store.get_incident(incident_id)
    if not record:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Human confirmation is the source of truth: this overwrites the confirmed
    # columns only — the agent's prediction stays in predicted_root_cause_family.
    store.save_resolution(incident_id, resolution.root_cause_family, resolution.fix_steps, resolution.outcome)

    memory_status = "ok"
    errors: list[str] = []
    # Retention happens NOW: the incident is resolved, so it becomes genuine
    # historical experience (including FAILED outcomes, which are kept as
    # negative evidence but never promoted to a runbook).
    try:
        await memory.retain_episodic_resolution(
            incident_id,
            record["service"],
            resolution.root_cause_family,
            resolution.fix_steps,
            resolution.outcome,
            symptom=record.get("symptom") or "",
            occurred_at=record.get("timestamp") or "",
            predicted_root_cause=record.get("predicted_root_cause_family"),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("episodic resolution retain failed #%s: %s: %s", incident_id, type(e).__name__, e)
        memory_status = "degraded"
        errors.append(f"episodic retain failed: {type(e).__name__}: {e}")

    # Three DIFFERENT layer documents per resolution (what happened / why it
    # happens / how to fix it). Always mirrored locally; semantic + procedural are
    # also retained into their own Hindsight banks (episodic was retained above).
    layer_report: list[dict] = []
    try:
        docs = layers.record_resolution(
            incident_id=incident_id, service=record["service"], symptom=record.get("symptom") or "",
            occurred_at=record.get("timestamp") or "",
            root_cause_family=resolution.root_cause_family, fix_steps=resolution.fix_steps,
            outcome=resolution.outcome, predicted_root_cause=record.get("predicted_root_cause_family"),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("layer build failed #%s: %s: %s", incident_id, type(e).__name__, e)
        docs = []
        memory_status = "degraded"
        errors.append(f"layer build failed: {type(e).__name__}: {e}")
    for doc in docs:
        entry = {"layer": doc.layer, "gist": doc.gist, "category": doc.category, "hindsight": "ok"}
        if doc.layer != "episodic":
            try:
                await memory.retain_layer_doc(doc)
            except Exception as e:  # noqa: BLE001
                logger.warning("%s layer retain failed #%s: %s: %s", doc.layer, incident_id, type(e).__name__, e)
                memory_status = "degraded"
                entry["hindsight"] = f"error: {type(e).__name__}"
                errors.append(f"{doc.layer} retain failed: {type(e).__name__}: {e}")
        layer_report.append(entry)
    if resolution.outcome != "success":
        layer_report.append({"layer": "procedural", "gist": "not stored — a failed fix is a lesson (kept in "
                             "episodic + semantic), never a runbook", "hindsight": "skipped"})

    promotion = await consolidation.maybe_promote_procedural(
        resolution.root_cause_family, resolution.fix_steps, resolution.outcome)
    if isinstance(promotion, dict) and promotion.get("retained") is False:
        memory_status = "degraded"
        errors.append("procedural retain failed")

    consolidation_result = await consolidation.maybe_run_consolidation()

    attempts, success_rate = store.get_success_rate(resolution.root_cause_family)
    updated = store.get_incident(incident_id)
    return {
        "incident": _to_record(updated),
        "memory_status": memory_status,
        "errors": errors,
        "layers": layer_report,
        "promotion": promotion,
        "consolidation": consolidation_result,
        "success_rate": {"attempts": attempts, "success_rate": success_rate},
    }


@router.get("", response_model=list[IncidentRecord])
async def list_all_incidents():
    return [_to_record(i) for i in store.list_incidents()]


@router.get("/{incident_id}", response_model=IncidentRecord)
async def get_one_incident(incident_id: int):
    record = store.get_incident(incident_id)
    if not record:
        raise HTTPException(status_code=404, detail="Incident not found")
    return _to_record(record)


def _load_json(value, default):
    import json

    if not value:
        return default
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return default
    return parsed if isinstance(parsed, type(default)) else default


def _to_record(row: dict) -> IncidentRecord:
    return IncidentRecord(
        id=row["id"],
        service=row["service"],
        symptom=row["symptom"],
        timestamp=row["timestamp"],
        status=row["status"],
        diagnosis_text=row.get("diagnosis_text"),
        memory_hit_type=row.get("memory_hit_type") or "none",
        confidence_pct=row.get("confidence_pct"),
        # --- predicted (agent) vs confirmed (engineer) stay separate ---
        predicted_root_cause_family=row.get("predicted_root_cause_family"),
        root_cause_family=row.get("root_cause_family"),
        suggested_fix_steps=_load_json(row.get("suggested_fix_steps_json"), []),
        fix_steps=_load_json(row.get("fix_steps_json"), []),
        outcome=row.get("outcome"),
        # --- historical evidence actually used for this diagnosis ---
        memory_hits=[MemoryHit(**h) for h in _load_json(row.get("memory_hits_json"), [])],
        evidence=_load_json(row.get("evidence_json"), []),
        memory_trace=[MemoryTraceStep(**t) for t in _load_json(row.get("memory_trace_json"), [])],
        memory_status=row.get("memory_status") or "ok",
        memory_errors=_load_json(row.get("memory_errors_json"), []),
        incident_category=row.get("incident_category"),
        incident_category_label=(taxonomy.label_for(row.get("incident_category"))
                                 if row.get("incident_category") else None),
        incident_signals=_load_json(row.get("incident_signals_json"), []),
        recommendations=[Recommendation(**r) for r in _load_json(row.get("recommendations_json"), [])],
        model_used=row.get("model_used"),
        created_at=row["created_at"],
    )
