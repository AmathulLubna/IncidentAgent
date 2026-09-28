"""
Local SQLite bookkeeping.

Hindsight is the memory layer (episodic/semantic/procedural facts), but the
app still needs a plain source of truth for: incident history, confirmed
fix outcomes, per-family success rates, and the "rising accuracy" metric
shown in the UI. That bookkeeping lives here, deliberately separate from
Hindsight, so it's obvious in the code/demo which parts are "memory" vs
which parts are ordinary app state.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service TEXT NOT NULL,
    symptom TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'diagnosed',
    diagnosis_text TEXT,
    memory_hit_type TEXT NOT NULL DEFAULT 'none',
    confidence_pct REAL,
    predicted_root_cause_family TEXT,
    root_cause_family TEXT,
    suggested_fix_steps_json TEXT,
    fix_steps_json TEXT,
    outcome TEXT,
    model_used TEXT,
    memory_hits_json TEXT,
    evidence_json TEXT,
    memory_trace_json TEXT,
    memory_errors_json TEXT,
    memory_status TEXT NOT NULL DEFAULT 'ok',
    incident_category TEXT,
    incident_signals_json TEXT,
    recommendations_json TEXT,
    created_at TEXT NOT NULL
);

-- Local mirror of the three memory layers (one row per layer per resolved
-- incident). Each layer stores DIFFERENT content — see layers.py. Local search
-- over this table is deterministic, so layer results never collapse into the
-- same text even when Hindsight recall is fuzzy.
CREATE TABLE IF NOT EXISTS memory_layers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    layer TEXT NOT NULL,
    incident_id INTEGER NOT NULL,
    category TEXT NOT NULL,
    service TEXT NOT NULL,
    outcome TEXT NOT NULL,
    title TEXT NOT NULL,
    gist TEXT NOT NULL,
    content TEXT NOT NULL,
    fields_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (layer, incident_id)
);

CREATE TABLE IF NOT EXISTS consolidation_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    resolved_since_last_consolidation INTEGER NOT NULL DEFAULT 0,
    last_consolidated_incident_id INTEGER NOT NULL DEFAULT 0
);

-- Which runbook evidence level was already written to Hindsight per family.
-- Hindsight is retain-only (no update), so without this the same runbook gets
-- appended again on every resolution and recall returns near-duplicates.
CREATE TABLE IF NOT EXISTS runbook_state (
    family TEXT PRIMARY KEY,
    attempts_retained INTEGER NOT NULL,
    retained_at TEXT NOT NULL
);
"""

# Columns added after the first release; init_db() migrates existing DBs.
_MIGRATIONS: list[tuple[str, str]] = [
    ("predicted_root_cause_family", "TEXT"),
    ("suggested_fix_steps_json", "TEXT"),
    ("memory_hits_json", "TEXT"),
    ("evidence_json", "TEXT"),
    ("memory_trace_json", "TEXT"),
    ("memory_errors_json", "TEXT"),
    ("memory_status", "TEXT NOT NULL DEFAULT 'ok'"),
    ("incident_category", "TEXT"),
    ("incident_signals_json", "TEXT"),
    ("recommendations_json", "TEXT"),
]


@contextmanager
def get_conn():
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn):
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(incidents)")}
    added: list[str] = []
    for column, ddl in _MIGRATIONS:
        if column not in existing:
            conn.execute(f"ALTER TABLE incidents ADD COLUMN {column} {ddl}")
            added.append(column)
    if "predicted_root_cause_family" in added:
        # Older releases stored the agent's *prediction* in root_cause_family
        # until an engineer resolved the incident. Move unresolved predictions
        # into the dedicated column so predicted/confirmed stay distinct.
        conn.execute(
            "UPDATE incidents SET predicted_root_cause_family = root_cause_family "
            "WHERE status = 'diagnosed' AND root_cause_family IS NOT NULL"
        )
        conn.execute(
            "UPDATE incidents SET root_cause_family = NULL WHERE status = 'diagnosed'"
        )


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.execute(
            "INSERT OR IGNORE INTO consolidation_state (id, resolved_since_last_consolidation, "
            "last_consolidated_incident_id) VALUES (1, 0, 0)"
        )


def create_incident(service: str, symptom: str, timestamp: datetime) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO incidents (service, symptom, timestamp, created_at) VALUES (?, ?, ?, ?)",
            (service, symptom, timestamp.isoformat(), datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def save_diagnosis(incident_id: int, diagnosis) -> None:
    """Persist the agent's diagnosis.

    ``diagnosis`` is a ``schemas.DiagnosisResult``. The agent's root-cause
    guess goes into ``predicted_root_cause_family`` — ``root_cause_family`` is
    reserved for the ENGINEER-CONFIRMED value written by ``save_resolution``,
    so an AI prediction can never overwrite human truth.
    """
    from .schemas import DiagnosisResult

    if not isinstance(diagnosis, DiagnosisResult):  # pragma: no cover - defensive
        raise TypeError("save_diagnosis expects a DiagnosisResult")
    with get_conn() as conn:
        conn.execute(
            """UPDATE incidents SET diagnosis_text=?, memory_hit_type=?, confidence_pct=?,
               predicted_root_cause_family=?, suggested_fix_steps_json=?, model_used=?,
               memory_hits_json=?, evidence_json=?, memory_trace_json=?,
               memory_errors_json=?, memory_status=?, incident_category=?,
               incident_signals_json=?, recommendations_json=? WHERE id=?""",
            (
                diagnosis.diagnosis_text,
                diagnosis.memory_hit_type,
                diagnosis.confidence_pct,
                diagnosis.root_cause_family_guess,
                json.dumps(diagnosis.suggested_fix_steps),
                diagnosis.model_used,
                json.dumps([h.model_dump(exclude_none=True) for h in diagnosis.memory_hits]),
                json.dumps(diagnosis.evidence),
                json.dumps([t.model_dump(exclude_none=True) for t in diagnosis.memory_trace]),
                json.dumps(diagnosis.memory_errors),
                diagnosis.memory_status,
                diagnosis.incident_category,
                json.dumps(diagnosis.incident_signals),
                json.dumps([r.model_dump(exclude_none=True) for r in diagnosis.recommendations]),
                incident_id,
            ),
        )


def save_resolution(incident_id: int, root_cause_family: str, fix_steps: list[str], outcome: str):
    """Engineer-confirmed resolution. Overwrites the confirmed columns only —
    the agent's prediction (predicted_root_cause_family) is left intact."""
    with get_conn() as conn:
        conn.execute(
            """UPDATE incidents SET status='resolved', root_cause_family=?, fix_steps_json=?,
               outcome=? WHERE id=?""",
            (root_cause_family, json.dumps(fix_steps), outcome, incident_id),
        )
        conn.execute(
            "UPDATE consolidation_state SET resolved_since_last_consolidation = "
            "resolved_since_last_consolidation + 1 WHERE id=1"
        )


def get_incident(incident_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        return dict(row) if row else None


def list_incidents() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM incidents ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]


def get_resolved_incidents_for_family(root_cause_family: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM incidents WHERE status='resolved' AND root_cause_family=? ORDER BY id ASC",
            (root_cause_family,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_success_rate(root_cause_family: str) -> tuple[int, float]:
    """Returns (attempts, success_rate) for a given root-cause family."""
    rows = get_resolved_incidents_for_family(root_cause_family)
    if not rows:
        return 0, 0.0
    successes = sum(1 for r in rows if r["outcome"] == "success")
    return len(rows), successes / len(rows)


def resolved_since_last_consolidation() -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT resolved_since_last_consolidation FROM consolidation_state WHERE id=1"
        ).fetchone()
        return row["resolved_since_last_consolidation"] if row else 0


def get_last_n_resolved(n: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM incidents WHERE status='resolved' ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]


def mark_consolidated(last_incident_id: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE consolidation_state SET resolved_since_last_consolidation=0, "
            "last_consolidated_incident_id=? WHERE id=1",
            (last_incident_id,),
        )


def get_retained_runbook_attempts(root_cause_family: str) -> Optional[int]:
    """Confirmed-attempts count of the runbook already retained for a family
    (None when no runbook was retained yet)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT attempts_retained FROM runbook_state WHERE family=?",
            (root_cause_family,),
        ).fetchone()
        return int(row["attempts_retained"]) if row else None


def mark_runbook_retained(root_cause_family: str, attempts: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO runbook_state (family, attempts_retained, retained_at) VALUES (?, ?, ?) "
            "ON CONFLICT(family) DO UPDATE SET attempts_retained=excluded.attempts_retained, "
            "retained_at=excluded.retained_at",
            (root_cause_family, attempts, datetime.now(timezone.utc).isoformat()),
        )


def upsert_layer(*, layer: str, incident_id: int, category: str, service: str, outcome: str,
                 title: str, gist: str, content: str, fields_json: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO memory_layers (layer, incident_id, category, service, outcome, title, gist, "
            "content, fields_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(layer, incident_id) DO UPDATE SET category=excluded.category, "
            "service=excluded.service, outcome=excluded.outcome, title=excluded.title, "
            "gist=excluded.gist, content=excluded.content, fields_json=excluded.fields_json",
            (layer, incident_id, category, service, outcome, title, gist, content, fields_json,
             datetime.now(timezone.utc).isoformat()),
        )


def list_layer_rows(layer: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM memory_layers WHERE layer=? ORDER BY incident_id ASC", (layer,)
        ).fetchall()
        return [dict(r) for r in rows]


def layer_incident_ids() -> set[int]:
    with get_conn() as conn:
        return {r["incident_id"] for r in conn.execute("SELECT DISTINCT incident_id FROM memory_layers")}


def layer_counts() -> dict[str, int]:
    with get_conn() as conn:
        rows = conn.execute("SELECT layer, COUNT(*) AS n FROM memory_layers GROUP BY layer").fetchall()
        return {r["layer"]: r["n"] for r in rows}


def reset_demo():
    """Dev-only: clear incidents + consolidation counters. Hindsight untouched."""
    with get_conn() as conn:
        conn.execute("DELETE FROM memory_layers")
        conn.execute("DELETE FROM incidents")
        conn.execute("UPDATE consolidation_state SET resolved_since_last_consolidation=0, "
                     "last_consolidated_incident_id=0 WHERE id=1")


def get_stats() -> dict:
    incidents = list_incidents()
    total = len(incidents)
    # A "hit" only ever means: a genuinely HISTORICAL memory was retrieved.
    # The agent layer guarantees memory_hit_type is derived from validated
    # historical hits (never from the current incident referencing itself),
    # so a self-hit cannot inflate this metric.
    hits = [1 if i["memory_hit_type"] != "none" else 0 for i in incidents]

    # Rolling hit-rate series (window of 5) — this is the "rising accuracy" metric.
    series = []
    window = 5
    for idx in range(total):
        lo = max(0, idx - window + 1)
        chunk = hits[lo : idx + 1]
        series.append(round(sum(chunk) / len(chunk) * 100, 1))

    last_5 = series[-1] if series else 0.0

    family_counts: dict[str, int] = {}
    for i in incidents:
        fam = i["root_cause_family"]
        if fam:
            family_counts[fam] = family_counts.get(fam, 0) + 1

    return {
        "total_incidents": total,
        "memory_hit_rate_last_5": last_5,
        "memory_hit_rate_series": series,
        "root_cause_family_counts": family_counts,
    }
