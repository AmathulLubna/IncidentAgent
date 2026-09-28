from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

MemoryHitType = Literal["none", "episodic", "semantic", "procedural"]
Outcome = Literal["success", "fail"]


class IncidentIn(BaseModel):
    """What the on-call engineer submits when a new incident comes in."""
    service: str = Field(..., examples=["auth-service"])
    symptom: str = Field(..., description="Error trace / symptom description / logs")
    timestamp: Optional[datetime] = None
    # Only used by the seed script / synthetic generator to pre-label data;
    # the agent never sees this field when diagnosing.
    known_root_cause_family: Optional[str] = None


class MemoryHit(BaseModel):
    """One HISTORICAL memory retrieved while diagnosing the current incident.

    A MemoryHit must always refer to something that existed BEFORE the current
    diagnosis began (validated in application code — see
    ``agent.historical_verdict``). The current incident is never a MemoryHit.

    Provenance fields are only populated from what Hindsight actually returned
    (recall result id, recall metadata, raw chunk text, recall scores) or from
    SQLite incident history. Fields stay ``None`` when the source doesn't
    provide them — nothing here is invented.
    """
    memory_type: Literal["episodic", "semantic", "procedural"]
    query: str
    snippet: str
    # --- provenance (optional; only when Hindsight/SQLite actually provides it) ---
    memory_id: Optional[str] = None              # Hindsight recall result id
    historical_incident_id: Optional[int] = None  # incident id of the PAST incident
    service: Optional[str] = None
    timestamp: Optional[str] = None               # when the memory's incident/event occurred
    root_cause_family: Optional[str] = None
    outcome: Optional[str] = None                 # 'success' | 'fail' when known
    relevance_score: Optional[float] = None       # Hindsight recall scores.final
    source: Optional[str] = None                  # 'hindsight' | 'local' (SQLite layer store)
    topical_reason: Optional[str] = None          # why it was judged relevant to THIS incident


class CommandHint(BaseModel):
    """A read-only inspection command and what it tells the engineer."""
    label: str
    command: str


class ProvenFix(BaseModel):
    """A fix engineers CONFIRMED in an earlier, topically-matching resolved incident."""
    incident_id: int
    service: Optional[str] = None
    outcome: Optional[str] = None
    steps: list[str] = []


class Recommendation(BaseModel):
    """Developer-facing problem -> solution guidance for one incident category."""
    category: str
    category_label: str
    role: Literal["primary", "secondary"] = "primary"
    severity: Literal["critical", "high", "medium"] = "medium"
    title: str
    problem: str
    likely_causes: list[str] = []
    immediate_actions: list[str] = []
    investigate: list[CommandHint] = []
    permanent_fix: list[str] = []
    verify: list[str] = []
    prevent: list[str] = []
    rollback: Optional[str] = None
    caution: Optional[str] = None
    matched_signals: list[str] = []
    confidence_pct: float = 0.0
    proven_in_history: list[ProvenFix] = []


class MemoryTraceStep(BaseModel):
    """One memory-tier lookup in the diagnosis pipeline.

    Zero-result tiers are still recorded ("returned=0") so the UI can show the
    real pipeline, and failures are recorded as ``status='failed'`` with the
    actual error — never as an empty-but-successful search.
    """
    tier: Literal["episodic", "semantic", "procedural"]
    query: str
    status: Literal["ok", "failed"]
    returned: int = 0    # results Hindsight returned for this query
    accepted: int = 0    # results that passed historical-provenance validation
    rejected: int = 0    # results rejected (current incident / unresolved)
    low_relevance: int = 0  # historically valid, but too weak a match to count as evidence
    off_topic: int = 0      # historically valid, but about a DIFFERENT kind of problem
    local_added: int = 0    # extra topical matches supplied by the local layer store
    error: Optional[str] = None


class DiagnosisResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    incident_id: int
    diagnosis_text: str
    memory_hit_type: MemoryHitType
    confidence_pct: Optional[float] = None
    root_cause_family_guess: Optional[str] = None
    suggested_fix_steps: list[str] = []
    # HISTORICAL evidence only — never contains the current incident.
    memory_hits: list[MemoryHit] = []
    model_used: str
    # Operational transparency (never faked — reflects what actually happened).
    memory_status: str = "ok"
    memory_errors: list[str] = []
    memory_trace: list[MemoryTraceStep] = []
    evidence: list[str] = []
    # Deterministic read of the CURRENT incident + developer guidance.
    incident_category: Optional[str] = None
    incident_category_label: Optional[str] = None
    incident_signals: list[str] = []
    recommendations: list[Recommendation] = []


class ResolveIn(BaseModel):
    root_cause_family: str = Field(..., description="Confirmed/edited root cause family")
    fix_steps: list[str] = Field(..., description="Confirmed/edited ordered fix steps")
    outcome: Outcome


class IncidentRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: int
    service: str
    symptom: str
    timestamp: datetime
    status: Literal["diagnosed", "resolved"]
    diagnosis_text: Optional[str] = None
    memory_hit_type: MemoryHitType = "none"
    confidence_pct: Optional[float] = None
    # --- predicted vs confirmed are deliberately separate fields ---
    predicted_root_cause_family: Optional[str] = None   # agent's (unconfirmed) guess
    root_cause_family: Optional[str] = None             # engineer-confirmed root cause
    suggested_fix_steps: list[str] = []                 # agent recommendation
    fix_steps: list[str] = []                           # engineer-confirmed fix
    outcome: Optional[Outcome] = None
    # --- historical evidence actually used for this diagnosis ---
    memory_hits: list[MemoryHit] = []
    evidence: list[str] = []
    memory_trace: list[MemoryTraceStep] = []
    memory_status: str = "ok"
    memory_errors: list[str] = []
    incident_category: Optional[str] = None
    incident_category_label: Optional[str] = None
    incident_signals: list[str] = []
    recommendations: list[Recommendation] = []
    model_used: Optional[str] = None
    created_at: datetime


class StatsOut(BaseModel):
    total_incidents: int
    memory_hit_rate_last_5: float
    memory_hit_rate_series: list[float]
    episodic_count: int
    semantic_count: int
    procedural_count: int
    root_cause_family_counts: dict[str, int]
