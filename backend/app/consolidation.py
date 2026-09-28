"""
Semantic consolidation + procedural promotion.

1. Semantic consolidation — every CONSOLIDATION_EVERY_N resolved incidents,
   look at the recent window; for any family occurring >= MIN_OCCURRENCES,
   confidence = count / window * 100 (computed arithmetically, never by LLM).
   Groq only phrases the pattern sentence.
2. Procedural promotion — human-confirmed outcome == SUCCESS, plus
   attempts >= MIN_ATTEMPTS and success_rate >= MIN
   → retain runbook. FAILED fixes are never promoted; they stay episodic
   evidence only. Hindsight is retain-only; a family's runbook is retained
   again only when the confirmed-attempts count grows (store.runbook_state),
   so the same runbook is never appended twice.
"""
import logging

from . import memory, store
from .config import settings
from .groq_client import achat_completion, chat_completion

logger = logging.getLogger("consolidation")


async def maybe_run_consolidation():
    if store.resolved_since_last_consolidation() < settings.CONSOLIDATION_EVERY_N:
        return None
    return await run_consolidation()


async def run_consolidation():
    window = store.get_last_n_resolved(settings.CONSOLIDATION_EVERY_N)
    if not window:
        return None

    family_counts: dict[str, int] = {}
    for inc in window:
        fam = inc.get("root_cause_family")
        if fam:
            family_counts[fam] = family_counts.get(fam, 0) + 1

    written = []
    for fam, count in family_counts.items():
        if count < settings.CONSOLIDATION_MIN_OCCURRENCES:
            continue
        confidence_pct = round(count / len(window) * 100, 1)
        examples = [inc["symptom"] for inc in window if inc.get("root_cause_family") == fam][:3]

        pattern_text = await _summarize_pattern(fam, examples)
        try:
            await memory.retain_semantic_pattern(fam, pattern_text, confidence_pct, len(window))
        except Exception as e:  # noqa: BLE001
            logger.warning("semantic retain failed for %s: %s: %s", fam, type(e).__name__, e)
            continue
        written.append({"root_cause_family": fam, "confidence_pct": confidence_pct,
                        "pattern_text": pattern_text, "window_n": len(window),
                        "occurrences": count})

    last_id = window[-1]["id"]
    store.mark_consolidated(last_id)
    return written


async def _summarize_pattern(root_cause_family: str, example_symptoms: list[str]) -> str:
    prompt = (
        f"Across several recent incidents, the confirmed root cause was '{root_cause_family}'. "
        f"Here are example symptoms:\n" + "\n---\n".join(example_symptoms) + "\n\n"
        "In ONE sentence, describe the general pattern an on-call engineer should recognize "
        "(e.g. 'auth-service incidents spike after deploys touching token-refresh logic'). "
        "Do not mention counts or percentages, just the pattern itself."
    )
    result = await achat_completion([{"role": "user", "content": prompt}])
    if result.error or not result.message:
        return f"Recurring root cause: {root_cause_family}."
    return (result.message.content or f"Recurring root cause: {root_cause_family}.").strip()


async def maybe_promote_procedural(root_cause_family: str, fix_steps: list[str], outcome: str = "success"):
    """Promote a fix into procedural memory (a reusable runbook).

    Promotion requires a HUMAN-CONFIRMED successful resolution. A FAIL outcome
    is still retained as episodic evidence (so future diagnoses can learn what
    did not work), but it must never become a "successful runbook" — even when
    the family's aggregate success rate happens to clear the threshold.
    Returns None when nothing is promoted (not an error).
    """
    if outcome != "success":
        logger.info(
            "procedural promotion skipped for %s: outcome=%s (failed fixes are episodic "
            "evidence only, never a runbook)", root_cause_family, outcome,
        )
        return None

    attempts, success_rate = store.get_success_rate(root_cause_family)
    if attempts < settings.PROCEDURAL_MIN_ATTEMPTS or success_rate < settings.PROCEDURAL_MIN_SUCCESS_RATE:
        return None

    # Hindsight is retain-only: re-retaining the SAME evidence level only
    # appends a duplicate runbook (recall then returns near-identical runbooks
    # for every future incident). Retain only when there is new evidence —
    # a higher confirmed-attempts count.
    already_retained = store.get_retained_runbook_attempts(root_cause_family)
    if already_retained is not None and attempts <= already_retained:
        logger.info(
            "procedural retain skipped for %s: runbook already retained at %d confirmed "
            "attempt(s) and still %d now — nothing new to learn",
            root_cause_family, already_retained, attempts,
        )
        return None

    try:
        await memory.retain_procedural_runbook(root_cause_family, fix_steps, success_rate, attempts)
    except Exception as e:  # noqa: BLE001
        logger.warning("procedural retain failed for %s: %s: %s", root_cause_family, type(e).__name__, e)
        return {"root_cause_family": root_cause_family, "success_rate": success_rate,
                "attempts": attempts, "retained": False, "error": f"{type(e).__name__}"}
    store.mark_runbook_retained(root_cause_family, attempts)
    return {"root_cause_family": root_cause_family, "success_rate": success_rate,
            "attempts": attempts, "retained": True}
