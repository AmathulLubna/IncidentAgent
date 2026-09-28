"""
Feeds incidents through the LIVE backend (not directly into Hindsight) so that
episodic retain, semantic consolidation, and procedural promotion all happen
exactly as they would during the real demo. This also builds up the local
sqlite bookkeeping (success rates, accuracy metric) consistently.

Order is curated so that, with the default settings
(CONSOLIDATION_EVERY_N=6, CONSOLIDATION_MIN_OCCURRENCES=3,
PROCEDURAL_MIN_ATTEMPTS=2, PROCEDURAL_MIN_SUCCESS_RATE=0.66):

  - Incident #1 is from a family that won't repeat soon -> "no memory" beat
  - By incident #6 (resolved), `auth_token_expiry` has appeared >= 3 times
    in the window -> consolidation on incident #6's resolve writes a
    semantic pattern -> incident #7 (also auth_token_expiry) gets a
    semantic-memory hit with a confidence stat
  - Procedural promotion is NOT windowed (it's a running success rate per
    family), so once >= PROCEDURAL_MIN_ATTEMPTS confirmed `auth_token_expiry`
    fixes have happened -- which, with the curation below, tends to land
    around the same time as or shortly after the semantic reveal, not at a
    fixed incident number. If you want a cleaner "semantic THEN procedural"
    staging for a recorded demo, either raise PROCEDURAL_MIN_ATTEMPTS in
    .env, or just pick which live incident you narrate each beat on rather
    than relying on exact indices.

Run the backend first (`uvicorn app.main:app --reload` from backend/), then:

    python data/seed_memory.py
"""
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from generate_synthetic import generate  # noqa: E402

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
TARGET_PATTERN_FAMILY = "auth_token_expiry"
DATA_DIR = os.path.dirname(__file__)


def _load(path: str) -> list[dict]:
    full = os.path.join(DATA_DIR, path)
    if not os.path.exists(full):
        return []
    with open(full) as f:
        items = json.load(f)
    return [i for i in items if "_comment" not in i]


def _curate_order(pool: list[dict]) -> list[dict]:
    by_family: dict[str, list[dict]] = {}
    for item in pool:
        fam = item.get("known_root_cause_family", "unknown")
        by_family.setdefault(fam, []).append(item)

    ordered: list[dict] = []

    # 1) Incident #1: pick a single incident from a family that is NOT the target pattern.
    other_families = [f for f in by_family if f != TARGET_PATTERN_FAMILY and by_family[f]]
    if other_families:
        first_family = other_families[0]
        ordered.append(by_family[first_family].pop(0))

    # 2) Ensure >=3 target-family incidents land inside the first 6 slots.
    target_pool = by_family.get(TARGET_PATTERN_FAMILY, [])
    for _ in range(min(3, len(target_pool))):
        ordered.append(target_pool.pop(0))

    # 3) Fill up to slot 6 with a mix of other families.
    filler_families = [f for f in other_families if by_family.get(f)]
    fi = 0
    while len(ordered) < 6 and any(by_family.get(f) for f in filler_families):
        fam = filler_families[fi % len(filler_families)]
        if by_family.get(fam):
            ordered.append(by_family[fam].pop(0))
        fi += 1

    # 4) Slot 7+: one more target-family incident -> should trigger the semantic-hit demo beat.
    if target_pool:
        ordered.append(target_pool.pop(0))

    # 5) A couple more target-family incidents before slot 12, to feed procedural promotion.
    for _ in range(min(2, len(target_pool))):
        ordered.append(target_pool.pop(0))

    # 6) Dump everything else in arbitrary order.
    remaining = target_pool[:]
    for fam, items in by_family.items():
        if fam == TARGET_PATTERN_FAMILY:
            continue
        remaining.extend(items)
    ordered.extend(remaining)

    return ordered


def run(delay_seconds: float = 0.0, resolve_outcome_default: str = "success"):
    real = _load("real_seed_incidents.json")
    postmortem = _load("postmortem_seed_incidents.json")
    synthetic_path = os.path.join(DATA_DIR, "synthetic_incidents.json")
    if os.path.exists(synthetic_path):
        synthetic = _load("synthetic_incidents.json")
    else:
        print("No synthetic_incidents.json found — generating in-memory (won't hit disk).")
        synthetic = generate(per_family=4)

    pool = real + postmortem + synthetic
    if not pool:
        print("No seed incidents found. Fill in real_seed_incidents.json / postmortem_seed_incidents.json "
              "or run generate_synthetic.py first.")
        return

    ordered = _curate_order(pool)
    print(f"Seeding {len(ordered)} incidents through {BACKEND_URL} ...\n")

    for idx, item in enumerate(ordered, start=1):
        payload = {
            "service": item["service"],
            "symptom": item["symptom"],
            "timestamp": item.get("timestamp"),
        }
        resp = requests.post(f"{BACKEND_URL}/incidents", json=payload, timeout=60)
        resp.raise_for_status()
        diagnosis = resp.json()
        print(f"[{idx}] #{diagnosis['incident_id']} service={item['service']} "
              f"memory_hit={diagnosis['memory_hit_type']} "
              f"confidence={diagnosis.get('confidence_pct')}")

        resolve_payload = {
            "root_cause_family": item.get("known_root_cause_family", diagnosis.get("root_cause_family_guess") or "unknown"),
            "fix_steps": item.get("known_fix_steps") or diagnosis.get("suggested_fix_steps") or ["Manual triage"],
            "outcome": resolve_outcome_default,
        }
        r2 = requests.post(f"{BACKEND_URL}/incidents/{diagnosis['incident_id']}/resolve", json=resolve_payload, timeout=60)
        r2.raise_for_status()

        if delay_seconds:
            time.sleep(delay_seconds)

    print("\nDone. Check GET /stats for bank sizes + accuracy metric.")


if __name__ == "__main__":
    run()
