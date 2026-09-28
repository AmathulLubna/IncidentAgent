"""Rebuild the three memory layers (episodic / semantic / procedural) for every
RESOLVED incident already in SQLite.

Use after upgrading, or if layers look empty/identical:

    cd backend
    python scripts/rebuild_layers.py              # local layer store only (no network)
    python scripts/rebuild_layers.py --hindsight  # also retain semantic+procedural docs in Hindsight

Episodic docs already in Hindsight are left alone (Hindsight is retain-only); pass
--episodic to add the new, richer episodic document as well.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import layers, memory, store  # noqa: E402


async def main(push: bool, episodic: bool) -> None:
    store.init_db()
    done = 0
    for rec in store.list_incidents():
        if rec.get("status") != "resolved":
            continue
        steps = json.loads(rec.get("fix_steps_json") or "[]")
        docs = layers.record_resolution(
            incident_id=rec["id"], service=rec["service"], symptom=rec["symptom"],
            occurred_at=rec.get("timestamp") or "",
            root_cause_family=rec.get("root_cause_family") or "unknown", fix_steps=steps,
            outcome=rec.get("outcome") or "success",
            predicted_root_cause=rec.get("predicted_root_cause_family"),
        )
        done += 1
        if push:
            for d in docs:
                if d.layer == "episodic" and not episodic:
                    continue
                try:
                    if d.layer == "episodic":
                        await memory.retain_episodic_resolution(
                            rec["id"], rec["service"], rec.get("root_cause_family") or "unknown", steps,
                            rec.get("outcome") or "success", symptom=rec["symptom"],
                            occurred_at=rec.get("timestamp") or "",
                            predicted_root_cause=rec.get("predicted_root_cause_family"))
                    else:
                        await memory.retain_layer_doc(d)
                except Exception as e:  # noqa: BLE001
                    print(f"  #{rec['id']} {d.layer}: retain failed: {type(e).__name__}: {e}")
        print(f"#{rec['id']} {rec['service']}: " + ", ".join(d.layer for d in docs))
    print(f"\nRebuilt layers for {done} resolved incident(s). Local counts: {store.layer_counts()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hindsight", action="store_true", help="also retain semantic/procedural docs in Hindsight")
    ap.add_argument("--episodic", action="store_true", help="with --hindsight: also retain the new episodic doc")
    a = ap.parse_args()
    asyncio.run(main(a.hindsight, a.episodic))
