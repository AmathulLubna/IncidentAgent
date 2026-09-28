"""Inspect the stray SDK-probe memory in the live episodic bank.

Used for the final report: documents exactly what the probe left behind and
confirms whether it could ever surface as historical evidence.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import memory  # noqa: E402


async def main() -> None:
    client = memory.get_client()
    page = await client.alist_memories(bank_id=memory.EPISODIC_BANK, limit=200)
    items = getattr(page, "items", None) or getattr(page, "memories", None) or []
    hits = []
    for m in items:
        blob = json.dumps(m.model_dump() if hasattr(m, "model_dump") else str(m), default=str)
        if "999999" in blob or "probe" in blob.lower():
            hits.append(m)
    print(f"total episodic memories: {len(items)}")
    print(f"probe matches: {len(hits)}")
    for m in hits:
        print(json.dumps(m.model_dump() if hasattr(m, "model_dump") else str(m), indent=2, default=str))

    # Would it ever be recalled by a realistic incident query?
    batch = await memory.recall_episodic(
        "Order API HTTP 503 connection pool exhausted high latency", limit=5
    )
    print("recall error:", batch.error)
    for h in batch:
        print("-", getattr(h, "memory_id", None), "|", str(getattr(h, "text", ""))[:120])


if __name__ == "__main__":
    asyncio.run(main())
