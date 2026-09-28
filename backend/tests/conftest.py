"""Shared test setup: never touch the network from tests."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _no_network_layer_retain(monkeypatch):
    """Per-incident semantic/procedural layer retains go to Hindsight in
    production; in tests they are recorded instead (see test_improvements)."""
    from app import memory

    calls: list = []

    async def _record(doc):
        calls.append(doc)

    monkeypatch.setattr(memory, "retain_layer_doc", _record)
    return calls
