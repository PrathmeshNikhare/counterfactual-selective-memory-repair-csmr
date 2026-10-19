"""Public API (Spec 12): wraps a store with the CSMR pipeline."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from csmr.adapters.base import MemoryStoreAdapter
from csmr.config import load_config
from csmr.pipeline.overlay import observation_memory, render
from csmr.pipeline.run import run_pipeline
from csmr.types import Memory, Observation, Trace


class CSMR:
    def __init__(self, store: MemoryStoreAdapter, config: str | dict = "config.yaml"):
        self.store = store
        self.cfg = config if isinstance(config, dict) else load_config(config)

    def observe(self, text: str, source: str = "user", time: str | None = None) -> Trace:
        """Phases A-E on one observation (Paper IV). Repair runs only if a state change is extracted; any other
        non-question message is still stored as a new memory. Questions are ignored (empty trace)."""
        obs = Observation(text, source, time or datetime.now().isoformat(timespec="seconds"))
        if text.strip().endswith("?"):
            return Trace(observation=asdict(obs))
        trace = run_pipeline(obs, self.store, self.cfg)
        if trace.delta is None:
            m = observation_memory(None, obs, self.store)
            self.store.add(m)
            trace.transitions.append({"id": m.id, "action": "ADD", "from_version": 0, "to_version": m.version,
                                      "from_status": None, "to_status": m.status})
        return trace

    def retrieve_memories(self, query: str, k: int = 5) -> list[Memory]:
        """Top-k memories as the agent should see them: revoked excluded, pending_verify/conditional tagged."""
        return [render(m) for m in self.store.search(query, k) if m.status != "revoked"]

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        return [m.text for m in self.retrieve_memories(query, k)]

    def pending_verifications(self) -> list[Memory]:
        return [m for m in self.store.all() if m.status == "pending_verify"]
