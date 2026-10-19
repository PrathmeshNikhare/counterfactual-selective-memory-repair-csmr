"""Virtual view of the store under a repair set (Spec 8, Paper IV-D). Never mutates the store."""
from __future__ import annotations

import dataclasses

from csmr.adapters.base import MemoryStoreAdapter
from csmr.types import Action, Memory, Observation, RepairSet, StateChange


def observation_memory(delta: StateChange, obs: Observation, store: MemoryStoreAdapter) -> Memory:
    """The new observation as a memory (Spec 8): appended to every view and to the committed store."""
    return Memory(id=f"OBS{len(store.all(include_revoked=True)) + 1}", text=delta.statement or obs.text,
                  entity=delta.entity, attribute=delta.attribute, time=obs.time, source=obs.source)


def view(store: MemoryStoreAdapter, rs: RepairSet, revised: dict[str, str],
         conditions: dict[str, str], new_mem: Memory | None = None) -> list[Memory]:
    out = []
    for m in store.all():
        a = rs.actions.get(m.id, Action.KEEP)
        if a == Action.REVOKE:
            continue
        text = m.text
        if a == Action.REVISE:
            text = revised[m.id]
        elif a == Action.VERIFY:
            text = f"[unverified] {m.text}"
        elif a == Action.CONDITIONAL:
            text = f"{m.text} (only valid if: {conditions[m.id]})"
        out.append(dataclasses.replace(m, text=text))  # copy; history list is shared but never written
    return out + [dataclasses.replace(new_mem)] if new_mem else out
