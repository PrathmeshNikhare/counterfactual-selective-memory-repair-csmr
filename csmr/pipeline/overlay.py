"""Virtual view of the store under a repair set (Spec 8, Paper IV-D). Never mutates the store."""
from __future__ import annotations

import dataclasses

from csmr.adapters.base import MemoryStoreAdapter
from csmr.types import Action, Memory, Observation, RepairSet, StateChange


def observation_memory(delta: StateChange, obs: Observation, store: MemoryStoreAdapter) -> Memory:
    """The new observation as a memory (Spec 8): appended to every view and to the committed store."""
    return Memory(id=f"OBS{len(store.all(include_revoked=True)) + 1}", text=delta.statement or obs.text,
                  entity=delta.entity, attribute=delta.attribute, time=obs.time, source=obs.source)


def render(m: Memory) -> Memory | None:
    """What the agent sees for a memory, from its status/condition (Spec 8). None if revoked.
    Shared by view() (virtual repair) and commit.validate() (committed store) so both render identically."""
    if m.status == "revoked":
        return None
    text = m.text
    if m.status == "pending_verify":
        text = f"[unverified] {m.text}"
    elif m.status == "conditional" and m.condition:
        text = f"{m.text} (only valid if: {m.condition})"
    return dataclasses.replace(m, text=text)  # copy; history list is shared but never written


_ACTION_STATUS = {Action.REVOKE: "revoked", Action.VERIFY: "pending_verify", Action.CONDITIONAL: "conditional"}


def view(store: MemoryStoreAdapter, rs: RepairSet, revised: dict[str, str],
         conditions: dict[str, str], new_mem: Memory | None = None) -> list[Memory]:
    out = []
    for m in store.all():
        a = rs.actions.get(m.id, Action.KEEP)
        virtual = dataclasses.replace(
            m, text=revised[m.id] if a == Action.REVISE else m.text, status=_ACTION_STATUS.get(a, m.status),
            condition=conditions[m.id] if a == Action.CONDITIONAL else m.condition)
        if (r := render(virtual)) is not None:
            out.append(r)
    return out + [render(new_mem)] if new_mem else out
