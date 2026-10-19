"""Virtual view of the store under a repair set (Spec 8, Paper IV-D). Never mutates the store."""
from __future__ import annotations

import dataclasses

from csmr.adapters.base import MemoryStoreAdapter
from csmr.types import Action, Memory, RepairSet


def view(store: MemoryStoreAdapter, rs: RepairSet, revised: dict[str, str],
         conditions: dict[str, str]) -> list[Memory]:
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
    return out
