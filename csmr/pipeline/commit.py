"""Phase E: minimal versioned commit + validation (Spec 11; Paper IV-E)."""
from __future__ import annotations

import warnings

from csmr.adapters.base import MemoryStoreAdapter
from csmr.config import load_config
from csmr.pipeline.agent import answer
from csmr.pipeline.counterfactual import judge
from csmr.pipeline.overlay import render
from csmr.types import Action, Memory, Observation, StateChange

_STATUS = {Action.REVOKE: "revoked", Action.VERIFY: "pending_verify", Action.CONDITIONAL: "conditional"}


def commit(store: MemoryStoreAdapter, rs, revised: dict[str, str], conditions: dict[str, str],
           new_mem: Memory) -> list[dict]:
    """Apply rs to the store (every change pushes history, bumps version) and add the observation memory.
    Returns version transitions. Pending-verify queue = memories with status "pending_verify".
    A REVISE resets the memory to active and clears any condition."""
    if not rs.feasible:
        raise ValueError(f"cannot commit infeasible repair set {rs.name}: {rs.infeasible_reason}")
    transitions = []
    for mid, a in rs.actions.items():
        if a == Action.KEEP:
            continue
        before = store.get(mid)
        old_version, old_status = before.version, before.status
        revise = a == Action.REVISE
        m = store.update(mid, text=revised[mid] if revise else None, status="active" if revise else _STATUS[a],
                         condition=conditions[mid] if a == Action.CONDITIONAL else None, clear_condition=revise,
                         reason=f"{rs.name}:{a.value}")
        transitions.append({"id": mid, "action": a.value, "from_version": old_version, "to_version": m.version,
                            "from_status": old_status, "to_status": m.status})
    store.add(new_mem)
    transitions.append({"id": new_mem.id, "action": "ADD", "from_version": 0, "to_version": new_mem.version,
                        "from_status": None, "to_status": new_mem.status})
    return transitions


def validate(store: MemoryStoreAdapter, probes: list[dict], delta: StateChange, obs: Observation,
             expected_benefit: float, cfg: dict | None = None) -> dict:
    """Re-run the probes on the committed store; warn if benefit differs from the selection-time estimate.
    The judge gets each probe's pre-commit source_text, as during selection."""
    cfg = cfg or load_config()
    mems = [r for m in store.all() if (r := render(m))]  # same rendering as overlay.view()
    scores = [judge(answer(p["question"], mems, cfg=cfg), p["question"], delta, obs, cfg, p["source_text"])["score"]
              for p in probes]
    benefit = sum(scores) / len(scores) if scores else 0.0
    ok = abs(benefit - expected_benefit) <= cfg["validation_tol"]
    if not ok:
        warnings.warn(f"commit validation: benefit {benefit:.3f} != expected {expected_benefit:.3f}")
    return {"benefit": benefit, "expected": expected_benefit, "ok": ok}
