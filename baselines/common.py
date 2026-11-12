"""Shared harness for the baselines (Spec 13; Paper VIII "Baselines").

Every baseline gets exactly what CSMR gets: the same Phase A extraction (Δ, roots), the same REVISE / CONDITIONAL
rewrite helpers, the same versioned commit with the same appended observation memory, and returns a Trace in the
same format as run_pipeline, so eval/metrics.py scores it unchanged. Only `decide` (the repair rule) differs."""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Callable

from csmr import llm
from csmr.adapters.base import MemoryStoreAdapter
from csmr.config import load_config
from csmr.pipeline import counterfactual as cf
from csmr.pipeline.commit import commit
from csmr.pipeline.overlay import observation_memory
from csmr.pipeline.run import candidates_dict, phase_a
from csmr.types import Action, CandidateInfo, Observation, RepairSet, StateChange, Trace

# decide(delta, obs, roots, store, cfg) -> (actions for some memory ids, candidate set C or {} if the method has none)
Decide = Callable[[StateChange, Observation, list[str], MemoryStoreAdapter, dict],
                  tuple[dict[str, Action], dict[str, CandidateInfo]]]


def modify_action(is_root: bool, delta: StateChange) -> Action:
    """What a baseline does to a memory it decides to change: a removed root fact is REVOKEd, anything else REVISEd
    (the same root rule as CSMR's propose_action, Spec 6)."""
    return Action.REVOKE if is_root and delta.change_type == "remove" else Action.REVISE


def run_baseline(name: str, decide: Decide, obs: Observation, store: MemoryStoreAdapter,
                 cfg: dict | None = None) -> Trace:
    """Phase A -> decide -> commit. Like run_pipeline: mutates the store only when a state change is extracted."""
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    llm.reset_stats()
    trace = Trace(observation=asdict(obs))
    delta, trace.note, trace.roots = phase_a(obs, store, cfg)
    if delta is not None:
        trace.delta = asdict(delta)
        actions, cands = decide(delta, obs, trace.roots, store, cfg)
        trace.candidates = candidates_dict(cands)
        rs = RepairSet(name, {m: a for m, a in actions.items() if a != Action.KEEP})
        revised, conditions = cf.prepare_texts([rs], store, delta, obs, cfg)
        new_mem = observation_memory(delta, obs, store)
        trace.chosen = name
        trace.final_actions = {m.id: rs.actions.get(m.id, Action.KEEP).value for m in store.all()}
        trace.transitions = commit(store, rs, revised, conditions, new_mem)
    trace.llm_stats = llm.stats()
    trace.wall_time_s = time.perf_counter() - t0
    return trace
