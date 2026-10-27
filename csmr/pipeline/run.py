"""Orchestrator: chains Phases A-E for one observation (Spec 12; Paper IV)."""
from __future__ import annotations

import time
from dataclasses import asdict

from csmr import llm
from csmr.adapters.base import MemoryStoreAdapter
from csmr.config import load_config
from csmr.pipeline import counterfactual as cf
from csmr.pipeline import dependency, evidence
from csmr.pipeline.actions import propose_action
from csmr.pipeline.candidates import build_candidates
from csmr.pipeline.commit import commit, validate
from csmr.pipeline.extract import extract, find_roots
from csmr.pipeline.overlay import observation_memory
from csmr.pipeline.repair_sets import generate_repair_sets
from csmr.types import Action, Observation, Trace


def run_pipeline(obs: Observation, store: MemoryStoreAdapter, cfg: dict | None = None) -> Trace:
    """Phase A -> E. Mutates `store` only in Phase E; returns early (store untouched) if there is no state change."""
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    llm.reset_stats()
    trace = Trace(observation=asdict(obs))

    def finish() -> Trace:
        trace.llm_stats = llm.stats()
        trace.wall_time_s = time.perf_counter() - t0
        return trace

    # A
    related = [m.text for m in store.search(obs.text, cfg["extract_context_k"])]
    delta, trace.note = extract(obs, [m.attribute for m in store.all()], cfg, related=related,
                                known_entities=[m.entity for m in store.all()])
    if delta is None:
        return finish()
    trace.delta = asdict(delta)
    trace.roots = find_roots(delta, obs, store, cfg)
    # B, C
    cands = build_candidates(delta, obs, trace.roots, store, cfg)
    dependency.assess_dependency(cands, store, delta, obs, cfg)
    evidence.assess_evidence(cands, store, delta, obs, cfg)
    for mid, info in cands.items():
        info.proposed = propose_action(mid, info, cfg, delta.change_type)
    trace.candidates = {m: {**asdict(i), "proposed": i.proposed.value} for m, i in cands.items()}
    # D
    sets = generate_repair_sets(cands, {m: i.proposed for m, i in cands.items()}, cfg)
    new_mem = observation_memory(delta, obs, store)
    use_cf = cfg["ablation"]["use_counterfactual"]  # off: no probes, select() applies the proposed actions
    trace.probes, trace.probe_gaps = cf.generate_probes(delta, obs, cands, store, cfg["n_probes"], cfg) if use_cf \
        else ([], {})
    revised, conditions = cf.prepare_texts(sets, store, delta, obs, cfg)
    best, trace.repair_sets = cf.select(sets, trace.probes, store, revised, conditions, cands, delta, obs, cfg, new_mem)
    trace.chosen = best.name
    trace.final_actions = {m.id: best.actions.get(m.id, Action.KEEP).value for m in store.all()}  # not in C => KEEP
    # E
    trace.transitions = commit(store, best, revised, conditions, new_mem)
    if use_cf:
        trace.validation = validate(store, trace.probes, delta, obs, best.score["benefit"], cfg)
    return finish()
