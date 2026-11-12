"""graph_propagation (Spec 13; Paper VIII "fixed graph propagation"): on the same memory graph as Phase B hop
expansion (same entity and (same attribute or cosine >= hop_sim)), change every memory within cfg.graph_prop_hops
hops of the root(s): the root is REVOKEd if its fact is removed, everything else REVISEd. No dependency or evidence
model, no counterfactual stage."""
from __future__ import annotations

from baselines.common import modify_action, run_baseline
from csmr import models
from csmr.pipeline.candidates import hop_expansion

NAME = "graph_propagation"


def decide(delta, obs, roots, store, cfg):
    mems = {m.id: m for m in store.all()}
    if not mems or not roots:
        return {}, {}
    vecs = dict(zip(mems, models.embed([m.text for m in mems.values()])))
    reached = hop_expansion(roots, mems, vecs, cfg["graph_prop_hops"], cfg["hop_sim"])
    return {m: modify_action(h == 0, delta) for m, h in reached.items()}, {}


def run(obs, store, cfg=None):
    return run_baseline(NAME, decide, obs, store, cfg)
