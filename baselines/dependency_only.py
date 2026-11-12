"""dependency_only (Spec 13): CSMR Phases A-C with the evidence stage swapped for its trivial default
(ablation.use_evidence=false, S = 1), then change every candidate with P_dep >= tau_d. No counterfactual stage."""
from __future__ import annotations

import copy

from baselines.common import modify_action, run_baseline
from csmr.pipeline.run import phase_bc

NAME = "dependency_only"


def decide(delta, obs, roots, store, cfg):
    cfg = copy.deepcopy(cfg)
    cfg["ablation"]["use_evidence"] = False  # ignore evidence (Spec 13), via the ablation default (CLAUDE.md rule 7)
    cands = phase_bc(delta, obs, roots, store, cfg)
    return {m: modify_action(c.is_root, delta) for m, c in cands.items() if c.p_dep >= cfg["tau_d"]}, cands


def run(obs, store, cfg=None):
    return run_baseline(NAME, decide, obs, store, cfg)
