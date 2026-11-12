"""evidence_only (Spec 13): CSMR Phases A-C with the dependency stage swapped for its trivial default
(ablation.use_dependency=false, P_dep = 1, no causal-cue calls), then change every candidate with S >= tau_e.
No counterfactual stage."""
from __future__ import annotations

import copy

from baselines.common import modify_action, run_baseline
from csmr.pipeline.run import phase_bc

NAME = "evidence_only"


def decide(delta, obs, roots, store, cfg):
    cfg = copy.deepcopy(cfg)
    cfg["ablation"]["use_dependency"] = False  # ignore dependency (Spec 13), via the ablation default (CLAUDE.md rule 7)
    cands = phase_bc(delta, obs, roots, store, cfg)
    return {m: modify_action(c.is_root, delta) for m, c in cands.items() if c.s_evidence >= cfg["tau_e"]}, cands


def run(obs, store, cfg=None):
    return run_baseline(NAME, decide, obs, store, cfg)
