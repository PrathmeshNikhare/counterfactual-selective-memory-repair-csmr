"""similarity_prop (Spec 13; paper Section VI "broad semantic propagation"): REVISE every memory whose cosine
similarity to the observation text is >= cfg.sim_prop_threshold."""
from __future__ import annotations

from baselines.common import run_baseline
from csmr import models
from csmr.types import Action

NAME = "similarity_prop"


def decide(delta, obs, roots, store, cfg):
    mems = store.all()
    if not mems:
        return {}, {}
    sims = models.embed([m.text for m in mems]) @ models.embed([obs.text])[0]
    return {m.id: Action.REVISE for m, s in zip(mems, sims) if s >= cfg["sim_prop_threshold"]}, {}


def run(obs, store, cfg=None):
    return run_baseline(NAME, decide, obs, store, cfg)
