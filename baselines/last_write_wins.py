"""last_write_wins (Spec 13; paper Section VI "same-key update"): change only the memories with the same
entity + attribute as Δ (REVOKE for a removed fact, else REVISE)."""
from __future__ import annotations

from baselines.common import modify_action, run_baseline
from csmr.types import same_entity

NAME = "last_write_wins"


def decide(delta, obs, roots, store, cfg):
    return {m.id: modify_action(True, delta) for m in store.all()
            if same_entity(m.entity, delta.entity) and m.attribute == delta.attribute}, {}


def run(obs, store, cfg=None):
    return run_baseline(NAME, decide, obs, store, cfg)
