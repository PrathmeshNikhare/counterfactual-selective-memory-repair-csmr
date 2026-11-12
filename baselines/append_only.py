"""append_only (Spec 13): store the observation as a new memory, change nothing. With the shared date-aware agent
(Spec 9) this is also the recency baseline."""
from __future__ import annotations

from baselines.common import run_baseline

NAME = "append_only"


def decide(delta, obs, roots, store, cfg):
    return {}, {}


def run(obs, store, cfg=None):
    return run_baseline(NAME, decide, obs, store, cfg)
