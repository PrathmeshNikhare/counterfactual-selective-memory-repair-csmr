"""Baselines (Spec 13). Each module exposes run(obs, store, cfg=None) -> Trace, like csmr.pipeline.run.run_pipeline."""
from baselines import (append_only, dependency_only, evidence_only, graph_propagation, last_write_wins, llm_oneshot,
                       similarity_prop)

BASELINES = {m.NAME: m.run for m in (append_only, last_write_wins, similarity_prop, graph_propagation, dependency_only,
                                      evidence_only, llm_oneshot)}
