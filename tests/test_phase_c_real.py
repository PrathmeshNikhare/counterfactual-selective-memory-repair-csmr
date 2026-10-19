"""Regression guard: real-LLM Phases A-C on car_sale reproduce paper Table II. Run with `pytest -m llm`."""
import pytest

from csmr.adapters.memory import InMemoryAdapter
from csmr.config import load_config
from csmr.pipeline import dependency, evidence
from csmr.pipeline.actions import propose_action
from csmr.pipeline.candidates import build_candidates
from csmr.pipeline.extract import extract_state_change, find_roots
from csmr.scenario import load_scenario


@pytest.mark.llm
def test_car_sale_table_ii():
    sc = load_scenario("data/dev/car_sale.json")
    store = InMemoryAdapter(sc.memories)
    delta = extract_state_change(sc.observation, [m.attribute for m in sc.memories])
    roots = find_roots(delta, sc.observation, store)
    c = build_candidates(delta, sc.observation, roots, store)
    dependency.assess_dependency(c, store, delta, sc.observation)
    evidence.assess_evidence(c, store, delta, sc.observation)
    cfg = load_config()
    got = {m: propose_action(m, i, cfg, delta.change_type).value for m, i in c.items()}
    assert got == {"M1": "REVOKE", "M2": "REVISE", "M3": "VERIFY", "M4": "KEEP"}
    assert "M5" not in c
