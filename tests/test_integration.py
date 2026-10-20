"""I1/I2: end-to-end invariants on the dev scenarios (real LLM; cached on disk, so reruns are free)."""
import copy

import pytest

from csmr.adapters.memory import InMemoryAdapter
from csmr.config import load_config
from csmr.pipeline.run import run_pipeline
from csmr.scenario import load_scenario

SCENARIOS = ["car_sale", "bike_maybe", "dog_false_similarity", "job_quit", "moved_city", "phone_repair"]


def _run(name, cfg=None):
    sc = load_scenario(f"data/dev/{name}.json")
    before = {m.id: copy.deepcopy(m) for m in sc.memories}
    store = InMemoryAdapter(sc.memories)
    return before, store, run_pipeline(sc.observation, store, cfg)


@pytest.fixture(scope="module")
def runs():
    return {n: _run(n) for n in SCENARIOS}


@pytest.mark.llm
@pytest.mark.parametrize("name", SCENARIOS)
def test_pipeline_invariants(runs, name):
    before, store, t = runs[name]
    tau_e = load_config()["tau_e"]
    assert t.delta and t.candidates and t.repair_sets and t.chosen

    # Paper V hard rules
    for mid, a in t.final_actions.items():
        assert a == "KEEP" or mid in t.candidates, f"{mid} {a} outside C"
        if a in ("REVISE", "REVOKE"):
            assert t.candidates[mid]["s_evidence"] >= tau_e, f"{mid} {a} with S < tau_e"

    # never delete; changed => version+1 with history, unchanged => untouched
    now = {m.id: m for m in store.all(include_revoked=True)}
    assert set(before) <= set(now)
    for mid, old in before.items():
        m, changed = now[mid], t.final_actions[mid] != "KEEP"
        assert m.version == old.version + changed and len(m.history) == len(old.history) + changed
        if changed:
            assert m.history[-1]["text"] == old.text and m.history[-1]["version"] == old.version

    assert t.validation["ok"] is True


@pytest.mark.llm
def test_bike_maybe_hedged_has_no_modifying_action(runs):
    assert not {"REVISE", "REVOKE"} & set(runs["bike_maybe"][2].final_actions.values())


def _sig(t):
    return (
        {m: (c["p_dep"], c["s_evidence"], c["proposed"]) for m, c in t.candidates.items()},
        [(r["name"], r["score"]) for r in t.repair_sets],
        t.chosen,
    )


@pytest.mark.llm
@pytest.mark.parametrize("flag", ["use_dependency", "use_evidence", "use_counterfactual", "use_minimality"])
def test_each_ablation_flag_changes_output(runs, flag):
    cfg = copy.deepcopy(load_config())
    cfg["ablation"][flag] = False
    assert _sig(_run("car_sale", cfg)[2]) != _sig(runs["car_sale"][2])
