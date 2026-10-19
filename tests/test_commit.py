import pytest

from csmr.adapters.memory import InMemoryAdapter
from csmr.pipeline import commit as C
from csmr.pipeline.overlay import observation_memory
from csmr.scenario import load_scenario
from csmr.types import Action, Observation, RepairSet, StateChange


def _setup():
    store = InMemoryAdapter(load_scenario("data/dev/car_sale.json").memories)
    delta = StateChange("user", "commute", "car", "none", "2025-01-01", "user", 0.9, statement="The user sold their car.")
    obs = Observation("I sold my car", "user", "2025-01-01")
    return store, delta, obs


def test_commit_versions_history_and_pending():
    store, delta, obs = _setup()
    orig = store.get("M1").text
    n = len(store.all(include_revoked=True))
    rs = RepairSet("R2", {"M1": Action.REVOKE, "M2": Action.REVISE, "M3": Action.VERIFY, "M4": Action.CONDITIONAL})
    tr = C.commit(store, rs, {"M2": "new text"}, {"M4": "owns a car"}, observation_memory(delta, obs, store))
    m1 = store.get("M1")
    assert m1.version == 2 and m1.status == "revoked" and m1.history[0]["text"] == orig
    assert store.get("M2").text == "new text" and store.get("M2").version == 2
    assert store.get("M3").status == "pending_verify" and store.get("M4").condition == "owns a car"
    assert store.get("M5").version == 1
    assert len(store.all(include_revoked=True)) == n + 1 and len(tr) == 5


def test_validate_warns_on_mismatch(monkeypatch):
    store, delta, obs = _setup()
    monkeypatch.setattr(C, "answer", lambda q, m, **k: "unknown")
    monkeypatch.setattr(C, "judge", lambda *a: {"score": 1.0})
    cfg = {"validation_tol": 0.01}
    assert C.validate(store, ["q"], delta, obs, 1.0, cfg)["ok"]
    with pytest.warns(UserWarning):
        assert not C.validate(store, ["q"], delta, obs, 0.5, cfg)["ok"]


def test_committed_render_matches_overlay_view():
    from csmr.pipeline.overlay import render, view
    store, delta, obs = _setup()
    rs = RepairSet("R2", {"M1": Action.REVOKE, "M2": Action.REVISE, "M3": Action.VERIFY, "M4": Action.CONDITIONAL})
    nm = observation_memory(delta, obs, store)
    virtual = [(m.id, m.text) for m in view(store, rs, {"M2": "new"}, {"M4": "owns a car"}, nm)]
    C.commit(store, rs, {"M2": "new"}, {"M4": "owns a car"}, nm)
    assert [(r.id, r.text) for m in store.all(include_revoked=True) if (r := render(m))] == virtual
    assert any(t.startswith("[unverified] ") for _, t in virtual) and any("(only valid if: " in t for _, t in virtual)


@pytest.mark.llm
def test_car_phases_a_to_e_real_llm():
    from csmr.config import load_config
    from csmr.pipeline import counterfactual as cf, dependency, evidence
    from csmr.pipeline.actions import propose_action
    from csmr.pipeline.candidates import build_candidates
    from csmr.pipeline.extract import extract_state_change, find_roots
    from csmr.pipeline.repair_sets import generate_repair_sets
    cfg = load_config()
    sc = load_scenario("data/dev/car_sale.json")
    store = InMemoryAdapter(sc.memories)
    delta = extract_state_change(sc.observation, [m.attribute for m in sc.memories])
    c = build_candidates(delta, sc.observation, find_roots(delta, sc.observation, store), store)
    dependency.assess_dependency(c, store, delta, sc.observation)
    evidence.assess_evidence(c, store, delta, sc.observation)
    prop = {m: propose_action(m, i, cfg, delta.change_type) for m, i in c.items()}
    sets = generate_repair_sets(c, prop, cfg)
    probes = cf.generate_probes(delta, sc.observation, c, store, cfg["n_probes"])
    revised, conds = cf.prepare_texts(sets, store, delta, sc.observation)
    nm = observation_memory(delta, sc.observation, store)
    best, _ = cf.select(sets, probes, store, revised, conds, c, delta, sc.observation, cfg, nm)
    C.commit(store, best, revised, conds, nm)
    out = C.validate(store, probes, delta, sc.observation, best.score["benefit"], cfg)
    print(best.name, out)
    assert out["ok"]
    assert store.get("M1").status == "revoked"
    assert store.get("M2").text == revised["M2"] and store.get("M2").version == 2
    assert store.get(nm.id).text == nm.text


def test_commit_rejects_infeasible_set():
    store, delta, obs = _setup()
    rs = RepairSet("R4", {"M3": Action.REVISE}, feasible=False, infeasible_reason="S < tau_e")
    with pytest.raises(ValueError, match="infeasible"):
        C.commit(store, rs, {"M3": "x"}, {}, observation_memory(delta, obs, store))
    assert store.get("M3").version == 1  # nothing applied


@pytest.mark.parametrize("prior", ["pending_verify", "conditional"])
def test_revise_resets_status_and_condition(prior):
    store, delta, obs = _setup()
    store.update("M2", status=prior, condition="only if x")
    rs = RepairSet("R", {"M2": Action.REVISE})
    C.commit(store, rs, {"M2": "new text"}, {}, observation_memory(delta, obs, store))
    m = store.get("M2")
    assert (m.status, m.condition, m.text) == ("active", None, "new text")
    assert m.history[-1]["status"] == prior and m.history[-1]["condition"] == "only if x"
