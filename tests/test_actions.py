import pytest

from csmr import llm
from csmr.adapters.memory import InMemoryAdapter
from csmr.config import load_config
from csmr.pipeline import actions, dependency, evidence
from csmr.pipeline.candidates import build_candidates
from csmr.scenario import load_scenario
from csmr.types import Action, StateChange

CFG = load_config()


def _acts(path, delta, roots, monkeypatch, yes):
    monkeypatch.setattr(llm, "call", lambda p, **k: {"answer": "yes" if any(w in p for w in yes) else "no"})
    sc = load_scenario(path)
    store = InMemoryAdapter(sc.memories)
    c = build_candidates(delta, sc.observation, roots, store)
    dependency.assess_dependency(c, store, delta, sc.observation)
    evidence.assess_evidence(c, store, delta, sc.observation)
    return {m: actions.propose_action(m, i, CFG, delta.change_type) for m, i in c.items()}, c


def test_car_actions(monkeypatch):
    d = StateChange("user", "vehicle_ownership", "owns a car", "no car", "2026-06-01T11:00:00", "user", 0.95,
                    "The user sold their car.", "remove")
    a, _ = _acts("data/dev/car_sale.json", d, ["M1"], monkeypatch, ["drives", "fuel"])
    assert a["M1"] == Action.REVOKE and a["M2"] == Action.REVISE
    assert a["M3"] == Action.VERIFY and a["M4"] == Action.KEEP
    assert actions.propose_action("M5", None, CFG) == Action.KEEP


def test_phone_conditional(monkeypatch):
    d = StateChange("user", "phone", "iPhone 15", "dad's old Android", "2026-06-12T10:00:00", "user", 0.9,
                    "The user's iPhone is at the service centre for a week; they use an Android temporarily.")
    a, c = _acts("data/dev/phone_repair.json", d, ["M1"], monkeypatch, ["Apple Pay"])
    assert c["M1"].temporary and a["M1"] == Action.CONDITIONAL and a["M2"] == Action.CONDITIONAL


def test_text_helpers(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    d = StateChange("user", "vehicle_ownership", "car", "none", sc.observation.time, "user", 0.9, "The user sold their car.")
    monkeypatch.setattr(llm, "call", lambda p, **k: {"text": " T ", "condition": " C "})
    assert actions.revise_text(sc.memories[1], d, sc.observation) == "T"
    assert actions.condition_text(sc.memories[1], d, sc.observation) == "C"


@pytest.mark.parametrize("bad", [{}, {"text": ""}, {"other": "x"}])
def test_malformed_rewrites_retry_once_then_template(monkeypatch, bad):
    sc = load_scenario("data/dev/car_sale.json")
    d = StateChange("user", "vehicle_ownership", "car", "none", sc.observation.time, "user", 0.9, "The user sold their car.")
    prompts = []
    monkeypatch.setattr(llm, "call", lambda p, **k: prompts.append(p) or bad)
    llm.reset_stats()
    m = sc.memories[1]
    assert actions.revise_text(m, d, sc.observation) == f"Previously: {m.text.rstrip('. ')}. Possibly outdated after: The user sold their car."
    assert len(prompts) == 2 and prompts[0] != prompts[1]
    assert actions.condition_text(m, d, sc.observation) == "it is not affected by: The user sold their car."
    assert llm.stats()["fallbacks"] == {"revise_text": 1, "condition_text": 1}


def test_rewrite_retry_success_is_not_a_fallback(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    d = StateChange("user", "vehicle_ownership", "car", "none", sc.observation.time, "user", 0.9, "x")
    outs = iter([{}, {"text": "ok"}])
    monkeypatch.setattr(llm, "call", lambda p, **k: next(outs))
    llm.reset_stats()
    assert actions.revise_text(sc.memories[1], d, sc.observation) == "ok"
    assert llm.stats()["fallbacks"] == {}


def test_rewrite_max_tokens_from_config(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    d = StateChange("user", "vehicle_ownership", "car", "none", sc.observation.time, "user", 0.9, "x")
    seen = []
    monkeypatch.setattr(llm, "call", lambda p, **k: seen.append(k["max_tokens"]) or {"text": "t", "condition": "c"})
    actions.revise_text(sc.memories[1], d, sc.observation)
    actions.condition_text(sc.memories[1], d, sc.observation)
    assert seen == [CFG["rewrite_max_tokens"]] * 2
