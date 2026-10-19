import copy

from csmr.adapters.memory import InMemoryAdapter
from csmr.pipeline.overlay import view
from csmr.scenario import load_scenario
from csmr.types import Action, RepairSet


def test_view_applies_actions_without_mutating_store():
    store = InMemoryAdapter(load_scenario("data/dev/car_sale.json").memories)
    before = copy.deepcopy(store.all(include_revoked=True))
    rs = RepairSet("R", {"M1": Action.REVOKE, "M2": Action.REVISE, "M3": Action.VERIFY, "M4": Action.CONDITIONAL})
    out = {m.id: m.text for m in view(store, rs, {"M2": "revised"}, {"M4": "cond"})}
    assert "M1" not in out
    assert out["M2"] == "revised"
    assert out["M3"] == "[unverified] " + store.get("M3").text
    assert out["M4"] == store.get("M4").text + " (only valid if: cond)"
    assert out["M5"] == store.get("M5").text
    assert store.all(include_revoked=True) == before


import pytest


@pytest.mark.llm
def test_agent_answers_from_memories_only():
    from csmr.pipeline.agent import answer
    ms = load_scenario("data/dev/car_sale.json").memories
    assert "unknown" in answer("What is the user's favourite colour?", ms[:1]).lower()


@pytest.mark.llm
def test_agent_commute_under_repair_vs_unchanged():
    from csmr.pipeline.agent import answer
    store = InMemoryAdapter(load_scenario("data/dev/car_sale.json").memories)
    q = "How does the user commute to work?"
    rs = RepairSet("R", {"M1": Action.REVOKE, "M2": Action.REVISE})
    revised = {"M2": "The user previously drove to work by car; current commute mode is unknown after selling the car."}
    r0 = answer(q, store.all())
    repaired = answer(q, view(store, rs, revised, {}))
    print(f"\nR0:       {r0}\nrepaired: {repaired}")
    # ponytail: keyword check, not a judge; fine for a smoke test
    assert any(w in repaired.lower() for w in ("unknown", "previously", "no longer", "sold", "not determin"))


def test_agent_retries_empty_then_unknown(monkeypatch):
    from csmr import llm
    from csmr.pipeline import agent
    calls = []
    monkeypatch.setattr(llm, "call", lambda p, **k: calls.append(p) or "")
    assert agent.answer("q?", []) == "unknown" and len(calls) == 2 and calls[0] != calls[1]
    calls.clear()
    monkeypatch.setattr(llm, "call", lambda p, **k: calls.append(p) or ("" if len(calls) == 1 else "Blue."))
    assert agent.answer("q?", []) == "Blue." and len(calls) == 2
