import pytest

from csmr import llm
from csmr.adapters.memory import InMemoryAdapter
from csmr.api import CSMR
from csmr.pipeline import commit as commit_mod
from csmr.pipeline import counterfactual as cf
from csmr.pipeline.agent import answer
from csmr.scenario import load_scenario
from tests.test_run import _fake_judge, _llm


def _api(monkeypatch):
    monkeypatch.setattr(llm, "call", _llm({"text": "The user previously drove to work by car; commute now unknown."}))
    for mod in (cf, commit_mod):
        monkeypatch.setattr(mod, "answer", lambda q, mems, **k: " ".join(m.text for m in mems))
        monkeypatch.setattr(mod, "judge", _fake_judge)
    return CSMR(InMemoryAdapter(load_scenario("data/dev/car_sale.json").memories))


def test_observe_retrieve_pending(monkeypatch):
    api = _api(monkeypatch)
    assert any("drives" in t for t in api.retrieve("how does the user get to work?", k=10))
    t = api.observe("I sold my car.", time="2026-06-02T09:00:00")
    assert t.chosen == "R2" and api.store.get("M1").status == "revoked"
    got = api.retrieve("car", k=10)
    assert len(got) == len(api.store.all()) and not any(x == api.store.get("M1").text for x in got)
    assert api.pending_verifications() == []  # M3 is REVISE'd (Spec 5), nothing left to verify
    api.store.update("M4", status="pending_verify")
    assert [m.id for m in api.pending_verifications()] == ["M4"]
    assert any(x.startswith("[unverified]") for x in api.retrieve("car", k=10))


def test_observe_defaults_time(monkeypatch):
    api = _api(monkeypatch)
    api.observe("I sold my car.")
    assert api.store.get("OBS6").time


@pytest.mark.llm
def test_chat_flow_gives_non_car_answer():
    api = CSMR(InMemoryAdapter(load_scenario("data/dev/car_sale.json").memories))
    api.observe("I sold my car.")
    out = answer("how do I get to work?", api.retrieve_memories("how do I get to work?"), cfg=api.cfg).lower()
    assert "drive" not in out and "car" not in out.replace("no car", "").replace("sold", "")


def test_non_question_without_state_change_is_stored_and_questions_are_ignored(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "call", lambda p, **k: calls.append(p) or {"state_change": None})
    api = CSMR(InMemoryAdapter(load_scenario("data/dev/car_sale.json").memories))
    t = api.observe("I own a car and drive to work every day", time="2026-06-02T09:00:00")
    m = api.store.get("OBS6")
    assert t.delta is None and t.chosen is None and len(calls) == 1
    assert (m.text, m.entity, m.attribute, m.time, m.status) == (
        "I own a car and drive to work every day", "user", "general", "2026-06-02T09:00:00", "active")
    assert [(x["id"], x["action"]) for x in t.transitions] == [("OBS6", "ADD")]
    assert all(api.store.get(f"M{i}").version == 1 for i in range(1, 6))
    q = api.observe("How do I get to work?")
    assert q.delta is None and q.transitions == [] and len(calls) == 1 and len(api.store.all()) == 6


@pytest.mark.llm
def test_three_message_demo_sequence_pinned_time():
    api = CSMR(InMemoryAdapter(load_scenario("data/dev/car_sale_distractors.json").memories))
    t1 = api.observe("I own a car and drive to work every day", time="2026-06-02T09:00:00")
    assert all(a == "KEEP" for a in t1.final_actions.values())  # confirmation: no repairs
    t2 = api.observe("I sold my car", time="2026-06-02T09:05:00")
    assert api.store.get("M1").status == "revoked" and t2.final_actions["M1"] == "REVOKE"
    q = "How do I get to work?"
    out = answer(q, api.retrieve_memories(q), cfg=api.cfg).lower()
    hedges = ("sold", "no longer", "previously", "unknown", "unverified", "unclear", "not sure", "uncertain", "no car")
    assert "car" not in out or any(h in out for h in hedges), out
