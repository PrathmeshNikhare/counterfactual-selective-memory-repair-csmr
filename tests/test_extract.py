import pytest

from csmr import llm, models
from csmr.adapters.memory import InMemoryAdapter
from csmr.pipeline.extract import extract_state_change, find_roots
from csmr.scenario import load_scenario
from csmr.types import Observation, StateChange

CAR = {"state_change": {"entity": "user", "attribute": "vehicle_ownership", "old": "owns a car", "new": "no car",
                        "confidence": 0.95, "statement": "The user sold their car.", "change_type": "remove"}}


def test_extract_copies_time_source_and_none(monkeypatch):
    monkeypatch.setattr(llm, "call", lambda *a, **k: CAR)
    d = extract_state_change(Observation("I sold my car.", "tool", "2026-06-01"), ["vehicle_ownership"])
    assert (d.entity, d.attribute, d.time, d.source, d.confidence, d.change_type) == ("user", "vehicle_ownership", "2026-06-01", "tool", 0.95, "remove")
    monkeypatch.setattr(llm, "call", lambda *a, **k: {"state_change": None})
    assert extract_state_change(Observation("I like pizza."), []) is None


def test_roots_by_attribute_or_nli(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    monkeypatch.setattr(llm, "call", lambda *a, **k: CAR)
    d = extract_state_change(sc.observation, [])
    contra = lambda c: {"contradiction": c, "entailment": 0.0, "neutral": 1 - c}
    # nothing contradicts by NLI -> only the attribute match is a root
    monkeypatch.setattr(models, "nli_batch", lambda pairs: [contra(0.0)] * len(pairs))
    assert find_roots(d, sc.observation, InMemoryAdapter(sc.memories)) == ["M1"]
    # a high-NLI memory does not become a root when an attribute match exists
    monkeypatch.setattr(models, "nli_batch", lambda pairs: [contra(0.9 if "drives" in p[1] else 0.0) for p in pairs])
    assert find_roots(d, sc.observation, InMemoryAdapter(sc.memories)) == ["M1"]


def test_roots_nli_fallback_when_no_attribute_match(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    d = StateChange("user", "no_such_attribute", None, "no car", "", "user", 0.9, "The user sold their car.")
    contra = lambda c: {"contradiction": c, "entailment": 0.0, "neutral": 1 - c}
    monkeypatch.setattr(models, "nli_batch", lambda pairs: [contra(0.9 if "drives" in p[1] else 0.0) for p in pairs])
    assert find_roots(d, sc.observation, InMemoryAdapter(sc.memories)) == ["M2"]


@pytest.mark.llm
def test_car_sale_real():
    sc = load_scenario("data/dev/car_sale.json")
    store = InMemoryAdapter(sc.memories)
    d = extract_state_change(sc.observation, sorted({m.attribute for m in sc.memories}))
    assert d.entity == "user" and "vehicle" in d.attribute or "car" in d.attribute
    assert find_roots(d, sc.observation, store) == ["M1"]
    # TASKS B1: "I like pizza" -> None or no roots
    pz = Observation("I like pizza.", "user", "2026-06-01")
    d2 = extract_state_change(pz, sorted({m.attribute for m in sc.memories}))
    assert d2 is None or find_roots(d2, pz, store) == []


@pytest.mark.parametrize("bad", [{}, {"state_change": {"entity": "user"}}, {"state_change": {
    "entity": "user", "attribute": "a", "new": "x", "confidence": "high"}}])
def test_malformed_extraction_is_no_state_change(monkeypatch, bad):
    from csmr import llm
    from csmr.pipeline.extract import extract_state_change
    from csmr.types import Observation
    llm.reset_stats()
    monkeypatch.setattr(llm, "call", lambda p, **k: bad)
    assert extract_state_change(Observation("I sold my car", "user", "2025-01-01"), []) is None
    assert llm.stats()["fallbacks"] == {"extract": 1}


def test_undecodable_extraction_json_is_no_state_change(monkeypatch):
    import json
    from csmr import llm
    from csmr.pipeline.extract import extract_state_change
    from csmr.types import Observation

    def boom(p, **k):
        raise json.JSONDecodeError("bad", "", 0)

    llm.reset_stats()
    monkeypatch.setattr(llm, "call", boom)
    assert extract_state_change(Observation("x", "user", ""), []) is None
    assert llm.stats()["fallbacks"] == {"extract": 1}


def test_explicit_null_state_change_is_not_a_fallback(monkeypatch):
    from csmr import llm
    from csmr.pipeline.extract import extract_state_change
    from csmr.types import Observation
    llm.reset_stats()
    monkeypatch.setattr(llm, "call", lambda p, **k: {"state_change": None})
    assert extract_state_change(Observation("I like tea", "user", ""), []) is None
    assert llm.stats()["fallbacks"] == {}


def test_prompt_carries_related_memories_and_confirmation_rule(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "call", lambda p, **k: seen.append(p) or {"state_change": None})
    obs = Observation("I own a car and drive to work every day", "user", "2026-06-02T09:00:00")
    assert extract_state_change(obs, ["vehicle_ownership"], related=["The user owns a car.", "The user drives to work."]) is None
    assert "- The user owns a car.\n- The user drives to work." in seen[0]
    assert "repeats or confirms" in seen[0] and "yoga every Sunday" in seen[0]
    extract_state_change(obs, [])
    assert "(none)" in seen[1]


def test_action_and_judge_prompts_carry_date_only(monkeypatch):
    from csmr.config import load_config
    from csmr.pipeline import counterfactual as cf
    from csmr.pipeline.actions import revise_text
    from csmr.types import Memory
    seen = []
    monkeypatch.setattr(llm, "call", lambda p, **k: seen.append(p) or {"text": "t", "determined_by_change": True,
                                                                       "verdict": "consistent"})
    obs = Observation("I sold my car.", "user", "2026-06-02T09:41:07")
    d = StateChange("user", "vehicle_ownership", None, "no car", obs.time, "user", 0.9, "The user sold their car.")
    revise_text(Memory("M1", "The user owns a car.", "user", "vehicle_ownership", "2026-03-01"), d, obs)
    cf.judge("The user has no car.", "Does the user own a car?", d, obs, load_config())
    assert len(seen) == 2 and all("2026-06-02" in p and "09:41" not in p and "T09" not in p for p in seen)
