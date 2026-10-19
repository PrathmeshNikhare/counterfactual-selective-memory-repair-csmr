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
