import pytest

from csmr import models
from csmr.adapters.memory import InMemoryAdapter
from csmr.config import load_config
from csmr.pipeline import evidence
from csmr.pipeline.candidates import build_candidates
from csmr.scenario import load_scenario
from csmr.types import StateChange


def _run(path, delta, roots):
    sc = load_scenario(path)
    store = InMemoryAdapter(sc.memories)
    c = build_candidates(delta, sc.observation, roots, store)
    evidence.assess_evidence(c, store, delta, sc.observation)
    return c


def _car():
    return _run("data/dev/car_sale.json",
                StateChange("user", "vehicle_ownership", "owns a car", "no car", "2026-06-01T11:00:00", "user", 0.95,
                            "The user sold their car."), ["M1"])


def test_car_evidence():
    c, tau = _car(), load_config()["tau_e"]
    assert c["M1"].s_evidence >= tau and c["M2"].s_evidence >= tau
    assert not c["M1"].hedged and not c["M1"].temporary


def test_bike_maybe_all_below_tau():
    d = StateChange("user", "vehicle_ownership", "owns a bike", "sold bike", "2026-06-05T10:00:00", "user", 0.3,
                    "The user might sell their bike next year.")
    c = _run("data/dev/bike_maybe.json", d, ["M1"])
    assert c and all(i.s_evidence < load_config()["tau_e"] for i in c.values())
    assert all(i.hedged for i in c.values())


def test_ablation_use_evidence_off(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(evidence, "load_config", lambda: {**cfg, "ablation": {**cfg["ablation"], "use_evidence": False}})
    assert all(i.s_evidence == 1.0 for i in _car().values())


def test_fit_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "_MODEL_FILE", tmp_path / "evidence.json")
    x1, x0 = dict.fromkeys(evidence.FEATURES, 1.0), dict.fromkeys(evidence.FEATURES, 0.0)
    evidence.fit_evidence([x1, x0, x1, x0], [1, 0, 1, 0])
    assert evidence.evidence_score(x1) > 0.5 > evidence.evidence_score(x0)


def test_dependent_contradiction_comes_from_root(monkeypatch):
    # contradiction is high only involving the root text: a dependent inherits NLI(root -> delta), not NLI(delta, itself)
    sc = load_scenario("data/dev/car_sale.json")
    root_text = next(m.text for m in sc.memories if m.id == "M1")
    monkeypatch.setattr(models, "nli", lambda a, b: {"contradiction": 0.9 if root_text in (a, b) else 0.1,
                                                     "entailment": 0.0, "neutral": 0.0})
    c = _car()
    cfg = load_config()
    assert c["M1"].ev_features["contradiction_strength"] == 0.9
    assert c["M2"].ev_features["contradiction_strength"] == 0.9
    assert c["M2"].ev_features["directness"] == cfg["directness"][c["M2"].hop] < c["M1"].ev_features["directness"]


def test_temporary_from_extractor_flag_or_regex():
    d = StateChange("user", "vehicle_ownership", "owns a car", "no car", "2026-06-01T11:00:00", "user", 0.95,
                    "The user sold their car.", temporary=True)
    c = _run("data/dev/car_sale.json", d, ["M1"])  # observation text has no temporary wording
    assert all(i.temporary for i in c.values()) and all(i.ev_features["not_temporary"] == 0.0 for i in c.values())
    assert not _car()["M1"].temporary
