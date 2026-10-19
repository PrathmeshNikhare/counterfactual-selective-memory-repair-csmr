import pytest

from csmr import llm
from csmr.adapters.memory import InMemoryAdapter
from csmr.config import load_config
from csmr.pipeline import dependency
from csmr.pipeline.candidates import build_candidates
from csmr.scenario import load_scenario
from csmr.types import StateChange


def _car_delta(sc):
    return StateChange("user", "vehicle_ownership", "owns a car", "no car", sc.observation.time, "user", 0.95,
                       "The user sold their car.")


def _run(monkeypatch=None):
    sc = load_scenario("data/dev/car_sale.json")
    store, d = InMemoryAdapter(sc.memories), _car_delta(sc)
    c = build_candidates(d, sc.observation, ["M1"], store)
    dependency.assess_dependency(c, store, d, sc.observation)
    return c


def test_car_dependency_stubbed(monkeypatch):
    # stub the causal-cue LLM: only the car-driving / fuel memories depend on owning a car
    monkeypatch.setattr(llm, "call", lambda p, **k: {"answer": "yes" if ("drives" in p or "fuel" in p) else "no"})
    c, tau = _run(), load_config()["tau_d"]
    assert c["M1"].p_dep == 1.0
    assert c["M2"].p_dep >= tau and c["M4"].p_dep < tau
    assert "M5" not in c


@pytest.mark.llm
def test_car_dependency_real_llm():
    c, tau = _run(), load_config()["tau_d"]
    assert c["M2"].p_dep >= tau and c["M4"].p_dep < tau


def test_fit_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency, "_MODEL_FILE", tmp_path / "dependency.json")
    x1 = dict.fromkeys(dependency.FEATURES, 1.0)
    x0 = dict.fromkeys(dependency.FEATURES, 0.0)
    dependency.fit_dependency([x1, x0, x1, x0], [1, 0, 1, 0])
    assert dependency.dependency_prob(x1) > 0.5 > dependency.dependency_prob(x0)
