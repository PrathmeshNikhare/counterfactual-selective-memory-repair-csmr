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


def _cue(monkeypatch, fn):
    sc = load_scenario("data/dev/car_sale.json")
    store, d = InMemoryAdapter(sc.memories), _car_delta(sc)
    info = build_candidates(d, sc.observation, ["M1"], store)["M2"]
    monkeypatch.setattr(llm, "call", fn)
    llm.reset_stats()
    f = dependency.dependency_features(store.get("M1"), store.get("M2"), d, sc.observation, info)
    return f["causal_cue"]


@pytest.mark.parametrize("bad", [{}, {"answer": ""}, {"answer": "perhaps"}])
def test_malformed_cue_retries_once_then_no(monkeypatch, bad):
    prompts = []
    assert _cue(monkeypatch, lambda p, **k: prompts.append(p) or bad) == 0.0
    assert len(prompts) == 2 and prompts[0] != prompts[1]
    assert llm.stats()["fallbacks"] == {"causal_cue": 1}


def test_cue_retry_success_is_not_a_fallback(monkeypatch):
    outs = iter([{}, {"answer": "Yes"}])
    assert _cue(monkeypatch, lambda p, **k: next(outs)) == 1.0
    assert llm.stats()["fallbacks"] == {}


def test_cue_max_tokens_from_config(monkeypatch):
    seen = []
    _cue(monkeypatch, lambda p, **k: seen.append(k["max_tokens"]) or {"answer": "no"})
    assert seen == [load_config()["cue_max_tokens"]]


def test_nli_contra_topical_is_nli_times_cos(monkeypatch):
    from csmr import models
    monkeypatch.setattr(llm, "call", lambda p, **k: {"answer": "no"})
    monkeypatch.setattr(models, "nli", lambda a, b: {"entailment": 0.0, "contradiction": 0.8, "neutral": 0.2})
    sc = load_scenario("data/dev/car_sale.json")
    store, d = InMemoryAdapter(sc.memories), _car_delta(sc)
    info = build_candidates(d, sc.observation, ["M1"], store)["M2"]
    f = dependency.dependency_features(store.get("M1"), store.get("M2"), d, sc.observation, info)
    assert f["nli_contra_topical"] == pytest.approx(0.8 * f["cos_sim"]) and "nli_contra_new" not in f


def test_temporal_after_is_per_root_pair(monkeypatch):
    # an old memory is "after" an older root but not "after" a newer root (e.g. the stored message-1 note)
    monkeypatch.setattr(llm, "call", lambda p, **k: {"answer": "no"})
    sc = load_scenario("data/dev/car_sale_distractors.json")
    store, d = InMemoryAdapter(sc.memories), _car_delta(sc)
    info = build_candidates(d, sc.observation, ["M1"], store)["M2"]
    newer_root = store.get("M1").__class__(**{**store.get("M1").__dict__, "id": "OBS26", "time": "2026-06-02T09:00:00"})
    tf = lambda root, cand: dependency.dependency_features(root, cand, d, sc.observation, info)["temporal_after"]
    m6 = store.get("M6")  # 2026-03-02
    assert (tf(store.get("M1"), m6), tf(newer_root, m6)) == (1.0, 0.0)


@pytest.mark.parametrize("ans,cue", [("yes", 1.0), ("Maybe", 0.5), ("no", 0.0)])
def test_cue_is_graded_and_asks_about_the_change(monkeypatch, ans, cue):
    prompts = []
    assert _cue(monkeypatch, lambda p, **k: prompts.append(p) or {"answer": ans}) == cue
    assert "It changed: The user sold their car." in prompts[0] and "FALSE or OUT OF DATE" in prompts[0]
