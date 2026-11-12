"""baselines/ (Spec 13), all stubbed: Phase A returns a fixed Δ, LLM rewrites/one-shot replies are canned,
embeddings are hand-set. Expected actions are hand-derived from each rule (see comments)."""
import numpy as np
import pytest

import baselines
from baselines import common, dependency_only, evidence_only, llm_oneshot, similarity_prop
from csmr import llm
from csmr.adapters.memory import InMemoryAdapter
from csmr.config import load_config
from csmr.scenario import load_scenario
from csmr.types import Action, CandidateInfo, StateChange
from eval import metrics as M

CFG = load_config()
SOLD = StateChange("user", "vehicle_ownership", "owns a car", "no car", "2026-06-01T11:00:00", "user", 0.95,
                   "The user sold their car.", "remove")


def _scenario():
    sc = load_scenario("data/dev/car_sale.json")  # M1 owns a car .. M5 likes Italian food; obs "I sold my car."
    return sc, InMemoryAdapter(sc.memories)


@pytest.fixture
def stubs(monkeypatch):
    """Phase A -> SOLD with root M1; every other LLM call is a rewrite (or the one-shot call, set per test)."""
    state = {"delta": SOLD, "oneshot": None, "prompts": []}
    monkeypatch.setattr(common, "phase_a", lambda obs, store, cfg: (state["delta"], None, ["M1"] if state["delta"] else []))

    def call(p, **k):
        state["prompts"].append(p)
        if p.startswith("You maintain an assistant's long-term memory"):
            return state["oneshot"]
        return {"text": "REVISED", "condition": "COND"}

    monkeypatch.setattr(llm, "call", call)
    return state


def _cands(**spec):
    """memory id -> (p_dep, s_evidence); M1 is the root."""
    return {m: CandidateInfo(m, 0 if m == "M1" else 1, "root" if m == "M1" else "hop", is_root=m == "M1", p_dep=p,
                             s_evidence=s) for m, (p, s) in spec.items()}


def _changed(t):
    return {m: a for m, a in t.final_actions.items() if a != "KEEP"}


def test_registry_has_the_seven_baselines():
    assert list(baselines.BASELINES) == ["append_only", "last_write_wins", "similarity_prop", "graph_propagation",
                                         "dependency_only", "evidence_only", "llm_oneshot"]


# graph (hop_sim 0.35): G1-G2 cos .5, G2-G3 cos .43, G1-G3 0; G6 shares G2's attribute (linked despite cos 0);
# G4 shares nothing; G5 has G1's attribute but another entity (never linked)
_S = 0.75 ** 0.5
_GRAPH = [("G1", "user", "car", (1, 0, 0)), ("G2", "user", "commute", (.5, _S, 0)), ("G3", "user", "fuel", (0, .5, _S)),
          ("G4", "user", "food", (0, 0, -1)), ("G5", "Acme", "car", (1, 0, 0)), ("G6", "user", "commute", (0, 0, -1))]


@pytest.mark.parametrize("hops,change_type,expected", [
    (1, "remove", {"G1": "REVOKE", "G2": "REVISE"}),
    (1, "replace", {"G1": "REVISE", "G2": "REVISE"}),
    (2, "remove", {"G1": "REVOKE", "G2": "REVISE", "G3": "REVISE", "G6": "REVISE"}),
])
def test_graph_propagation_changes_everything_within_k_hops(stubs, monkeypatch, hops, change_type, expected):
    from baselines import graph_propagation
    from csmr.types import Memory, Observation
    store = InMemoryAdapter([Memory(i, f"text {i}", e, a, "2026-01-01T00:00:00") for i, e, a, _ in _GRAPH])
    vec = {f"text {i}": v for i, _, _, v in _GRAPH}
    monkeypatch.setattr(graph_propagation.models, "embed", lambda texts: np.array([vec[t] for t in texts], float))
    stubs["delta"] = StateChange(**{**SOLD.__dict__, "change_type": change_type})
    monkeypatch.setattr(common, "phase_a", lambda obs, s, cfg: (stubs["delta"], None, ["G1"]))
    t = baselines.BASELINES["graph_propagation"](Observation("x", "user", "2026-02-01T00:00:00"), store,
                                                 {**CFG, "graph_prop_hops": hops})
    assert _changed(t) == expected and t.candidates == {}  # no dependency/evidence model


def test_graph_propagation_uses_phase_b_graph():
    """The baseline and Phase B share one graph definition (hop_expansion), not two copies."""
    from baselines import graph_propagation
    from csmr.pipeline import candidates
    assert graph_propagation.hop_expansion is candidates.hop_expansion


@pytest.mark.parametrize("name", list(baselines.BASELINES))
def test_every_baseline_commits_the_same_observation_memory_and_csmr_trace_shape(stubs, monkeypatch, name):
    monkeypatch.setattr(dependency_only, "phase_bc", lambda *a: _cands(M1=(1.0, .9)))
    monkeypatch.setattr(evidence_only, "phase_bc", lambda *a: _cands(M1=(1.0, .9)))
    monkeypatch.setattr(similarity_prop.models, "embed", lambda texts: np.zeros((len(texts), 2)))
    stubs["oneshot"] = {"actions": {}}
    sc, store = _scenario()
    t = baselines.BASELINES[name](sc.observation, store, CFG)
    assert t.chosen == name and set(t.final_actions) == {"M1", "M2", "M3", "M4", "M5"}
    obs_mem = store.get("OBS6")  # same id/text as CSMR's observation_memory on this store
    assert (obs_mem.text, obs_mem.time) == ("The user sold their car.", sc.observation.time)
    assert t.transitions[-1]["id"] == "OBS6" and t.transitions[-1]["action"] == "ADD"
    assert "calls" in t.llm_stats and t.wall_time_s >= 0
    # metrics.py takes it unchanged
    assert M.repair(M.Record(sc, t.final_actions, t, store))["action_accuracy"][1] == 5


def test_append_only_changes_nothing(stubs):
    sc, store = _scenario()
    t = baselines.BASELINES["append_only"](sc.observation, store, CFG)
    assert _changed(t) == {} and [x["id"] for x in t.transitions] == ["OBS6"]
    assert all(store.get(f"M{i}").version == 1 for i in range(1, 6))


@pytest.mark.parametrize("change_type,action", [("remove", "REVOKE"), ("replace", "REVISE")])
def test_last_write_wins_same_entity_and_attribute_only(stubs, change_type, action):
    stubs["delta"] = StateChange(**{**SOLD.__dict__, "entity": "USER", "change_type": change_type})  # case-insensitive
    sc, store = _scenario()
    t = baselines.BASELINES["last_write_wins"](sc.observation, store, CFG)
    assert _changed(t) == {"M1": action}  # only M1 has attribute vehicle_ownership; M2 (commute_mode) is untouched
    if action == "REVISE":
        assert store.get("M1").text == "REVISED" and store.get("M1").history[0]["text"] == "The user owns a car."


def test_similarity_prop_revises_at_or_above_threshold(stubs, monkeypatch):
    # unit vectors: cos(obs, M_i) = 0.9, 0.4, 0.39, 0.6, 0.1 for M1..M5; threshold 0.4 is inclusive
    cos = {"I sold my car.": None, "The user owns a car.": .9, "The user drives to work by car.": .4,
           "The user spends money on fuel.": .39, "The user prefers driving.": .6, "The user likes Italian food.": .1}
    vec = lambda c: [1.0, 0.0] if c is None else [c, float(np.sqrt(1 - c * c))]
    monkeypatch.setattr(similarity_prop.models, "embed", lambda texts: np.array([vec(cos[t]) for t in texts]))
    sc, store = _scenario()
    t = baselines.BASELINES["similarity_prop"](sc.observation, store, {**CFG, "sim_prop_threshold": 0.4})
    assert _changed(t) == {"M1": "REVISE", "M2": "REVISE", "M4": "REVISE"}  # REVISE, never REVOKE, even for the root


def test_dependency_only_ignores_evidence(stubs, monkeypatch):
    seen = {}

    def bc(delta, obs, roots, store, cfg):
        seen["cfg"] = cfg
        return _cands(M1=(1.0, .9), M2=(.9, .1), M3=(.49, .9), M4=(.5, .2))

    monkeypatch.setattr(dependency_only, "phase_bc", bc)
    sc, store = _scenario()
    t = baselines.BASELINES["dependency_only"](sc.observation, store, CFG)
    # P_dep >= tau_d (0.5): M1 (root, removed -> REVOKE), M2, M4 (low S does not matter); M3 at 0.49 stays KEEP
    assert _changed(t) == {"M1": "REVOKE", "M2": "REVISE", "M4": "REVISE"}
    assert seen["cfg"]["ablation"]["use_evidence"] is False and CFG["ablation"]["use_evidence"] is True
    assert set(t.candidates) == {"M1", "M2", "M3", "M4"}  # C is in the trace (dependency/candidate metrics apply)


def test_evidence_only_ignores_dependency(stubs, monkeypatch):
    seen = {}

    def bc(delta, obs, roots, store, cfg):
        seen["cfg"] = cfg
        return _cands(M1=(1.0, .9), M2=(1.0, .59), M3=(1.0, .6), M5=(1.0, .95))

    monkeypatch.setattr(evidence_only, "phase_bc", bc)
    sc, store = _scenario()
    t = baselines.BASELINES["evidence_only"](sc.observation, store, CFG)
    # S >= tau_e (0.6): M1 (root -> REVOKE), M3, M5; M2 at 0.59 stays KEEP
    assert _changed(t) == {"M1": "REVOKE", "M3": "REVISE", "M5": "REVISE"}
    assert seen["cfg"]["ablation"]["use_dependency"] is False and CFG["ablation"]["use_dependency"] is True


def test_llm_oneshot_one_call_actions_per_memory(stubs):
    stubs["oneshot"] = {"actions": {"M1": "REVOKE", "M2": "revise", "M3": "VERIFY", "M4": "CONDITIONAL",
                                    "M9": "REVOKE"}}  # M9 unknown -> ignored; M5 missing -> KEEP
    sc, store = _scenario()
    t = baselines.BASELINES["llm_oneshot"](sc.observation, store, CFG)
    assert _changed(t) == {"M1": "REVOKE", "M2": "REVISE", "M3": "VERIFY", "M4": "CONDITIONAL"}
    oneshot = [p for p in stubs["prompts"] if p.startswith("You maintain")]
    assert len(oneshot) == 1 and '"I sold my car."' in oneshot[0]
    assert all(f"- M{i} [" in oneshot[0] for i in range(1, 6))  # every memory, with its date
    assert store.get("M2").text == "REVISED" and store.get("M4").condition == "COND"  # CSMR's rewrite helpers
    assert t.llm_stats["fallbacks"] == {}


@pytest.mark.parametrize("bad", [None, {"actions": {"M1": "DELETE"}}, {"actions": ["M1"]}, {"other": 1}])
def test_llm_oneshot_malformed_falls_back_to_all_keep(stubs, bad):
    stubs["oneshot"] = bad
    sc, store = _scenario()
    t = baselines.BASELINES["llm_oneshot"](sc.observation, store, CFG)
    assert _changed(t) == {} and t.llm_stats["fallbacks"] == {"llm_oneshot": 1}
    assert len([p for p in stubs["prompts"] if p.startswith("You maintain")]) == 2  # retried once, then fallback


def test_no_state_change_stores_nothing_like_run_pipeline(stubs):
    stubs["delta"] = None
    sc, store = _scenario()
    t = baselines.BASELINES["last_write_wins"](sc.observation, store, CFG)
    assert t.delta is None and t.final_actions == {} and t.transitions == []
    assert len(store.all(include_revoked=True)) == 5
