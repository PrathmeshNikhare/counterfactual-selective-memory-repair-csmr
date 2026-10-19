import copy

import pytest

from csmr import llm
from csmr.adapters.memory import InMemoryAdapter
from csmr.config import load_config
from csmr.pipeline import counterfactual as cf
from csmr.pipeline.overlay import observation_memory, view
from csmr.pipeline.repair_sets import generate_repair_sets
from csmr.scenario import load_scenario
from csmr.types import Action, CandidateInfo, StateChange

CFG = load_config()
DELTA = StateChange("user", "vehicle_ownership", "owns a car", "no car", "2026-06-01T11:00:00", "user", 0.95,
                    "The user sold their car.", "remove")


def _c(mid, hop, p, s, root=False):
    return CandidateInfo(memory_id=mid, hop=hop, retrieved_by="root" if root else "hop", is_root=root, p_dep=p, s_evidence=s)


def _setup(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    store = InMemoryAdapter(sc.memories)
    cands = {"M1": _c("M1", 0, 1.0, 0.9, True), "M2": _c("M2", 1, 0.9, 0.9), "M3": _c("M3", 1, 0.6, 0.4),
             "M4": _c("M4", 2, 0.1, 0.9)}
    prop = {"M1": Action.REVOKE, "M2": Action.REVISE, "M3": Action.VERIFY}
    sets = generate_repair_sets(cands, prop, CFG)
    revised = {"M2": "The user previously drove to work by car; commute now unknown."}
    # fake agent returns the whole view; fake judge checks which stale facts survive
    monkeypatch.setattr(cf, "answer", lambda q, mems: " ".join(m.text for m in mems))

    def fake_judge(a, q, d, o, cfg):
        if "own" in q:
            v = "contradicts" if "owns a car" in a else "consistent"
        elif "commute" in q:
            v = "contradicts" if "drives to work" in a else "consistent"
        else:
            v = "consistent" if "[unverified] The user spends" in a else "vague"
        return {"determined_by_change": True, "verdict": v, "score": cf.VERDICT_SCORE[v]}

    monkeypatch.setattr(cf, "judge", fake_judge)
    probes = ["Does the user own a car?", "How does the user commute?", "Does the user spend on fuel?"]
    return store, cands, sets, revised, probes, sc.observation


def test_car_selects_r2_with_breakdown(monkeypatch):
    store, cands, sets, revised, probes, obs = _setup(monkeypatch)
    before = copy.deepcopy(store.all(include_revoked=True))
    best, rows = cf.select(sets, probes, store, revised, {}, cands, DELTA, obs, CFG)
    assert best.name == "R2"
    assert best.actions == {"M1": Action.REVOKE, "M2": Action.REVISE, "M3": Action.VERIFY}
    assert set(best.score) == {"benefit", "collateral", "evidence_cost", "size", "total"}
    assert [r["name"] for r in rows if not r["feasible"]] == ["R4"]  # M3 REVISE needs S >= tau_e
    assert store.all(include_revoked=True) == before


def test_no_counterfactual_ablation_picks_r3(monkeypatch):
    store, cands, sets, revised, probes, obs = _setup(monkeypatch)
    cfg = {**CFG, "ablation": {**CFG["ablation"], "use_counterfactual": False}}
    best, _ = cf.select(sets, probes, store, revised, {}, cands, DELTA, obs, cfg)
    assert best.name == "R2"  # R3 deduplicated into R2


def test_minimality_ablation_zeroes_penalties(monkeypatch):
    store, cands, sets, revised, probes, obs = _setup(monkeypatch)
    cfg = {**CFG, "ablation": {**CFG["ablation"], "use_minimality": False}, "lambda_e": 0.0}
    rs = next(r for r in sets if r.name == "R2")
    s = cf.score(rs, probes, store, revised, {}, cands, DELTA, obs, cfg)
    assert s["total"] == s["benefit"]


@pytest.mark.parametrize("determined,verdict,expected", [
    (False, "consistent", 1.0),   # unaffected / undeterminable: "unknown" is fine
    (True, "vague", 0.5),         # change determines the answer but the answer is "unknown"/vague
    (True, "contradicts", 0.0),   # asserts what the change contradicts
    (True, "consistent", 1.0),
])
def test_judge_rubric_cases(monkeypatch, determined, verdict, expected):
    sc = load_scenario("data/dev/car_sale.json")
    seen = {}

    def stub(p, **k):
        seen["prompt"] = p
        return {"determined_by_change": determined, "verdict": verdict}

    monkeypatch.setattr(llm, "call", stub)
    out = cf.judge("a", "q", DELTA, sc.observation, CFG)
    assert out == {"determined_by_change": determined, "verdict": verdict, "score": expected}
    for phrase in ("determined_by_change", '"unknown" or an\n     uncertain answer is consistent',
                   "is true but the answer is", "asserts something the new fact contradicts"):
        assert phrase in seen["prompt"]


def test_select_rows_carry_judge_fields(monkeypatch):
    store, cands, sets, revised, probes, obs = _setup(monkeypatch)
    _, rows = cf.select(sets, probes, store, revised, {}, cands, DELTA, obs, CFG)
    j = next(r for r in rows if r["name"] == "R2")["judgements"]
    assert [x["question"] for x in j] == probes
    assert all({"answer", "determined_by_change", "verdict", "score"} <= set(x) for x in j)


def test_presupposing_probes_rejected(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")  # observation: "I sold my car."
    cands = {"M1": _c("M1", 0, 1, 1, True), "M2": _c("M2", 1, .9, .9)}
    monkeypatch.setattr(llm, "call", lambda p, **k: {
        "change_question": "Does the user own a car?",
        "memory_questions": {"M1": "What will you do now that you sold your car?",
                             "M2": "How does the user commute to work?"}})
    qs = cf.generate_probes(DELTA, sc.observation, cands, InMemoryAdapter(sc.memories), 6)
    assert qs == ["Does the user own a car?", "How does the user commute to work?"]


def test_probe_prompt_demands_neutrality(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    seen = {}
    monkeypatch.setattr(llm, "call", lambda p, **k: seen.update(p=p) or {"change_question": "q?"})
    cf.generate_probes(DELTA, sc.observation, {"M1": _c("M1", 0, 1, 1, True)}, InMemoryAdapter(sc.memories), 6)
    assert "Do NOT mention or presuppose" in seen["p"] and sc.observation.text not in seen["p"]


def test_generate_probes_delta_plus_one_per_candidate_capped(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    cands = {"M1": _c("M1", 0, 1, 1, True), "M2": _c("M2", 1, .9, .9), "M4": _c("M4", 2, .1, .9)}
    monkeypatch.setattr(llm, "call", lambda p, **k: {"change_question": "D?", "memory_questions":
                                                     {"M4": "q4", "M2": "q2", "M1": "q1", "M9": "junk"}})
    store = InMemoryAdapter(sc.memories)
    assert cf.generate_probes(DELTA, sc.observation, cands, store, 6) == ["D?", "q1", "q2", "q4"]
    assert cf.generate_probes(DELTA, sc.observation, cands, store, 2) == ["D?", "q1"]


def test_observation_memory_in_every_view(monkeypatch):
    store, cands, sets, revised, probes, obs = _setup(monkeypatch)
    nm = observation_memory(DELTA, obs, store)
    assert (nm.text, nm.entity, nm.attribute, nm.time, nm.source) == (
        "The user sold their car.", "user", "vehicle_ownership", obs.time, "user")
    assert nm.id not in {m.id for m in store.all(include_revoked=True)}
    for rs in [r for r in sets if r.feasible]:
        assert nm.text in [m.text for m in view(store, rs, revised, {}, nm)]
    assert nm.id not in {m.id for m in store.all(include_revoked=True)}  # store untouched


@pytest.mark.llm
def test_car_real_llm_prefers_r2():
    from csmr.pipeline import actions
    sc = load_scenario("data/dev/car_sale.json")
    store = InMemoryAdapter(sc.memories)
    cands = {"M1": _c("M1", 0, 1.0, 0.9, True), "M2": _c("M2", 1, 0.9, 0.9), "M3": _c("M3", 1, 0.6, 0.4),
             "M4": _c("M4", 2, 0.1, 0.9)}
    prop = {m: actions.propose_action(m, c, CFG, "remove") for m, c in cands.items()}
    sets = generate_repair_sets(cands, prop, CFG)
    probes = cf.generate_probes(DELTA, sc.observation, cands, store, CFG["n_probes"])
    revised, conds = cf.prepare_texts(sets, store, DELTA, sc.observation)
    best, rows = cf.select(sets, probes, store, revised, conds, cands, DELTA, sc.observation, CFG,
                           observation_memory(DELTA, sc.observation, store))
    print(probes, rows)
    assert best.name in ("R1", "R2")


@pytest.mark.parametrize("ans", ["unknown", "I'm not sure.", "Cannot determine that.", "It is uncertain whether so.", ""])
@pytest.mark.parametrize("determined,verdict,score", [(False, "consistent", 1.0), (True, "vague", 0.5)])
def test_unknown_like_verdict_set_in_code(monkeypatch, ans, determined, verdict, score):
    sc = load_scenario("data/dev/car_sale.json")
    # the LLM says "contradicts" for everything; code must override for unknown-like answers
    monkeypatch.setattr(llm, "call", lambda p, **k: {"determined_by_change": determined, "verdict": "contradicts"})
    assert cf.judge(ans, "q", DELTA, sc.observation, CFG) == {
        "determined_by_change": determined, "verdict": verdict, "score": score}


def test_claim_answer_uses_llm_verdict(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    monkeypatch.setattr(llm, "call", lambda p, **k: {"determined_by_change": True, "verdict": "contradicts"})
    assert cf.judge("Yes, the user owns a car.", "q", DELTA, sc.observation, CFG)["score"] == 0.0


def test_probe_prompt_requires_third_person(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    seen = {}
    monkeypatch.setattr(llm, "call", lambda p, **k: seen.update(p=p) or {"change_question": "q?"})
    cf.generate_probes(DELTA, sc.observation, {"M1": _c("M1", 0, 1, 1, True)}, InMemoryAdapter(sc.memories), 6)
    assert 'third person ("the user"), never "you"' in seen["p"]


def test_long_claim_with_uncertain_goes_to_llm(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    monkeypatch.setattr(llm, "call", lambda p, **k: {"determined_by_change": True, "verdict": "contradicts"})
    long_claim = ("The user drives to work by car every day and spends money on fuel weekly, although it is "
                  "uncertain how long they will keep doing so.")
    assert len(long_claim.split()) > CFG["unknown_max_words"]
    assert cf.judge(long_claim, "q", DELTA, sc.observation, CFG)["verdict"] == "contradicts"
    long_unknown = "Unknown, since " + "the memories say nothing about it " * 3
    assert cf.judge(long_unknown, "q", DELTA, sc.observation, CFG)["verdict"] == "vague"
