import copy
import re

import pytest

from csmr import llm
from csmr.adapters.memory import InMemoryAdapter
from csmr.config import load_config
from csmr.pipeline import counterfactual as cf
from csmr.pipeline.overlay import observation_memory, view
from csmr.pipeline.repair_sets import generate_repair_sets
from csmr.scenario import load_scenario
from csmr.types import Action, CandidateInfo, Memory, StateChange

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
    monkeypatch.setattr(cf, "answer", lambda q, mems, **k: " ".join(m.text for m in mems))

    def fake_judge(a, q, d, o, cfg):
        if "own" in q:
            v = "contradicts" if "owns a car" in a else "consistent"
        elif "commute" in q:
            v = "contradicts" if "drives to work" in a else "consistent"
        else:
            v = "consistent" if "[unverified] The user spends" in a else "vague"
        return {"determined_by_change": True, "verdict": v, "score": cf.VERDICT_SCORE[v]}

    monkeypatch.setattr(cf, "judge", fake_judge)
    probes = ["What vehicle does the user own?", "How does the user commute?", "What does the user spend on fuel?"]
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
        "change_question": "What vehicle does the user own?",
        "memory_questions": {"M1": "What will the user do after they sold the car?",
                             "M2": "How does the user commute to work?"}})
    qs = cf.generate_probes(DELTA, sc.observation, cands, InMemoryAdapter(sc.memories), 6)
    assert qs == ["What vehicle does the user own?", "How does the user commute to work?"]


def test_yes_no_and_change_hinting_probes_rejected(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    cands = {f"M{i}": _c(f"M{i}", 1, .9, .9) for i in range(1, 6)}
    cands["M1"].is_root = True
    monkeypatch.setattr(llm, "call", lambda p, **k: {
        "change_question": "Which vehicle does the user own?",
        "memory_questions": {"M1": "Does the user own a car?",
                             "M2": "How does the user still commute to work?",
                             "M3": "Where does the user no longer park?",
                             "M4": "What does the user spend on fuel anymore?",
                             "M5": "In which city does the user drive?"}})
    qs = cf.generate_probes(DELTA, sc.observation, cands, InMemoryAdapter(sc.memories), 9)
    assert qs == ["Which vehicle does the user own?", "In which city does the user drive?"]


def test_template_probe_with_hint_word_kept(monkeypatch):
    store = InMemoryAdapter([Memory("M1", "The user now lives in Pune.", "user", "city", "2026-03-02")])
    obs = load_scenario("data/dev/car_sale.json").observation
    llm.reset_stats()
    monkeypatch.setattr(llm, "call", lambda p, **k: {})  # malformed -> template, which mentions "now" from the memory
    assert cf.generate_probes(DELTA, obs, {"M1": _c("M1", 0, 1, 1, True)}, store, 6) == [
        "What is the current state of this: The user now lives in Pune?"]
    assert llm.stats()["fallbacks"] == {"probes": 1}


def test_all_llm_probes_filtered_falls_back_to_templates(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    cands = {"M1": _c("M1", 0, 1, 1, True), "M2": _c("M2", 1, .9, .9)}
    store = InMemoryAdapter(sc.memories)
    monkeypatch.setattr(llm, "call", lambda p, **k: {
        "change_question": "Does the user own a car?",
        "memory_questions": {"M1": "Is the car still owned?", "M2": "How does the user commute after they sold it?"}})
    llm.reset_stats()
    assert cf.generate_probes(DELTA, sc.observation, cands, store, 6) == [
        f"What is the current state of this: {store.get(m).text.rstrip('. ')}?" for m in ("M1", "M2")]
    assert llm.stats()["fallbacks"] == {"probes": 1}


def test_probe_prompt_asks_open_questions(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    seen = {}
    monkeypatch.setattr(llm, "call", lambda p, **k: seen.update(p=p) or {"change_question": "q?"})
    cf.generate_probes(DELTA, sc.observation, {"M1": _c("M1", 0, 1, 1, True)}, InMemoryAdapter(sc.memories), 6)
    assert "never yes/no questions" in seen["p"] and '"still", "anymore", "no longer"' in seen["p"]
    assert not re.search(r'"(Does|Is|Do|Are) ', seen["p"])  # no yes/no examples left in the prompt


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


def test_malformed_probes_fall_back_to_templates(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    cands = {"M1": _c("M1", 0, 1, 1, True), "M2": _c("M2", 1, .9, .9)}
    store = InMemoryAdapter(sc.memories)
    llm.reset_stats()
    for bad in ({}, {"change_question": ""}, {"change_question": "q?", "memory_questions": "oops"}):
        monkeypatch.setattr(llm, "call", lambda p, **k: bad)
        qs = cf.generate_probes(DELTA, sc.observation, cands, store, 6)
        assert qs == [f"What is the current state of this: {store.get(m).text.rstrip('. ')}?" for m in ("M1", "M2")]
    assert llm.stats()["fallbacks"] == {"probes": 3}


def test_probe_missing_memory_question_not_a_fallback(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    monkeypatch.setattr(llm, "call", lambda p, **k: {"change_question": "D?", "memory_questions": {}})
    llm.reset_stats()
    assert cf.generate_probes(DELTA, sc.observation, {"M1": _c("M1", 0, 1, 1, True)},
                              InMemoryAdapter(sc.memories), 6) == ["D?"]
    assert llm.stats()["fallbacks"] == {}


def test_judge_retries_once_with_changed_prompt(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    prompts = []

    def stub(p, **k):
        prompts.append(p)
        return {"verdict": "nonsense"} if len(prompts) == 1 else {"determined_by_change": True, "verdict": "contradicts"}

    llm.reset_stats()
    monkeypatch.setattr(llm, "call", stub)
    out = cf.judge("Yes, the user owns a car.", "q", DELTA, sc.observation, CFG)
    assert out["score"] == 0.0 and len(prompts) == 2 and prompts[0] != prompts[1]
    assert llm.stats()["fallbacks"] == {}


@pytest.mark.parametrize("ans", ["Yes, the user owns a car.", "unknown"])
def test_judge_still_malformed_is_vague_and_counted(monkeypatch, ans):
    sc = load_scenario("data/dev/car_sale.json")
    llm.reset_stats()
    monkeypatch.setattr(llm, "call", lambda p, **k: {"determined_by_change": "maybe"})
    assert cf.judge(ans, "q", DELTA, sc.observation, CFG) == {
        "determined_by_change": None, "verdict": "vague", "score": 0.5}
    assert llm.stats()["fallbacks"] == {"judge": 1}


def test_judge_undecodable_json_is_vague(monkeypatch):
    import json
    sc = load_scenario("data/dev/car_sale.json")

    def boom(p, **k):
        raise json.JSONDecodeError("bad", "", 0)

    llm.reset_stats()
    monkeypatch.setattr(llm, "call", boom)
    assert cf.judge("Yes.", "q", DELTA, sc.observation, CFG)["score"] == 0.5
    assert llm.stats()["fallbacks"] == {"judge": 1}


def test_max_tokens_come_from_config(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    seen = []
    monkeypatch.setattr(llm, "call", lambda p, **k: seen.append(k["max_tokens"]) or {
        "change_question": "q?", "memory_questions": {}, "determined_by_change": False, "verdict": "consistent"})
    cf.generate_probes(DELTA, sc.observation, {"M1": _c("M1", 0, 1, 1, True)}, InMemoryAdapter(sc.memories), 6)
    cf.judge("Yes.", "q", DELTA, sc.observation, CFG)
    assert seen == [CFG["probe_max_tokens"], CFG["judge_max_tokens"]]


def test_near_ties_prefer_smaller_set(monkeypatch):
    store, cands, sets, revised, probes, obs = _setup(monkeypatch)
    big = next(r for r in sets if r.name == "R2")
    small = next(r for r in sets if r.name == "R1")
    totals = {"R1": 0.5, "R2": 0.5 + CFG["tie_eps"] / 2}  # R2 higher by float noise only

    def fake_eval(rs, *a, **k):
        return {"benefit": 0, "collateral": 0, "evidence_cost": 0, "size": rs.size(), "total": totals.get(rs.name, 0.0)}, []

    monkeypatch.setattr(cf, "_evaluate", fake_eval)
    best, _ = cf.select([small, big], probes, store, revised, {}, cands, DELTA, obs, CFG)
    assert best.name == "R1"


def test_ablation_infeasible_r3_falls_back_to_largest_feasible_lower_set(monkeypatch):
    store, cands, _, revised, probes, obs = _setup(monkeypatch)
    cands["M1"].s_evidence = 0.3  # weak-evidence root: R1+ force it to REVISE -> infeasible
    sets = generate_repair_sets(cands, {"M1": Action.KEEP, "M2": Action.REVISE}, CFG)
    assert [r.name for r in sets if not r.feasible] == ["R1", "R2", "R4"]  # R3 deduplicated into R2
    cfg = {**CFG, "ablation": {**CFG["ablation"], "use_counterfactual": False}}
    best, _ = cf.select(sets, probes, store, revised, {}, cands, DELTA, obs, cfg)
    assert best.name == "R0"


def test_probe_order_roots_then_p_dep_then_numeric_id(monkeypatch):
    sc = load_scenario("data/dev/car_sale_distractors.json")
    cands = {"M12": _c("M12", 1, .5, .9), "M2": _c("M2", 2, .9, .9), "M1": _c("M1", 2, .1, 1, True),
             "M3": _c("M3", 1, .5, .9)}
    qs = {m: f"q{m}" for m in cands}
    monkeypatch.setattr(llm, "call", lambda p, **k: {"change_question": "D?", "memory_questions": qs})
    out = cf.generate_probes(DELTA, sc.observation, cands, InMemoryAdapter(sc.memories), 9)
    assert out == ["D?", "qM1", "qM2", "qM3", "qM12"]  # root first; higher p_dep next; ties by numeric id


@pytest.mark.parametrize("ans", ["Yes, the user drives to work every day.", "Yes, the user spends money on fuel."])
@pytest.mark.parametrize("determined", [True, False])
def test_stale_claim_is_contradicts_whatever_determined_is(monkeypatch, ans, determined):
    sc = load_scenario("data/dev/car_sale.json")
    seen = {}
    monkeypatch.setattr(llm, "call", lambda p, **k: seen.update(p=p) or {"determined_by_change": determined, "verdict": "contradicts"})
    out = cf.judge(ans, "Does the user drive to work every day?", DELTA, sc.observation, CFG)
    assert out == {"determined_by_change": determined, "verdict": "contradicts", "score": 0.0}
    assert "OR makes likely outdated" in seen["p"] and "whatever determined_by_change is" in seen["p"]


ALEMBIC_PAST = ("They used Alembic migrations generated from SQLAlchemy models, though the current method after moving "
                "to MongoDB Atlas is unknown.")
DB = StateChange("ShopKart", "project_database", "PostgreSQL 15", "MongoDB Atlas", "2026-10-21T17:45:00", "user", 0.95,
                 "ShopKart's backend now runs entirely on MongoDB Atlas.")


def _rule_aware_judge(p, **k):
    """Stub judge that follows the prompt's past-mention rule: past-only mentions are not contradictions."""
    past_rule = "NOT asserting it" in p and "Only claims about the CURRENT state can contradict" in p
    current_claim = "Assistant answer: They use Alembic" in p
    return {"determined_by_change": True, "verdict": "contradicts" if current_claim or not past_rule else "consistent"}


def test_judge_past_mention_is_not_a_contradiction(monkeypatch):
    sc = load_scenario("data/dev/car_sale.json")
    assert len(ALEMBIC_PAST.split()) > CFG["unknown_max_words"]  # not unknown-like: the LLM verdict decides
    monkeypatch.setattr(llm, "call", _rule_aware_judge)
    q = "How are schema changes in ShopKart managed?"
    assert cf.judge(ALEMBIC_PAST, q, DB, sc.observation, CFG)["verdict"] != "contradicts"
    short = "Used Alembic before; current method unknown."  # short + "unknown": verdict set in code -> vague
    assert cf.judge(short, q, DB, sc.observation, CFG)["verdict"] == "vague"
    assert cf.judge("They use Alembic migrations.", q, DB, sc.observation, CFG)["score"] == 0.0


@pytest.mark.llm
def test_judge_past_mention_real_llm():
    sc = load_scenario("data/showcase/02_db_migration_coding_assistant.json")
    q = "How are schema changes in ShopKart managed?"
    assert cf.judge(ALEMBIC_PAST, q, DB, sc.observation, CFG)["verdict"] != "contradicts"
    assert cf.judge("They use Alembic migrations generated from the SQLAlchemy models.", q, DB, sc.observation,
                    CFG)["verdict"] == "contradicts"


def test_probe_prompt_names_subject_not_attribute_key(monkeypatch):
    store = InMemoryAdapter([
        Memory("M1", "The user's ShopKart backend uses PostgreSQL 15 as its main database.", "user", "project_database",
               "2026-03-02"),
        Memory("M11", "The user's side project stores data in SQLite.", "user", "side_project", "2026-07-10")])
    obs = load_scenario("data/showcase/02_db_migration_coding_assistant.json").observation
    seen = []
    monkeypatch.setattr(llm, "call", lambda p, **k: seen.append(p) or {"change_question": "q?"})
    cf.generate_probes(DB, obs, {"M1": _c("M1", 0, 1, 1, True), "M11": _c("M11", 1, .3, .9)}, store, 6)
    p = seen[0]
    assert "project_database" not in p and "project database" not in p
    assert "current value of this stored fact: \"The user's ShopKart backend uses PostgreSQL 15" in p
    assert "Name the specific subject exactly as the memory text names it" in p
    cf.generate_probes(DB, obs, {"M11": _c("M11", 1, .3, .9)}, store, 6)  # no root: first candidate is the fact
    assert "stored fact: \"The user's side project stores data in SQLite.\"" in seen[1]
