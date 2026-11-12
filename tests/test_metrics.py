"""eval/metrics.py: stubbed agent/judge, every expected value computed by hand (see comments)."""
import json
import math
from pathlib import Path

import pytest

from csmr import llm
from csmr.adapters.memory import InMemoryAdapter
from csmr.pipeline import counterfactual as cf
from csmr.scenario import GoldLabels, Scenario, load_scenario
from csmr.types import Memory, Observation, Trace
from eval import metrics as M

BOOT = {"seed": 42, "bootstrap_samples": 2000, "ci_level": 0.95}
CFG = {**BOOT, "tau_d": 0.5, "persistence_distractors": ["d1", "d2", "d3"]}


def _mem(mid, text, time="2026-01-01T00:00:00", status="active"):
    return Memory(id=mid, text=text, entity="user", attribute="a", time=time, status=status)


def _mems(**status):
    return [_mem(f"M{i}", f"mem-{i}", status=status.get(f"M{i}", "active")) for i in range(1, 7)]


def _record(gold_actions=None, later=()):
    gold = GoldLabels(
        actions=gold_actions or {"M1": "REVOKE", "M2": "REVISE", "M3": "REVISE", "M4": "VERIFY", "M5": "KEEP",
                                 "M6": "KEEP"},
        dependencies=[("M1", "M2"), ("M2", "M3"), ("M1", "M4")],
        eval_questions=[{"q": "q1", "gold_answer": "mem-1"}, {"q": "q2", "gold_answer": "[unverified] mem-2"},
                        {"q": "q3", "gold_answer": "mem-6"}],
        persistence_memories=list(later))
    sc = Scenario("s", _mems(), Observation("o", "user", "2026-02-01T00:00:00"), gold)
    trace = Trace(observation={}, roots=["M1"], wall_time_s=1.5,
                  llm_stats={"calls": 7, "hits": 2, "misses": 5, "prior_hits": 0},
                  candidates={"M1": {"is_root": True, "p_dep": 1.0}, "M2": {"is_root": False, "p_dep": 0.8},
                              "M3": {"is_root": False, "p_dep": 0.3}, "M5": {"is_root": False, "p_dep": 0.5}})
    final = {"M1": "REVISE", "M2": "VERIFY", "M4": "VERIFY", "M5": "REVOKE", "M6": "KEEP"}  # M3 left out => KEEP
    return M.Record(sc, final, trace, InMemoryAdapter(_mems(M1="revoked", M2="pending_verify")))


def _stub_agent(monkeypatch):
    """Agent sees only the 5 newest memories; judge = gold answer appears verbatim. Returns the views it was given."""
    views = []
    monkeypatch.setattr(M, "answer", lambda q, mems, **k: views.append(mems) or " | ".join(m.text for m in mems[-5:]))
    monkeypatch.setattr(M, "qa_judge", lambda q, gold, ans, cfg, pending=False: gold in ans)
    return views


def test_repair_counts():
    # changed = {M1, M5}; gold-changed = {M1, M2, M3}; gold KEEP = {M5, M6}
    assert M.repair(_record()) == {
        "repair_precision": (1, 2),   # M1 of {M1, M5}
        "repair_recall": (1, 3),      # M1 of {M1, M2, M3}
        "over_invalidation": (1, 2),  # M5 of {M5, M6}
        "under_repair": (1, 3),       # only M3 was left KEEP (M2 got VERIFY)
        "action_accuracy": (2, 6),    # M4, M6 (M1: REVISE != REVOKE)
    }


def test_gold_label_outside_the_five_actions_raises():
    with pytest.raises(ValueError):
        M.repair(_record({"M1": "REVOKE/REVISE", "M2": "KEEP", "M3": "KEEP", "M4": "KEEP", "M5": "KEEP", "M6": "KEEP"}))


def test_dependent_level_dependency_counts():
    # predicted dependents (P >= 0.5, root M1 excluded) = {M2, M5}; gold dependents = {M2, M3, M4}
    assert M.dependency(_record(), CFG) == {"dependent_precision": (1, 2), "dependent_recall": (1, 3),
                                            "dependent_f1": (2, 5)}


def test_candidate_recall():
    assert M.candidate_recall(_record()) == {"candidate_recall": (3, 4)}  # gold non-KEEP M1..M4; M4 is not in C


def test_cost_reports_wall_time_only_for_uncached_runs():
    rec = _record()
    assert M.cost(rec) == {"llm_calls": (7, 1), "wall_time_s": (1.5, 1)}  # the 2 hits are within-run
    rec.trace.llm_stats["prior_hits"] = 1
    assert M.cost(rec) == {"llm_calls": (7, 1)}
    del rec.trace.llm_stats["prior_hits"]  # trace written before the counter existed
    assert M.cost(rec) == {"llm_calls": (7, 1)}


def test_qa_accuracy(monkeypatch):
    _stub_agent(monkeypatch)
    # committed view = M2..M6 (M1 revoked, M2 rendered "[unverified] mem-2"): q1 wrong, q2 and q3 right
    assert M.qa(_record(), CFG) == {"qa_accuracy": (2, 3)}


def test_persistence_uses_scenario_memories(monkeypatch):
    views = _stub_agent(monkeypatch)
    rec = _record(later=[_mem("P1", "later-1", "2026-02-03T00:00:00"), _mem("P2", "later-2", "2026-02-09T00:00:00")])
    # the agent's 5 newest = M4, M5, M6, P1, P2 -> only q3 right
    assert M.persistence(rec, CFG) == {"persistence": (1, 3), "persistence_fallback": (0, 1)}
    assert [m.id for m in views[0]] == ["M2", "M3", "M4", "M5", "M6", "P1", "P2"]
    assert len(rec.store.all(include_revoked=True)) == 6  # later memories never reach the store


def test_persistence_falls_back_to_config_distractors(monkeypatch):
    views = _stub_agent(monkeypatch)
    rec = _record()
    # the agent's 5 newest = M5, M6, d1, d2, d3 -> only q3 right
    assert M.persistence(rec, CFG) == {"persistence": (1, 3), "persistence_fallback": (1, 1)}
    assert [(m.text, m.time) for m in views[0][-3:]] == [  # dated after the observation (2026-02-01)
        ("d1", "2026-02-02T00:00:00"), ("d2", "2026-02-03T00:00:00"), ("d3", "2026-02-04T00:00:00")]
    assert len(rec.store.all(include_revoked=True)) == 6


def test_persistence_memory_not_after_the_observation_raises(monkeypatch):
    _stub_agent(monkeypatch)
    with pytest.raises(ValueError, match="P1"):
        M.persistence(_record(later=[_mem("P1", "later-1", "2026-02-01T00:00:00")]), CFG)


def test_loader_reads_persistence_memories(tmp_path):
    assert load_scenario("data/dev/car_sale.json").gold.persistence_memories == []
    d = json.loads(Path("data/dev/car_sale.json").read_text(encoding="utf-8"))
    d["gold"]["persistence_memories"] = [{"id": "P1", "text": "x", "entity": "user", "attribute": "a",
                                          "time": "2026-06-02T09:00:00", "source": "user"}]
    (tmp_path / "s.json").write_text(json.dumps(d), encoding="utf-8")
    assert load_scenario(tmp_path / "s.json").gold.persistence_memories == [
        Memory("P1", "x", "user", "a", "2026-06-02T09:00:00")]


def test_methods_without_trace_or_store_skip_those_metrics(monkeypatch):
    _stub_agent(monkeypatch)
    rec = _record()
    rec.trace = rec.store = None
    assert set(M.scenario_metrics(rec, CFG)) == set(M.repair(rec))
    assert len(M.scenario_metrics(_record(), CFG)) == 14  # 5 repair, 3 dependent, 1 candidate, 1 qa, 2 persist., 2 cost


def test_qa_judge_pending_rule_only_for_pending_questions(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "try_json", lambda p, **k: seen.append(p) or {"correct": True})
    cfg = {"llm": {"judge_model": None}, "judge_max_tokens": 8}
    M.qa_judge("Where?", "Pune", "In Pune.", cfg)
    M.qa_judge("Where?", "Pune", "In Pune, though they may move.", cfg, pending=True)
    assert M.QA_JUDGE_PENDING not in seen[0] and '"unknown" is incorrect.\nReturn JSON only' in seen[0]  # unchanged
    assert M.QA_JUDGE_PENDING in seen[1]


def test_qa_passes_pending_flag_to_judge(monkeypatch):
    got = []
    monkeypatch.setattr(M, "answer", lambda q, mems, **k: "a")
    monkeypatch.setattr(M, "qa_judge", lambda q, gold, ans, cfg, pending=False: got.append((q, pending)) or True)
    rec = _record()
    rec.scenario.gold.eval_questions[1]["pending"] = True
    M.qa(rec, CFG)
    assert got == [("q1", False), ("q2", True), ("q3", False)]


def test_qa_judge_has_its_own_prompt_retries_once_then_fails_closed(monkeypatch):
    assert M.QA_JUDGE_PROMPT != cf.JUDGE_PROMPT
    seen, replies = [], [{"correct": "yes"}, {"correct": True}, None, {"verdict": "consistent"}]
    monkeypatch.setattr(llm, "try_json", lambda p, **k: seen.append(p) or replies.pop(0))
    cfg = {"llm": {"judge_model": None}, "judge_max_tokens": 8}
    llm.reset_stats()
    assert M.qa_judge("Where?", "Pune", "In Pune.", cfg) is True  # malformed, then valid on the retry
    assert all(s in seen[0] for s in ("Where?", "Reference answer: Pune", "Assistant answer: In Pune."))
    assert len(seen) == 2 and seen[1] != seen[0] and llm.stats()["fallbacks"] == {}
    assert M.qa_judge("Where?", "Pune", "In Pune.", cfg) is False  # malformed twice -> incorrect, and counted
    assert len(seen) == 4 and llm.stats()["fallbacks"] == {"qa_judge": 1}
    llm.reset_stats()


def test_aggregate_pools_counts_over_scenarios():
    out = M.aggregate([{"p": (1, 2)}, {"p": (2, 2)}, {"p": (0, 0)}], BOOT)["p"]
    assert out["value"] == 0.75 and out["n"] == 2  # 3/4 pooled, not the mean of 0.5 and 1.0 over 3 scenarios


def test_bootstrap_ci_hand_cases():
    # identical scenarios: every resample pools to 4/12
    assert M.aggregate([{"m": (1, 3)}] * 4, BOOT)["m"] == {"value": 4 / 12, "lo": 4 / 12, "hi": 4 / 12, "n": 4}
    # scenarios scoring 1 and 0: resamples are 0, 0.5, 1 with probability 1/4, 1/2, 1/4 -> 95% CI is [0, 1]
    assert M.aggregate([{"m": (1, 1)}, {"m": (0, 1)}], BOOT)["m"] == {"value": 0.5, "lo": 0.0, "hi": 1.0, "n": 2}
    # metric undefined in one scenario: resamples are 0.5 or undefined (dropped)
    assert M.aggregate([{"m": (1, 2)}, {}], BOOT)["m"] == {"value": 0.5, "lo": 0.5, "hi": 0.5, "n": 1}
    # never defined
    out = M.aggregate([{"m": (0, 0)}], BOOT)["m"]
    assert math.isnan(out["value"]) and math.isnan(out["lo"]) and math.isnan(out["hi"]) and out["n"] == 0
