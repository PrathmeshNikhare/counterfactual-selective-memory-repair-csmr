import json

from csmr import llm
from csmr.adapters.memory import InMemoryAdapter
from csmr.pipeline import commit as commit_mod
from csmr.pipeline import counterfactual as cf
from csmr.pipeline.run import run_pipeline
from csmr.scenario import load_scenario
from csmr.types import Observation

CAR = {"state_change": {"entity": "user", "attribute": "vehicle_ownership", "old": "owns a car", "new": "no car",
                        "confidence": 0.95, "statement": "The user sold their car.", "change_type": "remove"}}
PROBES = {"change_question": "What vehicle does the user own?",
          "memory_questions": {"M2": "How does the user commute?", "M3": "What does the user spend on fuel?"}}


def _llm(revise):
    def call(p, **k):
        if "reports a CHANGE" in p:
            return CAR
        if "FALSE or OUT OF DATE" in p:
            return {"answer": "yes" if "drives" in p.split("B:")[1] or "fuel" in p.split("B:")[1] else "no"}
        if "Write neutral questions" in p:
            return PROBES
        return revise
    return call


def _fake_judge(a, q, d, o, cfg, src=""):
    if "own" in q:
        v = "contradicts" if "owns a car" in a else "consistent"
    elif "commute" in q:
        v = "contradicts" if "drives to work" in a else "consistent"
    else:
        v = "consistent" if "[unverified] The user spends" in a else "vague"
    return {"determined_by_change": True, "verdict": v, "score": cf.VERDICT_SCORE[v]}


def _run(monkeypatch, revise):
    monkeypatch.setattr(llm, "call", _llm(revise))
    for mod in (cf, commit_mod):
        monkeypatch.setattr(mod, "answer", lambda q, mems, **k: " ".join(m.text for m in mems))
        monkeypatch.setattr(mod, "judge", _fake_judge)
    sc = load_scenario("data/dev/car_sale.json")
    store = InMemoryAdapter(sc.memories)
    return run_pipeline(sc.observation, store), store


def test_car_end_to_end_stubbed(monkeypatch):
    t, store = _run(monkeypatch, {"text": "The user previously drove to work by car; commute now unknown."})
    assert t.roots == ["M1"] and t.chosen == "R2"
    assert [p["source"] for p in t.probes] == ["M1", "M2", "M3"] and t.probe_gaps == {}
    # M3 REVISE (not Table II's VERIFY): dependents inherit the root's contradiction, Spec 5
    assert t.final_actions == {"M1": "REVOKE", "M2": "REVISE", "M3": "REVISE", "M4": "KEEP", "M5": "KEEP"}
    assert "M5" not in t.candidates and t.validation["ok"] and t.llm_stats["fallbacks"] == {}
    assert store.get("M1").status == "revoked" and store.get("M2").version == 2 and store.get("M5").version == 1
    assert {x["id"] for x in t.transitions} == {"M1", "M2", "M3", "OBS6"}
    json.dumps(t.to_dict())  # trace is serialisable


def test_fallbacks_reach_trace(monkeypatch):
    t, _ = _run(monkeypatch, {})
    assert t.llm_stats["fallbacks"]["revise_text"] >= 1


def test_no_state_change_returns_early_and_leaves_store(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "call", lambda p, **k: calls.append(p) or {"state_change": None})
    sc = load_scenario("data/dev/car_sale.json")
    store = InMemoryAdapter(sc.memories)
    t = run_pipeline(Observation("I like pizza.", "user", "2026-06-02T09:00:00"), store)
    assert t.delta is None and t.candidates == {} and t.chosen is None and t.transitions == []
    assert len(calls) == 1 and "fallbacks" in t.llm_stats
    assert [(m.id, m.version) for m in store.all(include_revoked=True)] == [(m.id, 1) for m in sc.memories]


def test_ablation_flags_in_passed_cfg_are_honoured(monkeypatch):
    from csmr.config import load_config
    cfg = load_config()
    cfg["ablation"] = {**cfg["ablation"], "use_dependency": False, "use_evidence": False}
    monkeypatch.setattr(llm, "call", _llm({"text": "r"}))
    sc = load_scenario("data/dev/car_sale.json")
    t = run_pipeline(sc.observation, InMemoryAdapter(sc.memories), cfg)
    assert all(c["p_dep"] == 1.0 and c["s_evidence"] == 1.0 for c in t.candidates.values())


def _total(row, cfg):
    s = row["score"]
    return s["benefit"] - cfg["lambda_c"] * s["collateral"] - cfg["lambda_e"] * s["evidence_cost"] - cfg["lambda_s"] * s["size"]


def test_passed_cfg_changes_scores_and_load_config_is_never_called(monkeypatch):
    import sys
    from csmr.config import load_config

    def forbidden(*a, **k):
        raise AssertionError("load_config called although cfg was passed")

    for name in ("llm", "pipeline.dependency", "pipeline.evidence", "pipeline.actions", "pipeline.extract",
                 "pipeline.candidates", "pipeline.agent", "pipeline.counterfactual", "pipeline.commit",
                 "pipeline.run"):
        monkeypatch.setattr(sys.modules[f"csmr.{name}"], "load_config", forbidden)
    rows = {}
    for lc in (0.5, 2.0):
        cfg = load_config()
        cfg["lambda_c"] = lc
        monkeypatch.setattr(llm, "call", _llm({"text": "The user previously drove to work by car; commute now unknown."}))
        for mod in (cf, commit_mod):
            monkeypatch.setattr(mod, "answer", lambda q, mems, **k: " ".join(m.text for m in mems))
            monkeypatch.setattr(mod, "judge", _fake_judge)
        sc = load_scenario("data/dev/car_sale.json")
        t = run_pipeline(sc.observation, InMemoryAdapter(sc.memories), cfg)
        rows[lc] = next(r for r in t.repair_sets if r["name"] == "R2")
        assert abs(rows[lc]["score"]["total"] - _total(rows[lc], cfg)) < 1e-9
    assert rows[0.5]["score"]["collateral"] == rows[2.0]["score"]["collateral"] > 0
    assert rows[0.5]["score"]["total"] > rows[2.0]["score"]["total"]


def test_cli_set_overrides_reach_pipeline(monkeypatch, tmp_path):
    import sys
    import run as cli
    seen = {}
    monkeypatch.setattr(cli, "run_pipeline", lambda obs, store, cfg: seen.update(cfg=cfg) or run_pipeline_stub())
    monkeypatch.setattr(sys, "argv", ["run.py", "data/dev/car_sale.json", "--set", "ablation.use_evidence=false",
                                      "--set", "lambda_c=0.9"])
    cli.main()
    assert seen["cfg"]["ablation"]["use_evidence"] is False and seen["cfg"]["lambda_c"] == 0.9


def run_pipeline_stub():
    from csmr.types import Trace
    return Trace(observation={"text": "x"})
