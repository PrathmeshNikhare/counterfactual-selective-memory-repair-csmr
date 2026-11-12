"""eval/fit.py, fully stubbed: a tiny synthetic train/val dataset in tmp_path, no LLM / NLI / embedder calls."""
import json
from dataclasses import asdict

import pytest
import yaml

from csmr.config import load_config
from csmr.pipeline import counterfactual as cf
from csmr.pipeline import dependency, evidence
from csmr.scenario import GoldLabels
from csmr.types import Action, CandidateInfo, RepairSet, StateChange, Trace
from eval import aggregate, fit, run_all

CFG = load_config()
T0, T1 = "2024-01-01T00:00:00", "2024-02-01T00:00:00"
DELTA = StateChange("user", "a", "x", "none", T1, "user", 0.9, "The user sold x.", "remove")
OBS = {"text": "I sold x.", "source": "user", "time": T1}


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Results, models and the train / val splits all under tmp_path."""
    for split in ("train", "val"):
        (tmp_path / split).mkdir()
        monkeypatch.setitem(run_all.SPLITS, split, str(tmp_path / split))  # ROOT / absolute path = absolute path
    monkeypatch.setattr(run_all, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(aggregate, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(dependency, "_MODEL_FILE", tmp_path / "models" / "dependency.json")
    monkeypatch.setattr(evidence, "_MODEL_FILE", tmp_path / "models" / "evidence.json")
    return tmp_path


def _scenario(d, sid, actions, deps=()):
    mems = [{"id": m, "text": f"memory {m}", "entity": "user", "attribute": "a", "time": T0, "source": "user"}
            for m in actions]
    (d / f"{sid}.json").write_text(json.dumps({
        "id": sid, "memories": mems, "observation": OBS, "category": "one_hop",
        "gold": {"actions": actions, "dependencies": [list(p) for p in deps], "eval_questions": []}}))


# ---------------------------------------------------------------- train

GOLD = {"M1": "REVOKE", "M2": "REVISE", "M3": "KEEP", "M4": "VERIFY"}
DEPS = [("M1", "M2"), ("M1", "M4")]


def _cands(noise: float) -> dict[str, CandidateInfo]:
    """M1 root; M2/M4 have a causal cue, M3 (a hard negative) does not; M4's evidence is hedged."""
    out = {}
    for m in GOLD:
        dep = {k: 0.5 + noise for k in dependency.FEATURES} | {"causal_cue": float(m in ("M2", "M4"))}
        ev = {k: 0.5 - noise for k in evidence.FEATURES} | {"not_hedged": float(m != "M4")}
        out[m] = CandidateInfo(m, 0 if m == "M1" else 1, "hop", is_root=m == "M1",
                               dep_features={} if m == "M1" else dep, ev_features=ev)
    return out


@pytest.fixture
def train_env(env, monkeypatch):
    for i in range(3):
        _scenario(env / "train", f"tr{i}", GOLD, DEPS)
    calls = {"a": 0, "cfgs": []}

    def phase_a(obs, store, cfg):
        calls["a"] += 1
        return DELTA, None, ["M1"]

    def phase_bc(delta, obs, roots, store, cfg):
        calls["cfgs"].append(cfg)
        return _cands(0.05 * calls["a"])

    monkeypatch.setattr(fit, "phase_a", phase_a)
    monkeypatch.setattr(fit, "phase_bc", phase_bc)
    return calls


def test_label_rows_follow_gold():
    rows = fit.label_rows(_cands(0), GoldLabels(GOLD, DEPS, []))
    assert [(r["memory"], r["label"]) for r in rows["dependency"]] == [("M2", 1), ("M3", 0), ("M4", 1)]  # no root
    assert [(r["memory"], r["label"]) for r in rows["evidence"]] == [("M1", 1), ("M2", 1), ("M4", 0)]  # no gold KEEP
    with pytest.raises(ValueError):
        fit.label_rows(_cands(0), GoldLabels({**GOLD, "M3": "REVOKE/REVISE"}, DEPS, []))


def test_label_rows_weak_evidence_dependent_and_missed_root():
    cands = _cands(0)
    cands["M1"].is_root, cands["M1"].dep_features = False, cands["M2"].dep_features  # Phase A missed gold root M1
    rows = fit.label_rows(cands, GoldLabels({**GOLD, "M4": "KEEP"}, DEPS, []))  # M4: dependent kept, "maybe"
    assert [r["memory"] for r in rows["dependency"]] == ["M2", "M3", "M4"]  # missed gold root: no dependency label
    assert [(r["memory"], r["label"]) for r in rows["evidence"]] == [("M1", 1), ("M2", 1), ("M4", 0)]


def test_label_rows_leave_conditional_out_of_evidence():
    rows = fit.label_rows(_cands(0), GoldLabels({**GOLD, "M2": "CONDITIONAL"}, DEPS, []))
    assert [(r["memory"], r["label"]) for r in rows["evidence"]] == [("M1", 1), ("M4", 0)]  # VERIFY stays, y = 0
    assert [r["memory"] for r in rows["dependency"]] == ["M2", "M3", "M4"]  # dependency rows unaffected


def test_train_fits_and_saves_both_models(env, train_env):
    cfg = load_config(overrides=["ablation.use_dependency=false"])
    summary = fit.train("t", "1/1", cfg)
    assert all(all(c["ablation"].values()) for c in train_env["cfgs"])  # stages forced on for feature extraction
    dep, ev = (json.loads(m._MODEL_FILE.read_text()) for m in (dependency, evidence))
    assert set(dep) == {"intercept", *dependency.FEATURES} and set(ev) == {"intercept", *evidence.FEATURES}
    assert dep["causal_cue"] > 0 and ev["not_hedged"] > 0
    assert summary["dependency"]["rows"] == 9 and summary["dependency"]["positives"] == 6
    assert summary["evidence"]["rows"] == 9 and summary["evidence"]["positives"] == 6
    assert summary["dependency"]["train_auc"] == summary["evidence"]["train_auc"] == 1.0
    assert (env / "results/fit/t/train_summary.json").exists()


def test_train_resumes_and_shards_skip_fitting(env, train_env):
    assert fit.train("t", "1/2", CFG) is None
    assert train_env["a"] == 2 and not dependency._MODEL_FILE.exists()
    fit.train("t", "1/1", CFG)
    assert train_env["a"] == 3 and dependency._MODEL_FILE.exists()  # only the missing scenario was extracted


def test_train_needs_both_labels(env, train_env, monkeypatch):
    monkeypatch.setattr(fit, "label_rows", lambda c, g: {"dependency": [{"memory": "M2", "features": {}, "label": 1}],
                                                         "evidence": []})
    with pytest.raises(SystemExit, match="need both labels"):
        fit.train("t", "1/1", CFG)


# ---------------------------------------------------------------- val

def test_val_runs_csmr_through_run_all(env, monkeypatch):
    got = []
    monkeypatch.setattr(run_all, "run", lambda *a: got.append(a) or 0)
    with pytest.raises(SystemExit, match="fitted models missing"):
        fit.val("t", "2/3", CFG, [])
    fit.val("t", "2/3", CFG, ["x=1"], allow_handset=True, overlay="o.yaml")
    assert got == [("val", ["csmr"], "t", "2/3", CFG, ["x=1"], "o.yaml")]
    with pytest.raises(SystemExit, match="full method"):
        fit.val("t", "1/1", load_config(overrides=["ablation.use_minimality=false"]), [], allow_handset=True)


# ---------------------------------------------------------------- tune
# Val scenario v1 (run at config.yaml: tau_d .5, tau_e .6): M1 root, M2 true dependent, M3 hard negative, M4 not in C.
# Saved Benefits: R0 .3, R1 .6, R2 = {M1, M2} .9, R4 = {M1, M2, M3} 1.0. Collateral: R2 .3, R4 .85.
# v2: no state change (all KEEP).
V1_GOLD = {"M1": "REVOKE", "M2": "REVISE", "M3": "KEEP", "M4": "KEEP"}
V1_CANDS = {"M1": dict(hop=0, is_root=True, p_dep=1.0, s_evidence=0.9),
            "M2": dict(hop=1, p_dep=0.7, s_evidence=0.65), "M3": dict(hop=1, p_dep=0.45, s_evidence=0.65)}
V1_SETS = [("R0", {}, 0.3), ("R1", {"M1": "REVOKE"}, 0.6), ("R2", {"M1": "REVOKE", "M2": "REVISE"}, 0.9),
           ("R4", {"M1": "REVOKE", "M2": "REVISE", "M3": "REVISE"}, 1.0)]


def _raw(env, sid, trace, overlay=None):
    p = env / "results/raw/t/val/csmr" / f"{sid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config(overlay=overlay)
    p.write_text(json.dumps({"split": "val", "tag": "t", "config_hash": run_all.config_hash(cfg, overlay),
                             "git_commit": "g", "overrides": [], "config_overlay": overlay,
                             "method": "csmr", "scenario": sid, "trace": trace.to_dict()}))


@pytest.fixture
def val_env(env):
    _scenario(env / "val", "v1", V1_GOLD, [("M1", "M2")])
    _scenario(env / "val", "v2", {"M1": "KEEP", "M2": "KEEP"})
    cands = {m: {**asdict(CandidateInfo(m, retrieved_by="hop", **c)), "proposed": "KEEP"} for m, c in V1_CANDS.items()}
    sets = [{"name": n, "actions": a, "feasible": True, "score": {"benefit": b}} for n, a, b in V1_SETS]
    probes = [{"question": "What does the user have?", "source": "M1", "source_text": "memory M1"},
              {"question": "Where does the user go?", "source": "M2", "source_text": "memory M2"}]
    _raw(env, "v1", Trace(observation=OBS, delta=asdict(DELTA), roots=["M1"], candidates=cands, probes=probes,
                          repair_sets=sets))
    _raw(env, "v2", Trace(observation=OBS))
    return env


def _grid(**g):
    cfg = load_config()
    cfg["fit"] = {f"{k}_grid": g.get(k, [CFG[k]]) for k in fit.GRID_KEYS}
    return cfg


def test_rescoring_at_the_run_config_reproduces_the_trace_choice(val_env):
    rec = json.loads((val_env / "results/raw/t/val/csmr/v1.json").read_text())
    s = fit.load_saved(rec, val_env / "val/v1.json")
    sets = fit.feasible_sets(s, CFG, online=False, stats={"filled": 0})
    assert [r.name for r in sets] == ["R0", "R1", "R2", "R4"]  # R3 deduplicated into R2
    assert fit.choose(s, sets, CFG) == {"M1": "REVOKE", "M2": "REVISE", "M3": "KEEP", "M4": "KEEP"}


def test_lambda_c_grid_offline(val_env):
    out = fit.tune("t", _grid(lambda_c=[0.1, 0.5], lambda_s=[0.0], lambda_e=[0.0]), offline_only=True,
                   out=val_env / "tuned.yaml")
    g = out["grid"]
    assert len(g) == 2 and out["filled"] == 0 and g["chosen"].sum() == 1
    # lc .1: R4 (1 - .085) beats R2 (.9 - .03) -> M3 over-invalidated: F1 = 4/5, OI = 1/4 (gold KEEP: v1 M3 M4, v2 M1 M2)
    lo = g[g.lambda_c == 0.1].iloc[0]
    assert lo.repair_f1 == pytest.approx(0.8) and lo.over_invalidation == pytest.approx(0.25)
    assert lo.objective == pytest.approx(0.55)
    # lc .5: R2 -> perfect
    assert out["tuned"] == {"tau_d": 0.5, "tau_e": 0.6, "lambda_c": 0.5, "lambda_s": 0.0, "lambda_e": 0.0}
    assert out["objective"] == pytest.approx(1.0)
    assert yaml.safe_load((val_env / "tuned.yaml").read_text()) == out["tuned"]
    assert (val_env / "results/fit/t/grid.csv").read_text().count("\n") == 3


def test_tune_refuses_a_config_changed_since_the_val_run(val_env, monkeypatch):
    assert run_all.config_hash({**CFG, "fit": {}}) == run_all.config_hash(CFG)  # grids may change freely
    monkeypatch.setattr(fit, "load_config", lambda overrides=None, overlay=None: {**CFG, "tie_eps": 0.5})
    with pytest.raises(SystemExit, match="changed since the val run"):
        fit.tune("t", _grid(), out=val_env / "tuned.yaml")


def test_tune_rebuilds_the_val_config_from_the_recorded_overlay(val_env):
    # val run made with an overlay lambda_c .1 (R4 wins, see above); tune gets no overlay flag of its own
    ov = val_env / "ov.yaml"
    ov.write_text("lambda_c: 0.1\nlambda_s: 0.0\nlambda_e: 0.0\n")
    for sid in ("v1", "v2"):
        rec_path = val_env / f"results/raw/t/val/csmr/{sid}.json"
        _raw(val_env, sid, Trace(**{k: v for k, v in json.loads(rec_path.read_text())["trace"].items()}), str(ov))
    rec = json.loads((val_env / "results/raw/t/val/csmr/v1.json").read_text())
    s = fit.load_saved(rec, val_env / "val/v1.json")
    base = load_config(overlay=str(ov))
    assert fit.choose(s, fit.feasible_sets(s, base, False, {"filled": 0}), base)["M3"] == "REVISE"
    fit.tune("t", _grid(), offline_only=True, out=val_env / "tuned.yaml")  # hash check passes with the overlay
    ov.write_text("lambda_c: 0.25\n")
    with pytest.raises(SystemExit, match="overlay"):
        fit.tune("t", _grid(), offline_only=True, out=val_env / "tuned.yaml")


def test_tau_change_needs_online_benefit_once(val_env, monkeypatch):
    # tau_e .7: M2 (S .65) becomes VERIFY -> R2 = {M1 REVOKE, M2 VERIFY}, never scored by the val run; R4 infeasible
    cfg = _grid(tau_e=[0.7], lambda_c=[0.1, 0.5])
    with pytest.raises(SystemExit, match="no grid point"):
        fit.tune("t", cfg, offline_only=True, out=val_env / "tuned.yaml")
    filled = []
    monkeypatch.setattr(fit, "fill_benefit", lambda s, rs, c: filled.append(rs.key()) or 0.95)
    out = fit.tune("t", cfg, out=val_env / "tuned.yaml")
    assert filled == [(("M1", "REVOKE"), ("M2", "VERIFY"))] and out["filled"] == 1  # memoised across lambdas
    assert out["grid"]["n_scored"].tolist() == [2, 2]


def test_fill_benefit_answers_saved_probes_over_the_virtual_view(val_env, monkeypatch):
    rec = json.loads((val_env / "results/raw/t/val/csmr/v1.json").read_text())
    s = fit.load_saved(rec, val_env / "val/v1.json")
    seen = []
    monkeypatch.setattr(cf, "answer", lambda q, mems, cfg=None: seen.append([m.text for m in mems]) or "ans")
    monkeypatch.setattr(cf, "judge", lambda a, q, d, o, cfg, src="": {"score": 0.5 if "have" in q else 1.0})
    b = fit.fill_benefit(s, RepairSet("R2", {"M1": Action.REVOKE, "M2": Action.VERIFY}), CFG)
    assert b == pytest.approx(0.75)
    # M1 revoked, M2 unverified, observation memory left out (probe_exclude_observation)
    assert seen[0] == ["[unverified] memory M2", "memory M3", "memory M4"]
