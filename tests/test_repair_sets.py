from csmr.config import load_config
from csmr.pipeline.repair_sets import generate_repair_sets
from csmr.types import Action, CandidateInfo

CFG = load_config()


def _c(mid, hop, p, s, root=False):
    return CandidateInfo(memory_id=mid, hop=hop, retrieved_by="root" if root else "hop", is_root=root, p_dep=p, s_evidence=s)


def _sets(cands, proposed):
    return {r.name: r for r in generate_repair_sets(cands, proposed, CFG)}


def test_nested_sets_and_dedup():
    cands = {"a": _c("a", 0, 1, 0.9, True), "b": _c("b", 1, 0.8, 0.9), "c": _c("c", 2, 0.6, 0.9), "d": _c("d", 1, 0.0, 0.9)}
    prop = {"a": Action.KEEP, "b": Action.VERIFY, "c": Action.REVISE}
    s = _sets(cands, prop)
    assert s["R0"].actions == {}
    assert s["R1"].actions == {"a": Action.REVISE}  # KEEP root -> REVISE
    assert set(s["R2"].actions) == {"a", "b"} and set(s["R3"].actions) == {"a", "b", "c"}
    assert s["R4"].actions == {"a": Action.REVISE, "b": Action.REVISE, "c": Action.REVISE}  # d has P < 0.2


def test_dedup_keeps_lowest_name():
    cands = {"a": _c("a", 0, 1, 0.9, True)}
    assert [r.name for r in generate_repair_sets(cands, {"a": Action.REVISE}, CFG)] == ["R0", "R1"]


def test_r4_infeasible_when_member_below_tau_e():
    cands = {"a": _c("a", 0, 1, 0.9, True), "b": _c("b", 1, 0.5, 0.2)}
    s = _sets(cands, {"a": Action.REVISE, "b": Action.VERIFY})
    assert s["R2"].feasible and not s["R4"].feasible and "b" in s["R4"].infeasible_reason
