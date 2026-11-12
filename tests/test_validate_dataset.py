"""eval/validate_dataset.py: the committed seeds pass, and the scenario checks catch each rule (embeddings stubbed)."""
import copy
import json
import zlib
from collections import Counter

import numpy as np
import pytest

from csmr.config import ROOT, load_config
from eval import validate_dataset as vd

LIM = load_config()["dataset"]


@pytest.fixture(autouse=True)
def fake_embed(monkeypatch):
    def emb(texts):  # deterministic near-orthogonal unit vectors; identical texts collide
        v = np.stack([np.random.default_rng(zlib.crc32(t.encode())).normal(size=256) for t in texts]) if texts \
            else np.zeros((0, 256))
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    monkeypatch.setattr(vd, "embed", emb)


def test_seeds_valid_and_split():
    errs, items, split_of = vd.check_seeds(ROOT / "data/seeds")
    assert errs == [] and vd.near_dups(items, vd.reference_texts(), LIM["near_dup_cosine"]) == []
    assert Counter(split_of.values()) == {"test": 18, "val": 8, "train": 34}


def test_stale_review_fails(tmp_path):
    stale = tmp_path / "seed_review.md"
    stale.write_text("old", encoding="utf-8")
    assert any("stale" in e for e in vd.check_seeds(ROOT / "data/seeds", stale)[0])


def test_scenario_counts_per_family():
    split_of = {"a": "test", "b": "test", "c": "val"}
    items = [{"split": "test", "family": "a"}] * 2 + [{"split": "test", "family": "b"}]
    # b is short; val has no scenarios yet, so it is not checked
    assert vd.check_counts(items, split_of, {"test": 2, "val": 13}) == ["test/b: 1 scenarios, need 2"]


def _scenario(fam):
    """A scenario built from a seed family: core memories + fillers to 30, 3 persistence memories."""
    mems = [{k: m[k] for k in ("id", "text", "entity", "attribute", "source")} | {"time": "2026-01-01T10:00:00"}
            for m in fam["core_memories"]]
    mems += [{"id": f"M{i}", "text": f"Filler fact number {i}.", "entity": "user", "attribute": f"a{i}",
              "time": "2026-01-02T10:00:00", "source": "user"} for i in range(len(mems) + 1, 31)]
    core = {m["id"]: m["gold_action"] for m in fam["core_memories"]}
    actions = {m["id"]: core.get(m["id"], "KEEP") for m in mems}
    pers = [{"id": f"P{i}", "text": f"Later fact {i}.", "entity": "user", "attribute": f"p{i}",
             "time": "2026-07-01T10:00:00", "source": "user"} for i in range(1, 4)]
    return {"id": "s1", "category": fam["category"], "domain": fam["domain"], "family": fam["family_id"],
            "memories": mems, "observation": fam["observation"] | {"time": "2026-06-01T10:00:00"},
            "gold": {"actions": actions, "dependencies": fam["dependencies"],
                     "eval_questions": fam["eval_questions"], "persistence_memories": pers}}


def _break(d, how):
    d = copy.deepcopy(d)
    g = d["gold"]
    if how == "action":
        g["actions"]["M1"] = "REVOKE/REVISE"
    elif how == "missing_label":
        del g["actions"]["M2"]
    elif how == "dep":
        g["dependencies"].append(["M1", "M99"])
    elif how == "depends_on":
        g["eval_questions"][0]["depends_on"] = []
    elif how == "pers_time":
        g["persistence_memories"][0]["time"] = "2026-05-01T10:00:00"
    elif how == "too_few":
        d["memories"] = d["memories"][:29]
        g["actions"] = {m["id"]: g["actions"][m["id"]] for m in d["memories"]}
    elif how == "family":
        d["family"] = "one_hop__vehicle"  # a train family, file sits in test/
    elif how == "pending":
        g["eval_questions"][0]["pending"] = True  # two_hop__vehicle M3 is REVISE, not VERIFY
    elif how == "convention":
        g["actions"]["M2"] = "CONDITIONAL"  # two_hop scenario
    return d


@pytest.mark.parametrize("how,msg", [("action", "invalid action"), ("missing_label", "do not cover"),
                                     ("dep", "not a pair of known ids"), ("depends_on", "depends_on"),
                                     ("pers_time", "after the observation"), ("too_few", "29 memories"),
                                     ("family", "is in split"), ("convention", "CONDITIONAL only"),
                                     ("pending", "pending: true iff")])
def test_scenario_checks(tmp_path, how, msg):
    split_of = vd.check_seeds(ROOT / "data/seeds")[2]
    fams = {f["family_id"]: f for f in json.loads((ROOT / "data/seeds/seeds.json").read_text("utf-8"))["families"]}
    good = _scenario(fams["two_hop__vehicle"])  # a test-split family
    path = tmp_path / "test" / "s1.json"
    path.parent.mkdir()
    path.write_text(json.dumps(good), encoding="utf-8")
    assert vd.check_scenario(path, "test", split_of, LIM)[0] == []
    path.write_text(json.dumps(_break(good, how)), encoding="utf-8")
    assert any(msg in e for e in vd.check_scenario(path, "test", split_of, LIM)[0])
