"""Phase C (dependency) — P(D=1 | z) via logistic regression (Paper IV-C)."""
from __future__ import annotations

import json
import math

import numpy as np

from csmr import llm, models
from csmr.adapters.base import MemoryStoreAdapter
from csmr.config import ROOT, load_config
from csmr.types import CandidateInfo, Memory, Observation, StateChange

CAUSAL_PROMPT = """Does statement B only make sense / stay true if statement A is true?
A: {a}
B: {b}
Answer yes or no. Return JSON only: {{"answer": "yes" or "no"}}"""

FEATURES = ["cos_sim", "same_entity", "same_attribute", "nli_entail", "nli_contra_new", "temporal_after",
            "same_source", "causal_cue", "hop_closeness"]
_MODEL_FILE = ROOT / "models" / "dependency.json"


def _weights() -> dict[str, float]:
    """Fitted coefficients if models/dependency.json exists, else hand-set config weights."""
    if _MODEL_FILE.exists():
        return json.loads(_MODEL_FILE.read_text())
    return load_config()["dependency_weights"]


def dependency_features(root: Memory, cand: Memory, delta: StateChange, obs: Observation,
                        info: CandidateInfo) -> dict[str, float]:
    """Paper IV-C feature vector z for the pair (root, cand); all values in [0, 1]."""
    a, b = models.embed([root.text, cand.text])
    cue = llm.call(CAUSAL_PROMPT.format(a=root.text, b=cand.text), json_mode=True)
    return {
        "cos_sim": max(0.0, float(a @ b)),
        "same_entity": float(root.entity == cand.entity),
        "same_attribute": float(root.attribute == cand.attribute),
        "nli_entail": models.nli(root.text, cand.text)["entailment"],
        "nli_contra_new": models.nli(delta.as_sentence(), cand.text)["contradiction"],
        "temporal_after": float(cand.time > root.time),
        "same_source": float(root.source == cand.source),
        "causal_cue": float(str(cue.get("answer", "")).strip().lower().startswith("y")),
        "hop_closeness": {0: 1.0, 1: 0.6}.get(info.hop, 0.3),
    }


def dependency_prob(features: dict[str, float]) -> float:
    w = _weights()
    z = w["intercept"] + sum(w[k] * v for k, v in features.items())
    return 1.0 / (1.0 + math.exp(-z))


def assess_dependency(cands: dict[str, CandidateInfo], store: MemoryStoreAdapter, delta: StateChange,
                      obs: Observation) -> None:
    """Fill dep_features / p_dep on every candidate in place. Roots get 1.0; others max over roots."""
    mems = {m.id: m for m in store.all()}
    roots = [i for i, c in cands.items() if c.is_root]
    for mid, info in cands.items():
        if info.is_root:
            info.p_dep = 1.0
            continue
        best = None
        for r in roots:
            f = dependency_features(mems[r], mems[mid], delta, obs, info)
            p = dependency_prob(f)
            if best is None or p > best[0]:
                best = (p, f)
        if best:
            info.p_dep, info.dep_features = best


def fit_dependency(X: list[dict[str, float]], y: list[int]) -> None:
    """Train on labelled pairs (gold dependencies; training code only) and save to models/dependency.json."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000).fit(np.array([[x[k] for k in FEATURES] for x in X]), y)
    _MODEL_FILE.parent.mkdir(exist_ok=True)
    _MODEL_FILE.write_text(json.dumps({"intercept": float(clf.intercept_[0]),
                                       **dict(zip(FEATURES, map(float, clf.coef_[0])))}, indent=2))
