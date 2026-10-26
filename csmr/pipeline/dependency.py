"""Phase C (dependency) — P(D=1 | z) via logistic regression (Paper IV-C)."""
from __future__ import annotations

import json
import math

import numpy as np

from csmr import llm, models
from csmr.adapters.base import MemoryStoreAdapter
from csmr.config import ROOT, load_config
from csmr.types import CandidateInfo, Memory, Observation, StateChange, same_entity

CAUSAL_PROMPT = """Stored fact A: {a}
It changed: {change}
Is statement B now likely FALSE or OUT OF DATE because of this change? Answer one word: yes, maybe, or no.
B: {b}
Return JSON only: {{"answer": "yes" | "maybe" | "no"}}"""
CUE_SCORE = {"y": 1.0, "m": 0.5, "n": 0.0}  # yes / maybe / no, keyed by first letter

FEATURES = ["cos_sim", "same_entity", "same_attribute", "nli_entail", "nli_contra_topical", "temporal_after",
            "same_source", "causal_cue", "hop_closeness"]
_MODEL_FILE = ROOT / "models" / "dependency.json"


def _weights(cfg: dict | None = None) -> dict[str, float]:
    """Fitted coefficients if models/dependency.json exists, else hand-set config weights."""
    if _MODEL_FILE.exists():
        return json.loads(_MODEL_FILE.read_text())
    return (cfg or load_config())["dependency_weights"]


def dependency_features(root: Memory, cand: Memory, delta: StateChange, obs: Observation,
                        info: CandidateInfo, cfg: dict | None = None) -> dict[str, float]:
    """Paper IV-C feature vector z for the pair (root, cand); all values in [0, 1]."""
    cfg = cfg or load_config()
    a, b = models.embed([root.text, cand.text])
    stmt = delta.statement or delta.as_sentence()
    cue = llm.try_json_valid(CAUSAL_PROMPT.format(a=root.text, change=stmt, b=cand.text),
                             lambda o: str(o.get("answer", "")).strip().lower()[:1] in CUE_SCORE,
                             max_tokens=cfg["cue_max_tokens"], cfg=cfg)
    if cue is None:
        llm.fallback("causal_cue", f"malformed response for {cand.id}; cue = no")
        cue = {"answer": "no"}
    cos = max(0.0, float(a @ b))
    return {
        "cos_sim": cos,
        "same_entity": float(same_entity(root.entity, cand.entity)),
        "same_attribute": float(root.attribute == cand.attribute),
        "nli_entail": models.nli(root.text, cand.text)["entailment"],
        # NLI calls unrelated sentences "contradictions"; only meaningful for topically related pairs
        "nli_contra_topical": models.nli(stmt, cand.text)["contradiction"] * cos,
        "temporal_after": float(cand.time > root.time),
        "same_source": float(root.source == cand.source),
        "causal_cue": CUE_SCORE[str(cue["answer"]).strip().lower()[0]],
        "hop_closeness": cfg["hop_closeness"].get(info.hop, cfg["hop_default"]),
    }


def dependency_prob(features: dict[str, float], cfg: dict | None = None) -> float:
    w = _weights(cfg)
    z = w["intercept"] + sum(w[k] * v for k, v in features.items())
    return 1.0 / (1.0 + math.exp(-z))


def assess_dependency(cands: dict[str, CandidateInfo], store: MemoryStoreAdapter, delta: StateChange,
                      obs: Observation, cfg: dict | None = None) -> None:
    """Fill dep_features / p_dep on every candidate in place. Roots get 1.0; others max over roots."""
    mems = {m.id: m for m in store.all()}
    roots = [i for i, c in cands.items() if c.is_root]
    cfg = cfg or load_config()
    use = cfg["ablation"]["use_dependency"]
    for mid, info in cands.items():
        if info.is_root or not use:  # use_dependency=false: trivial default P = 1, no NLI/LLM calls
            info.p_dep = 1.0
            continue
        best = None
        for r in roots:
            f = dependency_features(mems[r], mems[mid], delta, obs, info, cfg)
            p = dependency_prob(f, cfg)
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
