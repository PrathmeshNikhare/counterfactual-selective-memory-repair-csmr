"""Phase C (evidence) — S(E, m) via logistic regression (Paper IV-C)."""
from __future__ import annotations

import json
import math
import re

import numpy as np

from csmr import models
from csmr.adapters.base import MemoryStoreAdapter
from csmr.config import ROOT, load_config
from csmr.types import CandidateInfo, Memory, Observation, StateChange

FEATURES = ["source_reliability", "directness", "contradiction_strength", "not_hedged", "not_temporary",
            "corroboration", "extraction_conf", "freshness"]
_MODEL_FILE = ROOT / "models" / "evidence.json"


def _weights() -> dict[str, float]:
    """Fitted coefficients if models/evidence.json exists, else hand-set config weights."""
    if _MODEL_FILE.exists():
        return json.loads(_MODEL_FILE.read_text())
    return load_config()["evidence_weights"]


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def evidence_features(cand: Memory, delta: StateChange, obs: Observation, info: CandidateInfo,
                      corroboration: int = 1) -> dict[str, float]:
    """Paper IV-C feature vector for (observation, cand); also sets info.hedged / info.temporary."""
    cfg = load_config()
    info.hedged = _matches(cfg["hedge_patterns"], obs.text)
    info.temporary = _matches(cfg["temporary_patterns"], obs.text)
    return {
        "source_reliability": cfg["source_reliability"].get(delta.source, cfg["source_reliability_default"]),
        "directness": cfg["directness"].get(info.hop, cfg["hop_default"]),
        "contradiction_strength": models.nli(delta.statement or delta.as_sentence(), cand.text)["contradiction"],
        "not_hedged": float(not info.hedged),
        "not_temporary": float(not info.temporary),
        "corroboration": min(corroboration, cfg["corroboration_cap"]) / cfg["corroboration_cap"],
        "extraction_conf": delta.confidence,
        "freshness": 1.0 if obs.time >= cand.time else 0.5,
    }


def evidence_score(features: dict[str, float]) -> float:
    """S(E, m)."""
    w = _weights()
    return 1.0 / (1.0 + math.exp(-(w["intercept"] + sum(w[k] * v for k, v in features.items()))))


def assess_evidence(cands: dict[str, CandidateInfo], store: MemoryStoreAdapter, delta: StateChange,
                    obs: Observation) -> None:
    """Fill ev_features / s_evidence / hedged / temporary on every candidate in place.
    ablation.use_evidence=false swaps the stage for its trivial default S = 1."""
    use = load_config()["ablation"]["use_evidence"]
    mems = {m.id: m for m in store.all()}
    for mid, info in cands.items():
        info.ev_features = evidence_features(mems[mid], delta, obs, info)
        info.s_evidence = evidence_score(info.ev_features) if use else 1.0


def fit_evidence(X: list[dict[str, float]], y: list[int]) -> None:
    """Train on labelled pairs (gold actions; training code only) and save to models/evidence.json."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000).fit(np.array([[x[k] for k in FEATURES] for x in X]), y)
    _MODEL_FILE.parent.mkdir(exist_ok=True)
    _MODEL_FILE.write_text(json.dumps({"intercept": float(clf.intercept_[0]),
                                       **dict(zip(FEATURES, map(float, clf.coef_[0])))}, indent=2))
