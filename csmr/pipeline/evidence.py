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


def _weights(cfg: dict | None = None) -> dict[str, float]:
    """Fitted coefficients if models/evidence.json exists, else hand-set config weights."""
    if _MODEL_FILE.exists():
        return json.loads(_MODEL_FILE.read_text())
    return (cfg or load_config())["evidence_weights"]


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def evidence_features(cand: Memory, delta: StateChange, obs: Observation, info: CandidateInfo,
                      corroboration: int = 1, cfg: dict | None = None,
                      roots: list[Memory] | None = None) -> dict[str, float]:
    """Paper IV-C feature vector for (observation, cand); also sets info.hedged / info.temporary.
    A dependent is not contradicted by the change itself, its root is: so for a non-root, contradiction_strength
    = max over roots of NLI(premise root.text, hypothesis Δ.statement) (this direction: the small NLI model reads
    "sold their car" as entailing "owns a car"); directness still discounts it by hop."""
    cfg = cfg or load_config()
    info.hedged = _matches(cfg["hedge_patterns"], obs.text)
    info.temporary = delta.temporary or _matches(cfg["temporary_patterns"], obs.text)
    stmt = delta.statement or delta.as_sentence()
    if info.is_root or not roots:
        contra = models.nli(stmt, cand.text)["contradiction"]
    else:
        contra = max(models.nli(r.text, stmt)["contradiction"] for r in roots)
    return {
        "source_reliability": cfg["source_reliability"].get(delta.source, cfg["source_reliability_default"]),
        "directness": cfg["directness"].get(info.hop, cfg["hop_default"]),
        "contradiction_strength": contra,
        "not_hedged": float(not info.hedged),
        "not_temporary": float(not info.temporary),
        "corroboration": min(corroboration, cfg["corroboration_cap"]) / cfg["corroboration_cap"],
        "extraction_conf": delta.confidence,
        "freshness": 1.0 if obs.time >= cand.time else 0.5,
    }


def evidence_score(features: dict[str, float], cfg: dict | None = None) -> float:
    """S(E, m)."""
    w = _weights(cfg)
    return 1.0 / (1.0 + math.exp(-(w["intercept"] + sum(w[k] * v for k, v in features.items()))))


def assess_evidence(cands: dict[str, CandidateInfo], store: MemoryStoreAdapter, delta: StateChange,
                    obs: Observation, cfg: dict | None = None) -> None:
    """Fill ev_features / s_evidence / hedged / temporary on every candidate in place.
    ablation.use_evidence=false swaps the stage for its trivial default S = 1."""
    cfg = cfg or load_config()
    use = cfg["ablation"]["use_evidence"]
    mems = {m.id: m for m in store.all()}
    roots = [mems[i] for i, c in cands.items() if c.is_root]
    for mid, info in cands.items():
        info.ev_features = evidence_features(mems[mid], delta, obs, info, cfg=cfg, roots=roots)
        info.s_evidence = evidence_score(info.ev_features, cfg) if use else 1.0


def fit_evidence(X: list[dict[str, float]], y: list[int]) -> None:
    """Train on labelled pairs (gold actions; training code only) and save to models/evidence.json."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000).fit(np.array([[x[k] for k in FEATURES] for x in X]), y)
    _MODEL_FILE.parent.mkdir(exist_ok=True)
    _MODEL_FILE.write_text(json.dumps({"intercept": float(clf.intercept_[0]),
                                       **dict(zip(FEATURES, map(float, clf.coef_[0])))}, indent=2))
