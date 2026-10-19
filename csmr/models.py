"""Lazy embedder + NLI singletons (Spec 1.3). Model names come from config.yaml."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from csmr.config import load_config

_NLI_KEYS = ("contradiction", "entailment", "neutral")
_emb_cache: dict[str, np.ndarray] = {}
_nli_cache: dict[tuple[str, str], dict[str, float]] = {}


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(load_config()["models"]["embedder"])


@lru_cache(maxsize=1)
def _nli_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(load_config()["models"]["nli"])


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised embeddings, shape (len(texts), d); cached by text."""
    todo = list(dict.fromkeys(t for t in texts if t not in _emb_cache))
    if todo:
        for t, v in zip(todo, _embedder().encode(todo, normalize_embeddings=True)):
            _emb_cache[t] = v
    if not texts:
        return np.zeros((0, _embedder().get_sentence_embedding_dimension()))
    return np.stack([_emb_cache[t] for t in texts])


def nli_batch(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
    """Softmax NLI probs for (premise, hypothesis) pairs; cached by pair."""
    todo = list(dict.fromkeys(p for p in pairs if p not in _nli_cache))
    if todo:
        model = _nli_model()
        # id2label order differs between checkpoints, so map by name
        labels = [model.model.config.id2label[i].lower() for i in range(len(model.model.config.id2label))]
        logits = np.asarray(model.predict(todo, apply_softmax=False))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        for p, row in zip(todo, probs):
            d = dict(zip(labels, map(float, row)))
            _nli_cache[p] = {k: d[k] for k in _NLI_KEYS}
    return [dict(_nli_cache[p]) for p in pairs]


def nli(premise: str, hypothesis: str) -> dict[str, float]:
    return nli_batch([(premise, hypothesis)])[0]
