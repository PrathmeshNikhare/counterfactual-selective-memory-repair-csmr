import numpy as np
import pytest

from csmr.models import embed, nli, nli_batch


@pytest.mark.xfail(strict=True, reason="first-person premise; handled by third-person statement from Phase A")
def test_nli_first_person_premise():
    assert nli("I sold my car.", "The user owns a car.")["contradiction"] > 0.5


def test_nli_probs_sum_to_one():
    assert abs(sum(nli("A man is eating pizza.", "A man is sleeping.").values()) - 1) < 1e-4


def test_nli_contradiction():
    assert nli("The user owns a car.", "The user does not own a car.")["contradiction"] > 0.5


def test_nli_entailment():
    r = nli("The user owns a car.", "The user has a vehicle.")
    assert r["entailment"] > r["contradiction"]


def test_nli_batch_matches_single():
    pairs = [("I sold my car.", "The user owns a car."), ("I own a dog.", "The user has a pet.")]
    out = nli_batch(pairs)
    assert out[0] == nli(*pairs[0])
    assert out[1]["entailment"] > out[1]["contradiction"]


def test_embed_normalised_and_cached():
    v = embed(["a car", "a bike", "a car"])
    assert v.shape[0] == 3 and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4)
    assert np.allclose(v[0], v[2])
    assert embed([]).shape[0] == 0
