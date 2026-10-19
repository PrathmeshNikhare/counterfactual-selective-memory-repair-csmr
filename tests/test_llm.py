import json

import diskcache
import pytest

from csmr import llm


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "_cache", diskcache.Cache(str(tmp_path)))
    llm.reset_stats()
    replies, seen = [], []

    def _complete(model, system, prompt, json_mode, max_tokens):
        seen.append(prompt)
        return replies.pop(0)

    monkeypatch.setattr(llm, "_complete", _complete)
    return replies, seen


def test_cache_hit_after_miss(fake):
    replies, seen = fake
    replies.append("hello")
    assert llm.call("hi") == llm.call("hi") == "hello"
    assert len(seen) == 1
    assert llm.stats() == {"calls": 2, "hits": 1, "misses": 1}


def test_bad_json_retried_once(fake):
    replies, seen = fake
    replies += ["not json", json.dumps({"a": 1})]
    assert llm.call("q", json_mode=True) == {"a": 1}
    assert seen[1].endswith("Return valid JSON only.")


def test_bad_json_twice_raises_and_is_not_cached(fake):
    replies, _ = fake
    replies += ["x", "y"]
    with pytest.raises(json.JSONDecodeError):
        llm.call("q", json_mode=True)
    assert llm.stats()["misses"] == 1 and len(llm._cache) == 0
