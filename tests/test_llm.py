import json

import diskcache
import pytest

from csmr import llm


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "_cache", diskcache.Cache(str(tmp_path)))
    llm.reset_stats()
    replies, seen = [], []

    def _complete(model, system, prompt, json_mode, max_tokens, cfg):
        seen.append(prompt)
        return replies.pop(0)

    monkeypatch.setattr(llm, "_complete", _complete)
    return replies, seen


def test_cache_hit_after_miss(fake):
    replies, seen = fake
    replies.append("hello")
    assert llm.call("hi") == llm.call("hi") == "hello"
    assert len(seen) == 1
    assert llm.stats() == {"calls": 2, "hits": 1, "misses": 1, "fallbacks": {}}


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


class BadRequestError(Exception):  # matched by class name, like RateLimitError in _complete
    pass


def test_provider_json_error_is_invalid_json_and_retried(fake, monkeypatch):
    seen = []

    def boom_then_ok(model, system, prompt, json_mode, max_tokens, cfg):
        seen.append(prompt)
        if len(seen) == 1:
            raise BadRequestError("Error code: 400 - {'code': 'json_validate_failed'}")
        return json.dumps({"a": 1})

    monkeypatch.setattr(llm, "_complete", boom_then_ok)
    assert llm.try_json("q") == {"a": 1} and len(seen) == 2 and seen[1].endswith("Return valid JSON only.")


def test_provider_json_error_twice_gives_none_and_fallback_applies(fake, monkeypatch):
    def boom(*a):
        raise BadRequestError("response_format: json_object failed")

    monkeypatch.setattr(llm, "_complete", boom)
    assert llm.try_json("q") is None
    assert llm.try_json_valid("q2", lambda o: True) is None


def test_unrelated_bad_request_still_raises(fake, monkeypatch):
    def boom(*a):
        raise BadRequestError("context length exceeded")

    monkeypatch.setattr(llm, "_complete", boom)
    with pytest.raises(BadRequestError):
        llm.try_json("q")


def test_defaults_come_from_config(fake, monkeypatch):
    from csmr.config import load_config
    got = []
    monkeypatch.setattr(llm, "_complete", lambda m, s, p, j, mt, cfg: got.append(mt) or "x")
    llm.call("p")
    assert got == [load_config()["llm"]["max_tokens"]]
