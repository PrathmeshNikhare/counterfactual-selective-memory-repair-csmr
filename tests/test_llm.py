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


class RateLimitError(Exception):  # matched by class name, like the real openai one
    def __init__(self, msg, retry_after=None):
        super().__init__(msg)
        self.response = type("R", (), {"headers": {"retry-after": retry_after} if retry_after else {}})()


def _client(errors):
    """Fake OpenAI client that raises `errors` in order, then answers 'ok'."""
    def create(**kw):
        if errors:
            raise errors.pop(0)
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]})()
    return type("C", (), {"chat": type("H", (), {"completions": type("P", (), {"create": staticmethod(create)})()})()})()


def test_retry_after_is_honoured_and_capped(monkeypatch):
    from csmr.config import load_config
    cfg = load_config()
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    monkeypatch.setattr(llm, "_client", _client([RateLimitError("slow down", "7.5"), RateLimitError("slow down", "99999"),
                                                RateLimitError("slow down")]))
    assert llm._complete("m", None, "p", False, 8, cfg) == "ok"
    assert sleeps == [7.5, cfg["llm"]["max_retry_wait_s"], cfg["llm"]["backoff_base_s"] * 2**2]


def test_daily_quota_raises_clear_error_without_retrying(monkeypatch):
    from csmr.config import load_config
    errs = [RateLimitError("Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199578", "239")]
    monkeypatch.setattr(llm.time, "sleep", lambda s: pytest.fail("must not wait on a daily quota"))
    monkeypatch.setattr(llm, "_client", _client(errs))
    with pytest.raises(llm.QuotaExhausted, match="Daily LLM quota exhausted - resume later; cached calls are kept"):
        llm._complete("m", None, "p", False, 8, load_config())
