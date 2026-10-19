"""Cached LLM wrapper (CLAUDE.md rule 5; Spec 1.2). Every LLM call in the repo goes through `call`."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time

import diskcache
from dotenv import load_dotenv

from csmr.config import ROOT, load_config

_stats = {"calls": 0, "hits": 0, "misses": 0}
_fallbacks: dict[str, int] = {}  # malformed-LLM-output fallbacks by kind, reported in the trace
log = logging.getLogger(__name__)
_cache: diskcache.Cache | None = None
_client = None
_JSON_NUDGE = "Return valid JSON only."


def stats() -> dict:
    return {**_stats, "fallbacks": dict(_fallbacks)}


def reset_stats() -> None:
    for k in _stats:
        _stats[k] = 0
    _fallbacks.clear()


def fallback(kind: str, detail: str = "") -> None:
    """Record that a malformed LLM response was replaced by a safe default (counted in Trace.llm_stats)."""
    _fallbacks[kind] = _fallbacks.get(kind, 0) + 1
    log.warning("LLM fallback [%s] %s", kind, detail)


def _get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        d = load_config()["llm"]["cache_dir"]
        _cache = diskcache.Cache(str(ROOT / d))
    return _cache


def _complete(model: str, system: str | None, prompt: str, json_mode: bool, max_tokens: int) -> str:
    """The only function that touches the API (tests stub this)."""
    global _client
    cfg = load_config()["llm"]
    if _client is None:
        from openai import OpenAI

        load_dotenv(ROOT / ".env")
        _client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL"))
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    for attempt in range(cfg["max_retries"] + 1):
        try:
            r = _client.chat.completions.create(
                model=model, messages=messages, temperature=cfg["temperature"], max_tokens=max_tokens, **kw
            )
            return r.choices[0].message.content or ""
        except Exception as e:  # openai.RateLimitError et al.; only back off on rate limits
            if type(e).__name__ != "RateLimitError" or attempt == cfg["max_retries"]:
                raise
            time.sleep(2**attempt)


def call(prompt: str, *, system: str | None = None, json_mode: bool = False,
         model: str | None = None, max_tokens: int = 512) -> str | dict:
    model = model or load_config()["llm"]["model"]
    key = hashlib.sha256(json.dumps([model, system, prompt, json_mode]).encode()).hexdigest()
    cache = _get_cache()
    _stats["calls"] += 1
    if key in cache:
        _stats["hits"] += 1
        return cache[key]
    _stats["misses"] += 1
    out = _complete(model, system, prompt, json_mode, max_tokens)
    if json_mode:
        try:
            out = json.loads(out)
        except json.JSONDecodeError:
            out = json.loads(_complete(model, system, f"{prompt}\n\n{_JSON_NUDGE}", json_mode, max_tokens))
    cache[key] = out  # only successes are cached
    return out


def try_json(prompt: str, **kw) -> dict | None:
    """`call(json_mode=True)` that returns None instead of raising on undecodable or non-object JSON."""
    try:
        out = call(prompt, json_mode=True, **kw)
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None
