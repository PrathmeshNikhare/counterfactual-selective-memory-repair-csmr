"""eval/run_all.py + eval/aggregate.py, fully stubbed: fake methods, fake metrics, results in tmp_path."""
import json

import pytest

from csmr import llm
from csmr.config import load_config
from csmr.types import Trace
from eval import aggregate, run_all

CFG = load_config()
DEV = run_all.scenario_paths("dev")


@pytest.fixture
def env(monkeypatch, tmp_path):
    calls = []

    def fake(obs, store, cfg):
        calls.append(obs.text)
        return Trace(observation={}, final_actions={m.id: "KEEP" for m in store.all()}, llm_stats={"calls": 1})

    monkeypatch.setattr(run_all, "RESULTS", tmp_path)
    monkeypatch.setattr(aggregate, "RESULTS", tmp_path)
    monkeypatch.setattr(run_all, "METHODS", {"a": fake, "b": fake})
    monkeypatch.setattr(run_all, "scenario_metrics", lambda rec, cfg: {"action_accuracy": (1, 2), "llm_calls": (1, 1)})
    return calls, tmp_path


def _go(shard="1/1"):
    return run_all.run("dev", ["a", "b"], "t", shard, CFG, [])


def test_shards_partition_scenarios_exactly():
    for n in (1, 2, 3, len(DEV) + 2):
        parts = [run_all.shard(DEV, f"{i}/{n}") for i in range(1, n + 1)]
        flat = [p for part in parts for p in part]
        assert sorted(flat) == DEV and len(flat) == len(DEV)
    with pytest.raises(ValueError):
        run_all.shard(DEV, "0/2")


def test_resume_skips_done_files(env):
    calls, tmp = env
    assert _go() == 2 * len(DEV)
    rec = json.loads(next((tmp / "raw/t/dev/a").glob("*.json")).read_text())
    assert {"final_actions", "trace", "metrics", "config_hash", "git_commit", "llm_stats"} <= rec.keys()
    calls.clear()
    assert _go() == 0 and calls == []


def test_errors_logged_and_run_continues(env, monkeypatch):
    calls, tmp = env
    n = []

    def flaky(obs, store, cfg):  # fails on its first call only
        n.append(1)
        if len(n) == 1:
            raise RuntimeError("boom")
        return Trace(observation={})

    monkeypatch.setitem(run_all.METHODS, "a", flaky)
    assert _go() == 2 * len(DEV) - 1
    errs = [json.loads(l) for l in (tmp / "raw/t/errors.jsonl").read_text().splitlines()]
    assert len(errs) == 1 and errs[0]["method"] == "a" and "boom" in errs[0]["error"]


def test_quota_stops_shard_cleanly(env, monkeypatch):
    calls, tmp = env

    def quota(obs, store, cfg):
        raise llm.QuotaExhausted(llm.QUOTA_MSG)

    monkeypatch.setitem(run_all.METHODS, "b", quota)
    with pytest.raises(SystemExit, match="rerun the same command"):
        _go()
    assert len(list((tmp / "raw/t/dev/a").glob("*.json"))) == 1  # first scenario's "a" finished, then stopped
    assert not (tmp / "raw/t/errors.jsonl").exists()


def test_aggregate_writes_tables_and_refuses_hash_mismatch(env):
    calls, tmp = env
    _go()
    aggregate.main(["--tag", "t", "--split", "dev"])
    main = (tmp / "t/main.csv").read_text().splitlines()
    assert main[0].startswith("method,n_scenarios,action_accuracy") and len(main) == 3
    assert {"per_category.csv", "persistence.csv", "cost.csv", "summary.md"} <= {p.name for p in (tmp / "t").iterdir()}

    f = next((tmp / "raw/t/dev/b").glob("*.json"))
    f.write_text(json.dumps({**json.loads(f.read_text()), "config_hash": "other"}))
    with pytest.raises(SystemExit, match="config_hash"):
        aggregate.main(["--tag", "t", "--split", "dev"])
    aggregate.main(["--tag", "t", "--split", "dev", "--force"])
