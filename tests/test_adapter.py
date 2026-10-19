import dataclasses

from csmr.adapters.memory import InMemoryAdapter
from csmr.scenario import load_scenario
from csmr.types import Memory, Observation

PATH = "data/dev/car_sale.json"


def test_load_car_sale():
    sc = load_scenario(PATH)
    assert [m.id for m in sc.memories] == ["M1", "M2", "M3", "M4", "M5"]
    assert sc.observation.text == "I sold my car."
    assert sc.gold.actions["M1"] == "REVOKE"


def test_pipeline_facing_objects_have_no_gold():
    sc = load_scenario(PATH)
    fields = {f.name for cls in (Memory, Observation) for f in dataclasses.fields(cls)}
    assert not any("gold" in f for f in fields)
    assert "gold" not in str(sc.memories) + str(sc.observation)


def test_update_bumps_version_and_keeps_history():
    store = InMemoryAdapter(load_scenario(PATH).memories)
    m = store.update("M1", status="revoked", reason="sold")
    assert m.version == 2 and m.status == "revoked"
    assert len(m.history) == 1 and m.history[0]["status"] == "active" and m.history[0]["version"] == 1
    store.update("M1", text="The user previously owned a car.")
    assert store.get("M1").version == 3 and len(store.get("M1").history) == 2
    assert "M1" not in [x.id for x in store.all()] and "M1" in [x.id for x in store.all(include_revoked=True)]


def test_snapshot_independent_and_json_roundtrip(tmp_path):
    store = InMemoryAdapter(load_scenario(PATH).memories)
    snap = store.snapshot()
    store.update("M2", status="pending_verify")
    assert snap.get("M2").version == 1
    store.to_json(tmp_path / "s.json")
    assert InMemoryAdapter.from_json(tmp_path / "s.json").get("M2").history == store.get("M2").history


def test_search_ranks_relevant_first():
    store = InMemoryAdapter(load_scenario(PATH).memories)
    assert store.search("Italian cuisine", 1)[0].id == "M5"
