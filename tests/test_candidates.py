from csmr.adapters.memory import InMemoryAdapter
from csmr.pipeline.candidates import build_candidates
from csmr.scenario import load_scenario
from csmr.types import StateChange


def test_car_sale_candidates():
    sc = load_scenario("data/dev/car_sale.json")
    d = StateChange("user", "vehicle_ownership", "owns a car", "no car", sc.observation.time, "user", 0.95,
                    "The user sold their car.")
    c = build_candidates(d, sc.observation, ["M1"], InMemoryAdapter(sc.memories))
    print({k: (v.hop, v.retrieved_by) for k, v in c.items()})
    assert {"M1", "M2", "M3"} <= set(c) and "M5" not in c
    assert c["M1"].is_root and c["M1"].hop == 0 and c["M1"].retrieved_by == "root"
