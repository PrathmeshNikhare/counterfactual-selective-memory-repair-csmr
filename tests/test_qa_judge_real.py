"""Real-judge smoke check of the pending rule (DATASET_NOTE gold answers). Run with `pytest -m llm`."""
import json

import pytest

from csmr.config import ROOT, load_config
from eval.metrics import qa_judge

FAMS = {f["family_id"]: f for f in json.loads((ROOT / "data/seeds/seeds.json").read_text("utf-8"))["families"]}


def _q(family: str, mid: str) -> dict:
    return next(q for q in FAMS[family]["eval_questions"] if q["depends_on"][0] == mid)


@pytest.mark.llm
@pytest.mark.parametrize("ans,expected", [
    ("Runs 5 km.", True),                                                               # plain old value
    ("They run 5 km every morning, though they may stop if their physio advises it.", True),  # hedge on the pending change
    ("They stopped running.", False),                                                   # asserts the change happened
])
def test_pending_question_allows_hedge_not_change(ans, expected):
    q = _q("weak_evidence__health_habit", "M1")
    assert q.get("pending") and q["gold_answer"] == "Runs 5 km."
    assert qa_judge(q["q"], q["gold_answer"], ans, load_config(), q["pending"]) is expected


@pytest.mark.llm
def test_non_pending_question_gets_no_hedge_allowance():
    q = _q("false_similarity__diet", "M2")
    assert not q.get("pending") and q["gold_answer"] == "A glass of milk."
    ans = "A glass of milk, though that may have changed; the household might have switched to oat milk."
    assert qa_judge(q["q"], q["gold_answer"], ans, load_config(), False) is False
