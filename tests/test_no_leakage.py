"""Structural guard (CLAUDE.md rule 2): pipeline, public API and baselines never touch gold labels or eval/."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(r"\.gold\b|eval_questions|gold_answer|GoldLabels|^\s*(?:from|import)\s+eval\b", re.M)


def _files():
    yield ROOT / "csmr" / "api.py"
    for d in ("csmr/pipeline", "baselines"):
        yield from (ROOT / d).rglob("*.py")


def test_no_gold_access_outside_eval():
    files = list(_files())
    assert len(files) > 1  # guard against the globs silently matching nothing
    hits = [f"{f.relative_to(ROOT)}:{n}: {line.strip()}"
            for f in files for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
            if FORBIDDEN.search(line)]
    assert not hits, "gold/eval leakage:\n" + "\n".join(hits)
