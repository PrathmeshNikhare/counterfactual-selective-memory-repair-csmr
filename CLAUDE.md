# CSMR — Counterfactual Selective Memory Repair

Research implementation for a BTP minor project (IIIT Pune). We are implementing the method
described in `docs/paper.pdf` **exactly as proposed**. This is research code: correctness,
reproducibility and clean experiments matter; UI does not.

## What this project is

When an LLM agent receives a state-changing observation ("I sold my car"), CSMR decides which
*already-stored* memories to repair. Pipeline (paper Section IV):

1. **A. State-change extraction** — observation → Δ = (entity, attribute, old, new, time, source, confidence); find direct conflict roots. No modification.
2. **B. Candidate construction** — embedding top-K + entity/attribute filter + 1–2 hop expansion → candidate set C.
3. **C. Dependency + evidence assessment** — P(D=1|z) via logistic regression; S(E,m) via logistic regression; proposed action per candidate.
4. **D. Counterfactual repair-set selection** — build R0..R4, apply each to a *virtual overlay*, answer self-generated probes, judge vs Δ, score the objective, pick best feasible.
5. **E. Minimal commit + validation** — versioned commit, re-run probes, write a JSON trace.

Actions: KEEP, VERIFY, CONDITIONAL, REVISE, REVOKE.

Full module-by-module spec: **`docs/SPEC.md`** (read it before writing any module).
Ordered work plan with acceptance criteria: **`docs/TASKS.md`**.

## Non-negotiable rules

1. **Do not change the proposed method.** If something in the spec seems wrong, stop and ask; do not "improve" the algorithm.
2. **No label leakage.** The `gold` block in scenario files (`gold.actions`, `gold.dependencies`, `gold.eval_questions`) is read ONLY by `eval/` and by `fit()` training code. The pipeline under `csmr/pipeline/` must never read them. The counterfactual stage uses *probes the system generates itself*.
3. **Never delete memories.** Every change pushes the old version into `history`, increments `version`, and sets `status`.
4. **Constraints from paper Section V are hard rules:**
   - memory not in candidate set C ⇒ action is KEEP
   - action in {REVISE, REVOKE} ⇒ S(E, m) ≥ τe
5. **All LLM calls go through `csmr/llm.py`** (disk cache, temperature 0, retries, JSON mode). Never call an API client directly elsewhere.
6. **All tunables live in `config.yaml`** (K, hops, τd, τe, λc, λe, λs, model names, ablation flags). No magic numbers in code.
7. **Ablation flags must work by swapping a stage for its trivial default**, not by separate code paths duplicated across files.
8. Shared types are in `csmr/types.py` and the store interface in `csmr/adapters/base.py`. Do not change their fields without updating every user and telling the team.
9. Classifiers start with **hand-set coefficients** (in `config.yaml`) so the pipeline runs before the dataset exists; `fit()` replaces them later without other code changes.

## Tech stack

Python 3.10+, `openai` client (OpenAI-compatible: works with OpenAI, Groq, Together, OpenRouter, Ollama via `base_url`),
`sentence-transformers` (all-MiniLM-L6-v2), `transformers` cross-encoder NLI (cross-encoder/nli-deberta-v3-small),
`scikit-learn`, `numpy`, `pandas`, `matplotlib`, `pydantic`, `pyyaml`, `diskcache`, `pytest`.
No vector DB needed — stores are small; use numpy cosine.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env            # add LLM_API_KEY, LLM_BASE_URL
pytest -q                       # unit + integration tests
python run.py data/dev/car_sale.json          # run pipeline on one scenario, print trace
python run.py data/dev/car_sale.json --trace out/trace.json
python examples/chat_agent.py   # terminal demo agent using the public API
python eval/run_all.py --split test           # (later) all methods → results/*.csv
```

## Layout

```
csmr/
  types.py              shared dataclasses (Memory, StateChange, Action, RepairSet, Trace)
  config.py             loads config.yaml
  llm.py                cached LLM wrapper
  models.py             lazy-loaded embedder + NLI singletons
  adapters/base.py      MemoryStoreAdapter ABC
  adapters/memory.py    in-memory / JSON adapter (versioned)
  pipeline/extract.py       Phase A
  pipeline/candidates.py    Phase B
  pipeline/dependency.py    Phase C (dependency)
  pipeline/evidence.py      Phase C (evidence)
  pipeline/actions.py       proposed-action rule + REVISE text generation
  pipeline/repair_sets.py   Phase D (R0..R4 + feasibility)
  pipeline/overlay.py       virtual view of the store under a repair set
  pipeline/agent.py         minimal retrieval-QA agent (shared by CSMR and baselines)
  pipeline/counterfactual.py probes, judge, objective, argmax
  pipeline/commit.py        Phase E
  api.py                public CSMR class: observe(), retrieve(), pending_verifications()
baselines/              append_only, recency, last_write_wins, similarity_prop, dependency_only, evidence_only, llm_oneshot
eval/                   metrics.py, run_all.py, ablations.py, plots.py
data/dev/               hand-written dev scenarios (integration tests)
data/generated/         full dataset (later)
examples/chat_agent.py
tests/
```

## Workflow for Claude Code

- Work one task from `docs/TASKS.md` at a time. Read the task's acceptance criteria first, implement, write/extend tests, run `pytest -q`, then tick the box in TASKS.md.
- Keep functions small and typed. Docstrings reference the paper section they implement (e.g. `# Paper IV-C`).
- When a module needs an LLM prompt, put the prompt template as a module-level constant so prompts are reviewable in one place.
- Integration test: `data/dev/car_sale.json` must reproduce paper Table II (M1 REVOKE/REVISE, M2 REVISE, M3 VERIFY, M4 KEEP/CONDITIONAL, M5 KEEP; R2-equivalent set chosen).
- After finishing any pipeline module, run the `method-auditor` subagent to check it against the paper.
