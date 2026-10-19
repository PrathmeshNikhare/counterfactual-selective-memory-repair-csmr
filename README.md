# CSMR — Getting started with Claude Code

## One-time setup (any one teammate, ~15 min)

```bash
cd csmr
git init && git add . && git commit -m "CSMR harness"
# push to a private GitHub repo and add your teammates

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # put in your LLM base URL + API key
```

Install Claude Code (see Anthropic's docs for the current install command), then run `claude` inside the `csmr/` folder.
Claude Code automatically reads `CLAUDE.md`, the slash commands in `.claude/commands/`, and the `method-auditor` agent in `.claude/agents/`.

## What's in the harness

| File | Purpose |
|---|---|
| `CLAUDE.md` | Project rules Claude Code follows every session (method fidelity, no leakage, versioning, config discipline) |
| `docs/SPEC.md` | Module-by-module spec with function signatures and formulas |
| `docs/TASKS.md` | Ordered tasks in 4 parallel lanes, each with acceptance criteria |
| `docs/paper.pdf` | Your paper, the source of truth |
| `csmr/types.py`, `csmr/adapters/base.py` | Shared contracts, already written, so lanes don't conflict |
| `config.yaml` | Every threshold, weight, λ and ablation flag (hand-set starting weights included) |
| `data/dev/*.json` | 6 hand-written dev scenarios (car sale = paper Table II, 2-hop, direct contradiction, false similarity, weak evidence, temporary change) |
| `.claude/commands/` | `/next-task <lane>`, `/run-dev`, `/check-leakage` |
| `.claude/agents/method-auditor.md` | Reviewer subagent that checks code against the paper |

## Splitting the work (4 people, 4 lanes)

| Person | Lane | Branch | Builds |
|---|---|---|---|
| 1 | A | `lane-a` | config, LLM wrapper, models, store adapter, orchestrator, public API, demo agent |
| 2 | B | `lane-b` | Phase A (extraction) + Phase B (candidates) |
| 3 | C | `lane-c` | Phase C (dependency, evidence, action rule) |
| 4 | D | `lane-d` | Phase D (repair sets, overlay, agent, counterfactual) + Phase E (commit) |

Lane A should do A1–A4 first (about 1–2 hours) and push; B, C, D pull those and can then run real code instead of stubs.

## Prompts to use in Claude Code

**Start of every session:**
```
/next-task B
```
(use your own lane letter). Repeat until your lane is done.

**After finishing a module:**
```
Use the method-auditor agent to review csmr/pipeline/candidates.py
```

**When the whole pipeline is merged:**
```
/run-dev
```
Then iterate on `config.yaml` weights until `car_sale` reproduces paper Table II and the other dev scenarios mostly match. Then:
```
/check-leakage
```

**First message if someone starts without the slash command:**
```
Read CLAUDE.md, docs/SPEC.md and docs/TASKS.md. I'm on lane C. Implement task C1 only,
with tests, following the spec exactly. Don't change csmr/types.py.
```

## Tips

- Keep each Claude Code session to one task. Start a fresh session (`/clear`) between tasks so context stays clean.
- Run `pytest -q -m "not llm"` for fast checks; run the full suite before merging.
- The LLM cache in `.cache/llm` makes reruns free. Don't delete it during experiments.
- If Claude Code proposes changing the algorithm, say no: CLAUDE.md rule 1.

## After the pipeline works

Generate the dataset (Phase 2, see `docs/DATASET_NOTE.md`) in the same schema as `data/dev/`, then continue with tasks E1–E6 in `docs/TASKS.md`.
