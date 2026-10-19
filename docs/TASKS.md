# CSMR Task List

Work top to bottom within your lane. Tick `[x]` when acceptance criteria pass. Lanes A–D can run in parallel after T0.
Integration fixture for everything: `data/dev/car_sale.json`.

## T0 — Everyone (first 1 hour)
- [ ] Read `CLAUDE.md`, `docs/SPEC.md`, `csmr/types.py`, `csmr/adapters/base.py`
- [ ] `pip install -r requirements.txt`, fill `.env`, `pytest -q` runs (0 tests is fine)
- [ ] Create git repo, one branch per lane (`lane-a`, `lane-b`, …)

## Lane A — Infrastructure + orchestration
- [x] **A1 `csmr/config.py`** — loads `config.yaml`, CLI overrides. *Accept:* `load_config()["tau_e"] == 0.6`.
- [ ] **A2 `csmr/llm.py`** — cached, temp 0, JSON mode, retries, `stats()`. *Accept:* same prompt twice → 1 miss + 1 hit; bad JSON retried once.
- [ ] **A3 `csmr/models.py`** — `embed`, `nli`, `nli_batch`. *Accept:* nli("I sold my car.", "The user owns a car.").contradiction > 0.5.
- [ ] **A4 `csmr/adapters/memory.py`** + `csmr/scenario.py`. *Accept:* load car_sale.json; `update()` bumps version and keeps history; pipeline-facing objects contain no gold fields.
- [ ] **A5 `pipeline/run.py` + `run.py` CLI** — chains phases (stubs OK initially), writes trace JSON. *Accept:* `python run.py data/dev/car_sale.json` prints a trace.
- [ ] **A6 `csmr/api.py`** + `examples/chat_agent.py` (terminal loop: observe → retrieve → answer). *Accept:* typing "I sold my car" then "how do I get to work?" gives a non-car answer.

## Lane B — Phases A & B
- [ ] **B1 `pipeline/extract.py`** — `extract_state_change`, `find_roots`. *Accept:* car → entity user, attribute ~vehicle_ownership, roots == ["M1"]; "I like pizza" with no conflict → None or no roots.
- [ ] **B2 `pipeline/candidates.py`**. *Accept:* car → C ⊇ {M1, M2, M3}; M5 ∉ C (M4 may be in C).
- [ ] **B3 tests** for both across all 6 dev scenarios; log candidate recall vs gold (test-only reading of gold).

## Lane C — Phase C
- [ ] **C1 `pipeline/dependency.py`** — features, hand-set weights from config, `fit_dependency`. *Accept:* car → P(M2) ≥ τd, P(M4) < τd, P(M5) not computed (not in C).
- [ ] **C2 `pipeline/evidence.py`** — features, hedge/temporary flags, `fit_evidence`. *Accept:* `bike_maybe.json` → all S < τe; `car_sale.json` → S(M1), S(M2) ≥ τe.
- [ ] **C3 `pipeline/actions.py`** — `propose_action`, `revise_text`, `condition_text`. *Accept:* car → M1 REVOKE/REVISE, M2 REVISE, M3 VERIFY, M4 KEEP; `phone_repair.json` → CONDITIONAL on the root dependent.
- [ ] **C4** run `method-auditor` on C1–C3; fix findings.

## Lane D — Phases D & E
- [ ] **D1 `pipeline/repair_sets.py`** — R0..R4, dedup, feasibility. *Accept:* unit test with synthetic CandidateInfo: R4 infeasible when a member has S < τe.
- [ ] **D2 `pipeline/overlay.py`** + `pipeline/agent.py`. *Accept:* view never mutates store (test with deep-equality before/after).
- [ ] **D3 `pipeline/counterfactual.py`** — probes, judge, objective, select, answer cache. *Accept:* car → chosen set equals R2 (M1 changed, M2 REVISE, M3 VERIFY allowed, M4/M5 KEEP); score breakdown in trace.
- [ ] **D4 `pipeline/commit.py`** — versioned commit, pending-verify queue, validation. *Accept:* after commit M1.version == 2 and M1.history has the original text.
- [ ] **D5** run `method-auditor` on D1–D4 — especially the no-leakage check.

## Integration (whoever finishes first)
- [ ] **I1** `tests/test_integration.py`: all 6 dev scenarios end-to-end; car scenario reproduces paper Table II.
- [ ] **I2** ablation flags each change behaviour (test that toggling each flag changes at least one output on dev set).

## Phase 2 — after dataset exists (see docs/DATASET_NOTE.md)
- [ ] **E1 `baselines/`** — 7 baselines per SPEC §13.
- [ ] **E2 `eval/metrics.py`** — all metrics per SPEC §14.
- [ ] **E3** `fit_dependency` / `fit_evidence` on train split; tune τd, τe, λ on val.
- [ ] **E4 `eval/run_all.py`** → results/main.csv, per_category.csv.
- [ ] **E5 `eval/ablations.py`**, `eval/plots.py` (λc sweep).
- [ ] **E6** error analysis: sample 20 failures, tag with paper's 6 error categories → results/errors.csv.
