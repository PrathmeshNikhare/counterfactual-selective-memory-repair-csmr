# CSMR Implementation Spec

Every section maps to the paper (`docs/paper.pdf`). Signatures are the contract between team members.

---

## 0. Shared types (`csmr/types.py`) — already written

`Action`, `Memory`, `StateChange`, `CandidateInfo`, `RepairSet`, `Trace`. See file.

## 1. Infrastructure

### 1.1 `csmr/config.py`
`load_config(path="config.yaml") -> dict` (cached). Supports dotted override from CLI: `--set lambda_c=0.3`.

### 1.2 `csmr/llm.py`
```python
def call(prompt: str, *, system: str | None = None, json_mode: bool = False,
         model: str | None = None, max_tokens: int = 512) -> str | dict
```
- OpenAI-compatible client; `LLM_API_KEY`, `LLM_BASE_URL` from `.env`; model from config.
- temperature 0. Disk cache (`diskcache`) keyed by sha256(model + system + prompt + json_mode).
- `json_mode=True`: parse JSON; on failure retry once with "Return valid JSON only."; then raise.
- Count calls (cache hits vs misses) in a module-level counter → `stats()`; used for EvidenceCost and latency metrics.
- Exponential backoff on rate-limit errors.

### 1.3 `csmr/models.py`
Lazy singletons:
- `embed(texts: list[str]) -> np.ndarray` (all-MiniLM-L6-v2, normalized). Cache by text.
- `nli(premise: str, hypothesis: str) -> dict[str, float]` with keys `contradiction`, `entailment`, `neutral` (cross-encoder/nli-deberta-v3-small, softmax). Cache by pair. Batch version `nli_batch`.

### 1.4 Store adapters
`csmr/adapters/base.py` (written) defines `MemoryStoreAdapter`.
`csmr/adapters/memory.py`: `InMemoryAdapter(memories: list[Memory])`, `from_json(path)`, `to_json(path)`, `snapshot() -> InMemoryAdapter` (deep copy). `update()` MUST push previous version to `history` and bump `version`.

### 1.5 Scenario loader (`csmr/scenario.py`)
`load_scenario(path) -> Scenario` with `memories: list[Memory]`, `observation: Observation`, and a separate `gold: GoldLabels | None`. **Pipeline functions never receive `gold`.**

---

## 2. Phase A — State-change extraction (`pipeline/extract.py`) — Paper IV-A

```python
def extract_state_change(obs: Observation, known_attributes: list[str]) -> StateChange | None
def find_roots(delta: StateChange, obs: Observation, store) -> list[str]
```
- One LLM call, JSON mode. Prompt asks for `entity, attribute, old, new, confidence, statement, change_type` (`statement` = third-person restatement of the observation, e.g. "The user sold their car."), or `{"state_change": null}`.
  `change_type` ∈ {`remove` (fact stops holding with no new value of the same kind: "I cancelled my Netflix subscription", "I stopped going to yoga classes"), `replace` (a new value of the same kind: "I switched my bank account to HDFC", "My favourite team is now Mumbai Indians")} — the prompt carries these as few-shot examples, which must never mirror dev/test scenarios; stored on `StateChange.change_type` (default `replace`).
  Tentative / planned / hedged changes ("might", "planning to") are still extracted, with lower confidence; `null` only when the message changes no stored fact. Weak evidence is judged in Phase C, not here.
  Attribute should be a short snake_case key (e.g. `vehicle_ownership`, `employment`, `residence`).
  The prompt receives `known_attributes` (attribute keys already in the store) and must reuse one when it fits.
- `time`, `source` copied from the observation.
- Roots = memories with `m.entity == delta.entity` and `m.attribute == delta.attribute`.
  Only if that set is empty, fall back to memories with
  `nli(delta.statement, m.text).contradiction >= cfg.root_nli_threshold` (default 0.8).
  (A high-NLI memory alongside an attribute match is a validity dependency, not a root — paper Table II.)
- No store modification. If `None` → pipeline returns early with empty trace (cheap gating).

## 3. Phase B — Candidate construction (`pipeline/candidates.py`) — Paper IV-B

```python
def build_candidates(delta, obs, roots: list[str], store) -> dict[str, CandidateInfo]
```
1. cosine(obs embedding, memory embeddings) → top `cfg.top_k` (default 10); if `delta.entity` known, restrict to same entity (fallback to all if <K). Metadata filter (paper IV-B "semantic embeddings and metadata filters for entity, attribute and time"; Section VI "M5 should not enter the candidate set"): a top-K hit is kept only if it has the same entity AND (same attribute as Δ OR cosine ≥ `cfg.topk_min_sim`). Needed because with small stores top-K alone returns every memory.
2. Hop expansion from roots: hop-1 = memories sharing entity AND (attribute OR cosine ≥ `cfg.hop_sim` (0.35)) with any root; hop-2 = same from hop-1, up to `cfg.max_hops` (2).
3. Union with roots. Record per candidate: `hop` (0 for roots, 1, 2, or `None` if only from top-K → treat as 2), `retrieved_by` ("root" | "topk" | "hop").
- Permissive by design: entering C ≠ repair.

## 4. Phase C — Dependency (`pipeline/dependency.py`) — Paper IV-C

```python
def dependency_features(root: Memory, cand: Memory, delta, obs, info: CandidateInfo) -> dict[str, float]
def dependency_prob(features: dict) -> float     # logistic regression
def fit_dependency(X, y) -> None                 # trains and saves coef to models/dependency.json
```
Features (all in [0,1]):
| name | definition |
|---|---|
| `cos_sim` | cosine(root, cand) |
| `same_entity` | 1/0 |
| `same_attribute` | 1/0 |
| `nli_entail` | nli(root.text, cand.text).entailment |
| `nli_contra_new` | nli(delta.statement, cand.text).contradiction (falls back to "The {entity}'s {attribute} is now {new}." if statement is empty) |
| `temporal_after` | cand written after root (1/0) |
| `same_source` | 1/0 |
| `causal_cue` | LLM yes/no (JSON `{"answer": "yes"|"no"}`) → 1/0: "If statement A stopped being true, would statement B likely need to be re-checked or updated? A: {root.text} B: {cand.text} Answer yes or no." |
| `hop_closeness` | 1.0 hop0, 0.6 hop1, 0.3 hop2 |

For a candidate with multiple roots: P = max over roots. Root itself: P = 1.0.
Coefficients loaded from `models/dependency.json` if present else `config.yaml: dependency_weights` (hand-set).

## 5. Phase C — Evidence (`pipeline/evidence.py`) — Paper IV-C

```python
def evidence_features(cand: Memory, delta, obs, info: CandidateInfo, corroboration: int) -> dict[str, float]
def evidence_score(features) -> float            # S(E, m)
def fit_evidence(X, y) -> None
```
| name | definition |
|---|---|
| `source_reliability` | lookup in config: user 0.9, tool 0.8, third_party 0.5, inferred 0.3 |
| `directness` | 1.0 root, 0.6 hop1, 0.3 hop2 |
| `contradiction_strength` | nli(delta.statement, cand.text).contradiction |
| `not_hedged` | 1 − hedge flag (regex list in config: might, maybe, thinking of, planning to, not sure, probably) |
| `not_temporary` | 1 − temporary flag (regex: this week, for now, temporarily, until, for a few days) |
| `corroboration` | min(count of supporting observations, 3)/3 |
| `extraction_conf` | delta.confidence |
| `freshness` | 1.0 if obs.time ≥ cand.time else 0.5 |

Also expose flags `hedged: bool`, `temporary: bool` on `CandidateInfo` (used by the action rule).

## 6. Proposed action (`pipeline/actions.py`)

```python
def propose_action(mid, info: CandidateInfo, cfg) -> Action
def revise_text(mem: Memory, delta, obs) -> str          # one LLM call
def condition_text(mem: Memory, delta, obs) -> str       # one LLM call
```
Rule (encodes Section V constraints):
```
if mid not in C:                                   KEEP
elif is_root and S >= τe:                          REVOKE if delta.change_type == "remove" else REVISE
elif P >= τd and S >= τe:                          REVISE   (any modified non-root)
elif P >= τd and info.temporary:                   CONDITIONAL
elif P >= τd:                                      VERIFY
else:                                              KEEP
```
REVISE text must preserve history-compatible meaning, e.g. "The user previously drove to work by car; current commute mode unknown after selling the car (as of {time})."

## 7. Phase D — Repair sets (`pipeline/repair_sets.py`) — Paper IV-D

```python
def generate_repair_sets(cands: dict[str, CandidateInfo], proposed: dict[str, Action]) -> list[RepairSet]
def is_feasible(rs: RepairSet, cands, cfg) -> bool
```
- R0: {} (no change)
- R1: roots only, with their proposed action (if proposed is KEEP for a root, use REVISE)
- R2: R1 + hop-1 candidates with proposed action ≠ KEEP
- R3: R2 + hop-2 / top-K candidates with proposed action ≠ KEEP
- R4: every candidate with P ≥ 0.2 set to REVISE/REVOKE (broad propagation alternative)
- Feasibility: every REVISE/REVOKE must have S ≥ τe; every action key ∈ C. Infeasible sets are kept in the trace (marked) but not selectable.
- Deduplicate identical sets (keep lowest index name).

## 8. Overlay (`pipeline/overlay.py`)

```python
def view(store, rs: RepairSet, revised: dict[str,str], conditions: dict[str,str]) -> list[Memory]
```
Per memory: REVOKE → excluded; REVISE → text replaced; VERIFY → text prefixed "[unverified] "; CONDITIONAL → "{text} (only valid if: {condition})"; else unchanged. Never mutates the store.

## 9. Downstream agent (`pipeline/agent.py`)

```python
def answer(question: str, memories: list[Memory], k: int = 5) -> str
```
Retrieve top-k by cosine to the question from the given memory list; LLM answers using ONLY those memories; must say "unknown" if not determinable. Same agent is used by all baselines and by eval.

## 10. Counterfactual selection (`pipeline/counterfactual.py`) — Paper IV-D, V

```python
def generate_probes(delta, obs, cands, store, n: int) -> list[str]   # LLM; from Δ and candidates ONLY
def judge(answer: str, question: str, delta, obs) -> float           # 1 consistent / 0.5 unsupported / 0 contradicts
def score(rs, ...) -> dict  # {"benefit","collateral","evidence_cost","size","total"}
def select(repair_sets, ...) -> tuple[RepairSet, list[dict]]
```
Objective (paper Section V):
```
total = Benefit − λc·Collateral − λe·EvidenceCost − λs·|R|
Benefit       = mean judge score over probes under view(store, rs)
Collateral    = Σ_{m changed (REVISE/REVOKE)} (1 − P_dep(m))          # runtime proxy
EvidenceCost  = (#VERIFY actions in rs) * cfg.verify_cost + llm_calls_for_rs * cfg.call_cost
|R|           = number of non-KEEP actions
```
- argmax over feasible sets; ties → smaller |R|.
- Cache answers by (question, hash of view) — identical views reuse results.
- Ablation `use_counterfactual=false`: select R3 (apply all proposed actions) without probes.
- Ablation `use_minimality=false`: λc = λs = 0.

## 11. Phase E — Commit (`pipeline/commit.py`) — Paper IV-E

```python
def commit(store, rs, revised, conditions, obs) -> list[dict]   # version transitions
def validate(store, probes, delta, obs, expected_benefit) -> dict
```
- REVISE → update text; REVOKE → status "revoked"; VERIFY → status "pending_verify", add to pending queue; CONDITIONAL → status "conditional", store condition; all push history.
- Validation re-runs probes on committed store; warn if |benefit − expected| > 0.01.

## 12. Orchestrator + public API (`pipeline/run.py`, `csmr/api.py`)

```python
def run_pipeline(obs, store, cfg) -> Trace
class CSMR:
    def __init__(self, store: MemoryStoreAdapter, config="config.yaml")
    def observe(self, text, source="user", time=None) -> Trace
    def retrieve(self, query, k=5) -> list[str]     # excludes revoked, tags unverified/conditional
    def pending_verifications(self) -> list[Memory]
```
Trace JSON contains: Δ, roots, candidates with all features, P_dep, S, proposed actions, every repair set with score breakdown + feasibility, chosen set, version transitions, LLM call stats, wall time.

Ablation flags (`config.yaml: ablation`):
- `use_dependency=false` → P_dep = 1 for every candidate
- `use_evidence=false` → S = 1 for every candidate
- `use_counterfactual=false` → see §10
- `use_minimality=false` → see §10

## 13. Baselines (`baselines/`)

Each: `def run(obs, store, cfg) -> dict[str, Action]` + revised texts. Same agent, same inputs.
- `append_only`: store observation as new memory, change nothing.
- `recency`: append; agent instructed to prefer most recent memory.
- `last_write_wins`: REVISE/REVOKE memories with same entity+attribute as Δ only.
- `similarity_prop`: REVISE every memory with cosine(obs, m) ≥ cfg.sim_prop_threshold.
- `dependency_only`: CSMR Phase A–C, modify every candidate with P ≥ τd (ignore evidence, no counterfactual).
- `evidence_only`: modify every candidate with S ≥ τe (ignore dependency).
- `llm_oneshot`: give LLM all memories + observation, ask for an action per memory in JSON.

## 14. Evaluation (`eval/`) — Paper VIII

`metrics.py` (reads gold — the ONLY place besides fit code):
- dependency P/R/F1 (pairs with P ≥ τd vs gold_dependencies)
- candidate recall (gold non-KEEP memories ∈ C)
- repair precision = |changed ∩ gold-changed| / |changed|; repair recall = |changed ∩ gold-changed| / |gold-changed| (changed = REVISE/REVOKE; also report an action-level accuracy over 5 classes)
- over-invalidation = gold KEEP memories that were REVISE/REVOKE’d / gold KEEP count
- under-repair = gold REVISE/REVOKE memories left KEEP / gold-changed count
- downstream QA accuracy: agent answers `eval_questions` on committed store; LLM judge vs gold answer (different prompt from the selection judge)
- persistence: append 3 distractor memories, re-ask eval questions
- LLM calls, wall time per scenario

`run_all.py --split test --methods all` → `results/main.csv`, `results/per_category.csv`.
`ablations.py` → `results/ablations.csv`. `plots.py` → λc sweep plot (repair recall vs over-invalidation).
