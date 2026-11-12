# CSMR Implementation Spec

Every section maps to the paper (`docs/paper.pdf`). Signatures are the contract between team members.

---

## 0. Shared types (`csmr/types.py`) — already written

`Action`, `Memory`, `StateChange`, `CandidateInfo`, `RepairSet`, `Trace`. See file.
`Memory` has two dates: `time` = date of the current version (shown to the agent, §9; REVISE / CONDITIONAL set it to the observation time, §8, §11) and `created_at` = when the memory was first written (defaults to `time` on construction, never changed by `update()`). Features that ask "when was this written" (`temporal_after` §4, `freshness` §5) use `created_at`.

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
- Count calls (cache hits vs misses) in a module-level counter → `stats()`; used for EvidenceCost and latency metrics. `prior_hits` = hits on replies cached before the last `reset_stats()` (an earlier run); 0 means the run was uncached (§14 wall time).
- Exponential backoff on rate-limit errors.

### 1.3 `csmr/models.py`
Lazy singletons:
- `embed(texts: list[str]) -> np.ndarray` (all-MiniLM-L6-v2, normalized). Cache by text.
- `nli(premise: str, hypothesis: str) -> dict[str, float]` with keys `contradiction`, `entailment`, `neutral` (cross-encoder/nli-deberta-v3-small, softmax). Cache by pair. Batch version `nli_batch`.

### 1.4 Store adapters
`csmr/adapters/base.py` (written) defines `MemoryStoreAdapter`.
`csmr/adapters/memory.py`: `InMemoryAdapter(memories: list[Memory])`, `from_json(path)`, `to_json(path)`, `snapshot() -> InMemoryAdapter` (deep copy). `update()` MUST push previous version to `history` and bump `version`. `update(..., time=...)` sets a new date (used by the commit for REVISE / CONDITIONAL, §11).

### 1.5 Scenario loader (`csmr/scenario.py`)
`load_scenario(path) -> Scenario` with `memories: list[Memory]`, `observation: Observation`, and a separate `gold: GoldLabels | None` (`actions`, `dependencies`, `eval_questions`, optional `persistence_memories: list[Memory]`). **Pipeline functions never receive `gold`.**

---

## 2. Phase A — State-change extraction (`pipeline/extract.py`) — Paper IV-A

```python
def extract(obs, known_attributes, cfg, *, related, known_entities) -> tuple[StateChange | None, dict | None]   # (Δ, note)
def extract_state_change(obs, known_attributes, cfg=None, *, related=None, known_entities=None) -> StateChange | None   # extract(...)[0]
def find_roots(delta: StateChange, obs: Observation, store, cfg=None) -> list[str]
```
- One LLM call, JSON mode. The response ALWAYS carries `entity, attribute, statement` (`statement` = third-person restatement of the observation, e.g. "The user sold their car.") and `"state_change": true|false`. When true it also carries `old, new, confidence, change_type, temporary` and becomes Δ. When false, Δ is `None` and `{entity, attribute, statement}` is returned as the **note** (`None` if the model omitted any of the three; `Trace.note`). A legacy nested `"state_change": {...}` object is accepted too.
  A state change needs some stored fact to become false or outdated: adding a new fact compatible with the stored ones is NOT a state change (few-shot from an unrelated domain: "I also enjoy hiking" when memory says "likes swimming" → false).
  `change_type` ∈ {`remove` (fact stops holding with no new value of the same kind: "I cancelled my Netflix subscription", "I stopped going to yoga classes"), `replace` (a new value of the same kind: "I switched my bank account to HDFC", "My favourite team is now Mumbai Indians")} — the prompt carries these as few-shot examples, which must never mirror dev/test scenarios; stored on `StateChange.change_type` (default `replace`).
  Tentative / planned / hedged changes ("might", "planning to") are still extracted, with lower confidence; `null` only when the message changes no stored fact. Weak evidence is judged in Phase C, not here.
  Attribute should be a short snake_case key (e.g. `vehicle_ownership`, `employment`, `residence`).
  The prompt receives `known_attributes` (attribute keys already in the store) and must reuse one when it fits.
  It also receives `known_entities` (entity names already in the store) and MUST reuse that exact name when the change is about the same subject. Entity names are compared case-insensitively everywhere (`types.same_entity`: roots, candidates, `same_entity` feature, adapter filter).
  `temporary` (bool, `StateChange.temporary`, default false): true only if the change has a stated end date or duration (few-shot from unrelated domains: "the swimming pool is closed for maintenance until Friday", "I'm off caffeine for two weeks"); only a JSON `true` counts. The temporary flag used downstream is `delta.temporary OR` the `cfg.temporary_patterns` regex on the observation (§5).
  It also receives `related` = texts of the `cfg.extract_context_k` stored memories closest to the observation (the attribute is not known before extraction), and must return `null` when the observation only repeats or confirms them (few-shot from an unrelated domain: yoga).
- `time`, `source` copied from the observation.
- Roots = memories with `same_entity(m.entity, delta.entity)` and `m.attribute == delta.attribute`.
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
| `nli_contra_topical` | nli(delta.statement, cand.text).contradiction × `cos_sim` (NLI contradiction is only meaningful for topically related pairs; unrelated sentences score ≈1 otherwise; falls back to "The {entity}'s {attribute} is now {new}." if statement is empty) |
| `temporal_after` | cand written after root (1/0), per (root, cand) pair, by `created_at` (a re-dated memory does not become "after"); ISO timestamps compare correctly as strings |
| `same_source` | 1/0 |
| `causal_cue` | graded LLM answer (JSON `{"answer": "yes"|"maybe"|"no"}`) → yes 1.0, maybe 0.5, no 0.0 (by first letter): "Stored fact A: {root.text}. It changed: {delta.statement}. Is statement B now likely FALSE or OUT OF DATE because of this change? Answer one word: yes, maybe, or no. B: {cand.text}". Stricter than "needs re-checking": unrelated facts about the same subject (language, cache) should answer no. |
| `hop_closeness` | 1.0 hop0, 0.6 hop1, 0.3 hop2 |

For a candidate with multiple roots: P = max over roots. Root itself: P = 1.0.
Coefficients loaded from `models/dependency.json` if present else `config.yaml: dependency_weights` (hand-set).

## 5. Phase C — Evidence (`pipeline/evidence.py`) — Paper IV-C

```python
def evidence_features(cand: Memory, delta, obs, info: CandidateInfo, corroboration: int, cfg=None, roots: list[Memory] | None = None) -> dict[str, float]
def evidence_score(features) -> float            # S(E, m)
def fit_evidence(X, y) -> None
```
| name | definition |
|---|---|
| `source_reliability` | lookup in config: user 0.9, tool 0.8, third_party 0.5, inferred 0.3 |
| `directness` | 1.0 root, 0.6 hop1, 0.3 hop2 |
| `contradiction_strength` | root: nli(delta.statement, cand.text).contradiction. Non-root: max over roots of nli(root.text, delta.statement).contradiction (premise = root; the reverse direction fails on the small NLI model, which reads "The user sold their car." as ENTAILING "The user owns a car."; a dependent is not contradicted by the change itself, its root is; NLI against the dependent's own text rewards topical overlap). If there are no roots, the candidate's own text is used. Consequence (accepted): non-roots at the same hop share S, so evidence no longer separates them; car_sale M3 becomes REVISE instead of paper Table II's VERIFY. |
| `not_hedged` | 1 − hedge flag (regex list in config: might, maybe, thinking of, planning to, not sure, probably) |
| `not_temporary` | 1 − temporary flag; temporary = `delta.temporary` (extractor, §2) OR `cfg.temporary_patterns` regex on the observation (this week, for now, temporarily, until, for a few days, for a week, at the moment) |
| `corroboration` | min(count of supporting observations, 3)/3 |
| `extraction_conf` | delta.confidence |
| `freshness` | 1.0 if obs.time ≥ cand.created_at else 0.5 |

Also expose flags `hedged: bool`, `temporary: bool` on `CandidateInfo` (used by the action rule).

## 6. Proposed action (`pipeline/actions.py`)

```python
def propose_action(mid, info: CandidateInfo, cfg, change_type="replace") -> Action
def revise_text(mem: Memory, delta, obs) -> str          # one LLM call
def condition_text(mem: Memory, delta, obs) -> str       # one LLM call
```
Rule (encodes Section V constraints):
```
if mid not in C:                                   KEEP
elif P >= τd and info.temporary:                   CONDITIONAL   (checked first; roots included, P = 1)
elif is_root and S >= τe:                          REVOKE if delta.change_type == "remove" else REVISE
elif P >= τd and S >= τe:                          REVISE   (any modified non-root)
elif P >= τd:                                      VERIFY
else:                                              KEEP
```
REVISE text: ONE short sentence, third person, present tense, stating the CURRENT state first, with the old value only in brackets at the end; no date. Root + replace: "The ShopKart backend uses MongoDB Atlas (migrated from PostgreSQL 15)." Dependent whose current value is unknown: "ShopKart's current schema-migration method is unknown (it used Alembic before the move to MongoDB)." The prompt's own few-shot examples come from an unrelated domain (blog moved from WordPress to Ghost), never from dev/test/showcase scenarios. Prompts (REVISE, CONDITIONAL, judge) receive the observation date `obs.time[:10]` only, never the time of day, so identical input on the same day hits the LLM cache.

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
def view(store, rs: RepairSet, revised: dict[str,str], conditions: dict[str,str], new_mem: Memory | None = None) -> list[Memory]
def observation_memory(delta, obs, store) -> Memory
```
The new observation is itself a memory (`observation_memory`: text = delta.statement, entity/attribute from Δ, source/time from the observation, id `OBS<n>`) and is appended to EVERY view, R0 included. Repair actions apply only to existing memories. Exception: the counterfactual Benefit (§10) and commit validation (§11) remove it from the view again when `cfg.probe_exclude_observation` is true; it is still committed (§11) and still visible to the agent at evaluation time (§13, §14).
Per memory: REVOKE → excluded; REVISE → text replaced; VERIFY → text prefixed "[unverified] "; CONDITIONAL → "{text} (only valid if: {condition})"; else unchanged. Never mutates the store.
REVISE and CONDITIONAL (`REDATED_ACTIONS`) also set the memory's `time` to the observation's time (`new_mem.time`), exactly as the commit does (§11), so the agent sees the same dates during selection and after the commit. VERIFY and KEEP keep their original date.

## 9. Downstream agent (`pipeline/agent.py`)

```python
def answer(question: str, memories: list[Memory], k: int = 5, cfg=None) -> str
```
Retrieve top-k by cosine to the question from the given memory list; LLM answers using ONLY those memories. Each retrieved memory is shown with the date it was recorded, `- [YYYY-MM-DD] <text>` (`memory.time[:10]`; "undated" if empty). Instructions: if memories conflict, rely on the most recent one; answer exactly "unknown" only when no retrieved memory addresses the question. This is identical for every method (CSMR selection and validation, baselines, eval); no method may change the prompt. The date is `Memory.time`, the date of the current version (not `created_at`); a REVISE'd or CONDITIONAL memory carries the date of the observation that changed it (§8, §11). An empty LLM answer is retried once (changed prompt, double max tokens), then replaced by "unknown"; an empty answer never reaches the judge. Memories rendered with an `[unverified]` prefix (VERIFY) are flagged in the prompt ("Memories marked [unverified] may be outdated; mention the uncertainty."), so the agent surfaces the uncertainty instead of asserting the stale fact. Same agent is used by all baselines and by eval.

## 10. Counterfactual selection (`pipeline/counterfactual.py`) — Paper IV-D, V

```python
def generate_probes(delta, obs, cands, store, n: int, cfg=None) -> tuple[list[dict], dict[str, str]]   # LLM; from candidate memory texts ONLY
# Returns (probes, gaps). Each probe is {"question", "source", "source_text"}: the candidate memory it was written from (the change_question's source is the root, or the first candidate if there is no root) and that memory's pre-commit text.
# NEUTRAL questions about the user's CURRENT state ("What does the user grow in their garden?"), never mentioning or presupposing the change.
# Targeting: each memory question asks about the detail of that memory connected to the stored fact the change_question is about (same kind of detail: place, device, provider, person...), so its answer reveals whether the memory still holds; it never presupposes the memory's old value of that detail ("The user buys sailing supplies from a chandlery next to Harbor Point", fact "sailboat kept at the Harbor Point marina" -> "Where does the user buy sailing supplies?", not "What does the user buy?" nor "Which shop next to Harbor Point...?").
# Over-generic LLM probes (only wh-words, auxiliaries, "the user" and generic verbs, cfg.probe_generic_words; e.g. "What does the user have?") are dropped like filtered probes.
# Prompt examples use no wording from dev/showcase scenarios or their gold questions.
# The LLM writes one "change_question" about the CURRENT value of the root fact (root memory text(s); the first candidate in order if there is no root), then one per candidate (KEEP-proposed included, so a dependent KEEP memory can still expose over-repair), roots first, then by P_dep descending, ties by numeric memory id (never string order: "M12" must not outrank "M2").
# Coverage: duplicate questions (case/whitespace-insensitive; template fallback probes too) are dropped, first kept.
# Probe selection (strict; cfg.n_probes = 6 is a MAXIMUM, not a target; fewer probes is fine, slots are never filled with other probes):
#   - at most ONE probe per source memory: the change_question is the root's probe, so the root's own memory question is dropped;
#   - eligible: the change_question (always), and the probe of every root or candidate with P_dep >= tau_d; probes of
#     candidates with P_dep < tau_d are never kept;
#   - if more than n are eligible: the change_question first, then candidates proposed for a non-KEEP action, then the
#     rest, each in the order above; the output keeps that order. (A non-KEEP candidate always has P_dep >= tau_d or is
#     a root, §6, so it is always eligible; only the n cap can leave it unprobed.)
# gaps (Trace.probe_gaps): every non-KEEP-proposed candidate left without a probe -> "missing" (LLM gave none) | "filtered" (presupposing / yes-no / change-hinting) | "duplicate" | "capped" (eligible, but over the n maximum).
# Probes using a word only the observation introduces (e.g. "sold") are dropped.
# Probes are open questions (what/which/where/how/who): LLM probes starting with a yes/no word (cfg.probe_yes_no_starts) or containing a change-hinting word (cfg.probe_banned_patterns: "still", "anymore", "no longer", ...) are dropped. If no LLM probe survives, the template fallback probes (below) are used and a "probes" fallback is recorded.
def judge(answer: str, question: str, delta, obs, cfg, source_text: str = "") -> dict       # ONE call, two explicit steps:
# The prompt shows the probe's source memory (source_text). If the answer still asserts the part of that memory the change makes false or likely outdated, the verdict is "contradicts" even when the answer omits the specific detail (prompt example uses wording unrelated to any scenario).
# {"determined_by_change": bool, "verdict": "consistent"|"vague"|"contradicts"} -> score 1.0 / 0.5 / 0.0
#   determined_by_change=false: "unknown"/uncertain answers are consistent (1.0)
#   determined_by_change=true:  "unknown"/vague is vague (0.5)
#   Only claims about the CURRENT state can contradict: mentioning an old value explicitly as past/previous ("used Alembic before; current method unknown") is NOT asserting it.
#   An answer asserting something the change makes false OR likely outdated is "contradicts" (0.0) whatever determined_by_change is (e.g. "drives to work every day: Yes" after "sold their car"). determined_by_change only decides how unknown-like answers score.
#   "contradicts" is allowed ONLY in those two cases: the answer conflicts with the change (Δ / observation), or it still asserts the part of the source memory that the change makes false or outdated. An answer that merely differs from an unaffected source memory (other wording, more or less detail, a different reading) is NOT a contradiction: the source memory is context, not the reference answer (prompt example: a book-club schedule after a phone-carrier change). Motivation: car_sale_distractors, "On which days does the user work?" (source "works from 10am to 6pm on weekdays") was judged "contradicts" after "I sold my car" in every repair set.
# Unknown-like answers (empty, or matching cfg.unknown_patterns when <= cfg.unknown_max_words words, or when the answer starts with a pattern) get the verdict IN CODE: consistent (1.0) if determined_by_change is false, vague (0.5) if true.
#   The LLM verdict field is used only for answers that make a claim.
# Probes are always third person ("the user"), never "you".
# Probes name the specific subject as the memory text does (e.g. "ShopKart"), never a generic stand-in; the attribute key is never shown to the probe prompt.
# Both fields are stored per probe and per repair set in the trace ("judgements").
def score(rs, ...) -> dict  # {"benefit","collateral","evidence_cost","size","total"}
def select(repair_sets, ...) -> tuple[RepairSet, list[dict]]
```
Objective (paper Section V):
```
total = Benefit − λc·Collateral − λe·EvidenceCost − λs·|R|
Benefit       = mean judge score over probes, answered by the agent over view(store, rs, ..., new_mem) minus the
                observation memory itself when cfg.probe_exclude_observation (default true), i.e. over M ⊕ R alone;
                new_mem still supplies the date given to REVISE / CONDITIONAL memories (§8)
Collateral    = Σ_{m changed (REVISE/REVOKE)} (1 − P_dep(m))          # runtime proxy
EvidenceCost  = (#VERIFY actions in rs) * cfg.verify_cost + llm_calls_for_rs * cfg.call_cost
|R|           = number of non-KEEP actions
```
- **Benefit excludes the new observation memory** (`probe_exclude_observation: true`, design decision taken before the freeze). Rationale: Benefit should measure whether the *repaired store itself* answers correctly. In later turns the observation is one memory among many and is not reliably retrieved (the agent sees only the top `agent_top_k`), so a stale memory that the observation happens to outvote today will mislead the agent later. With the observation in every view, R0 can score as well as a real repair because the date-aware agent (§9) resolves the conflict in favour of the observation, and the objective then prefers no repair. The judge still receives Δ and the observation (it needs them to grade). The observation memory is still committed (§11) and is part of the store that baselines and evaluation answer from (§13, §14); only CSMR's internal Benefit (and the matching commit validation) leave it out. `false` restores the previous behaviour (observation in every view).
- argmax over feasible sets; ties → smaller |R|.
- Cache answers by (question, source_text, hash of view) — identical views reuse results. The view hash covers each memory's id, date and text (everything the agent sees).
- Commit validation re-judges each probe with the same pre-commit source_text, and under the same `probe_exclude_observation` setting (it is given the committed observation memory's id and leaves it out of the view), so selection and validation see the same memories.
- Malformed LLM JSON never crashes a run; each fallback is counted in `Trace.llm_stats["fallbacks"]` (`llm.fallback(kind)`): extract (missing/invalid keys) → no state change; probes (missing/invalid) → one template probe per candidate, "What is the current state of this: <memory text>?" (built from memory text only, so never filtered); judge → retried once with a changed prompt, then verdict "vague" (0.5), `determined_by_change` null.; causal cue (dependency) → retried once, then "no"; REVISE text → retried once, then template "Previously: <text>. Possibly outdated after: <statement>"; CONDITIONAL condition → retried once, then "it is not affected by: <statement>" (kinds `causal_cue`, `revise_text`, `condition_text`).
- Ties: totals within `cfg.tie_eps` are tied → smaller |R|. Token budgets: `cfg.probe_max_tokens`, `cfg.judge_max_tokens`.
- Ablation `use_counterfactual=false`: select R3 (apply all proposed actions) without probes.
- Ablation `use_minimality=false`: λc = λs = 0.

## 11. Phase E — Commit (`pipeline/commit.py`) — Paper IV-E

```python
def commit(store, rs, revised, conditions, new_mem) -> list[dict]   # version transitions; new_mem = observation_memory(...)
def validate(store, probes, delta, obs, expected_benefit, cfg=None, obs_mem_id=None) -> dict   # obs_mem_id: see §10
```
- The observation memory (`observation_memory`) is added to the store in the same commit (version 1, status active).
- `commit` raises ValueError if `rs.feasible` is False.
- REVISE → update text, set status "active" and clear any condition (via `update(..., clear_condition=True)`); REVOKE → status "revoked"; VERIFY → status "pending_verify", add to pending queue; CONDITIONAL → status "conditional", store condition; all push history.
- REVISE and CONDITIONAL also set `time` to the observation's time (`update(..., time=new_mem.time)`); the previous version, with its original date, is the history entry. REVOKE and VERIFY keep their date.
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
`observe()` semantics: a text ending in `?` is a question — nothing is run or stored (empty trace). Otherwise `run_pipeline` runs; repair happens only if a state change is extracted (Phase E then also stores the observation memory). If no state change is extracted, `observe()` still stores the observation as a new memory (`observation_memory(None, obs, store, trace.note)`: id `OBS<n>`; text/entity/attribute come from the extractor's note, so a later observation can find it as a root. Only if the note is missing does it fall back to the raw text, entity `user`, attribute `general`) and records an `ADD` transition in the trace; no existing memory is touched. `run_pipeline` itself never stores when Δ is None (baselines/eval rely on that).
LLM errors: a rate limit honours `Retry-After` (capped at `llm.max_retry_wait_s`); if the error text matches `llm.quota_patterns` (daily quota) `llm.QuotaExhausted` is raised ("Daily LLM quota exhausted - resume later; cached calls are kept") instead of retrying.

Trace JSON contains: Δ, note, roots, candidates with all features, P_dep, S, proposed actions, every repair set with score breakdown + feasibility, chosen set, version transitions, LLM call stats, wall time.

Ablation flags (`config.yaml: ablation`):
- `use_dependency=false` → P_dep = 1 for every candidate
- `use_evidence=false` → S = 1 for every candidate
- `use_counterfactual=false` → see §10
- `use_minimality=false` → see §10

## 13. Baselines (`baselines/`)

Each: `def run(obs, store, cfg) -> dict[str, Action]` + revised texts. Same agent, same inputs. Every baseline's final view includes the same appended observation memory (`observation_memory`); only repair actions differ.
- `append_only`: store observation as new memory, change nothing. Since the shared agent shows dates and resolves conflicts by recency (§9), this is also the recency baseline; there is no separate `recency` method.
- `last_write_wins`: REVISE/REVOKE memories with same entity+attribute as Δ only.
- `similarity_prop`: REVISE every memory with cosine(obs, m) ≥ cfg.sim_prop_threshold.
- `dependency_only`: CSMR Phase A–C, modify every candidate with P ≥ τd (ignore evidence, no counterfactual).
- `evidence_only`: modify every candidate with S ≥ τe (ignore dependency).
- `llm_oneshot`: give LLM all memories + observation, ask for an action per memory in JSON.

## 14. Evaluation (`eval/`) — Paper VIII

`metrics.py` (reads gold — the ONLY place besides fit code):

```python
@dataclass
class Record: scenario: Scenario; final_actions: dict[str, str]; trace: Trace | None = None; store: MemoryStoreAdapter | None = None
def repair(rec) / dependency(rec, cfg) / candidate_recall(rec) / qa(rec, cfg) / persistence(rec, cfg) / cost(rec) -> dict[str, tuple[num, den]]
def scenario_metrics(rec, cfg) -> dict[str, tuple[num, den]]      # all of the above for one record
def aggregate(rows, cfg) -> dict[str, {"value", "lo", "hi", "n"}]
```
- One `Record` per (method, scenario). `final_actions`: memory id → action; ids left out are KEEP. `trace` is None for a method with no CSMR trace (trace metrics are then skipped). `store` is the committed store; without it QA accuracy and persistence are skipped.
- Every metric is a per-scenario count pair `(numerator, denominator)`. Its value over a set of scenarios is **pooled**: Σ numerators / Σ denominators (over memories / questions, not a mean of per-scenario ratios). A metric with Σ denominators = 0 is NaN.
- Gold labels are strict: every memory needs a `gold.actions` label and it must be exactly one of the five actions (`"REVOKE/REVISE"` raises).

Definitions (gold = `gold.actions`; pred = `final_actions`; changed = pred ∈ {REVISE, REVOKE}; gold-changed = gold ∈ {REVISE, REVOKE}; C = `trace.candidates`):

| metric | numerator | denominator |
|---|---|---|
| `repair_precision` | \|changed ∩ gold-changed\| | \|changed\| |
| `repair_recall` | \|changed ∩ gold-changed\| | \|gold-changed\| |
| `over_invalidation` | gold KEEP memories that were changed | gold KEEP memories |
| `under_repair` | gold-changed memories left KEEP | \|gold-changed\| |
| `action_accuracy` (5-way) | memories with pred = gold | all memories |
| `dependent_precision` | \|pred-dep ∩ gold-dep\| | \|pred-dep\| |
| `dependent_recall` | \|pred-dep ∩ gold-dep\| | \|gold-dep\| |
| `dependent_f1` | 2·\|pred-dep ∩ gold-dep\| | \|pred-dep\| + \|gold-dep\| |
| `candidate_recall` | gold non-KEEP memories ∈ C | gold non-KEEP memories |
| `qa_accuracy` | eval questions judged correct | eval questions |
| `persistence` | eval questions judged correct with later memories in view | eval questions |
| `persistence_fallback` | 1 if the config distractors were used | 1 |
| `llm_calls` | `trace.llm_stats.calls` | 1 |
| `wall_time_s` | `trace.wall_time_s` | 1 (uncached runs only) |

- Repair: REVISE where gold says REVOKE (or the reverse) counts as a correct change for precision/recall but as wrong for `action_accuracy`. `under_repair` is not 1 − recall: a gold-changed memory given VERIFY/CONDITIONAL is neither repaired nor left KEEP.
- **Dependent-level dependency P/R/F1**: the classifier scores each non-root candidate against the roots and the trace keeps the max, so pairs cannot be recovered; a gold `(root, dependent)` pair is matched on its dependent. pred-dep = non-root candidates with P_dep ≥ τd; gold-dep = second elements of `gold.dependencies` (chains included); pipeline roots (P_dep = 1 by construction) are removed from both.
- Downstream QA accuracy: the shared agent (§9) answers each `eval_questions[].q` on the committed store, rendered as in §8 (revoked excluded, `[unverified]` / condition tags). `qa_judge` grades the answer against `gold_answer` with `QA_JUDGE_PROMPT` → `{"correct": bool}`; this prompt is separate from the selection judge (§10) and never sees Δ. A malformed reply is retried once with a changed prompt; if still malformed the answer counts as incorrect and a `qa_judge` fallback is recorded.
- Persistence: the eval questions are re-asked with later-turn memories appended to the agent's view (never to the store). They are the scenario's `gold.persistence_memories` (see DATASET_NOTE; each must be dated after the observation, else ValueError). A scenario without the field falls back to the fixed off-topic `cfg.persistence_distractors`, dated after the observation; `persistence_fallback` reports the share of scenarios that did.
- Cost: `llm_calls` counts every pipeline call, cache hits included, so a cached rerun reports the same number. `wall_time_s` is taken only from uncached runs: traces with `llm_stats.prior_hits == 0` (no reply served from an earlier run's cache; hits on replies cached earlier in the same run are normal). Traces without that counter are excluded.
- Confidence intervals (`aggregate`): scenario-level percentile bootstrap. Whole scenarios are resampled with replacement `cfg.bootstrap_samples` times (rng seeded with `cfg.seed`, the same resamples for every metric), the pooled ratio is recomputed on each resample, and `lo`/`hi` are the percentiles for `cfg.ci_level` (0.95 → 2.5 / 97.5). Resamples where the ratio is undefined are dropped. `n` = scenarios where the metric is defined.

`run_all.py --split test --methods all` → `results/main.csv`, `results/per_category.csv`.
`ablations.py` → `results/ablations.csv`. `plots.py` → λc sweep plot (repair recall vs over-invalidation).
