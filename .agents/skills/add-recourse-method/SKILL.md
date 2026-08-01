---

## name: add-recourse-method
description: Use when adding a new algorithmic recourse method to recourse_bench, including paper/reference-code analysis, MethodObject implementation, registry wiring, smoke config/driver setup, human review gates, validation, and a best-effort reproduction of the paper's reported results.
metadata:
  version: v0.1.1

# Add a New Recourse Method

You are integrating a new algorithmic-recourse paper into `recourse_bench`. Adding
a method is a **plug-in** operation, not a rewrite: you implement one
`MethodObject` subclass, register it, give it a smoke config, and format it.

**This skill operates on a source checkout of `recourse_bench`** — a cloned git
repo where you edit files under `method/` and `experiment/` and import the package
in-tree (editable install). It is *not* for the installed PyPI library; that is
`use-recourse-bench`. Every path below is repo-relative, and you should run
commands from the repo root.

Work through the phases below **in order**: two preparation phases (**Phase -1**
environment sanity and **Phase 0** reference search) and five main phases
(**1–5**) with **three human review gates**.

**The gates assume an interactive human reviewer.** At each `GATE`, when a human
is in the loop, STOP and wait for confirmation before continuing — do not proceed
autonomously past a gate. When you are running **non-interactively** (no human
available to respond), do not fabricate sign-off: write the gate's artifacts to
disk, record in the planning notes that the gate is **unreviewed**, and continue
only through mechanical phases. Never report the method as faithful or complete
while a GATE — especially GATE 3 — remains unreviewed; leave that judgment to the
human. Between gates, keep moving without asking permission for routine steps.

This workflow borrows the *paper2code* discipline of writing **YAML checkpoint
artifacts** for understanding and planning before any code is written, so that
each review gate signs off on a concrete artifact rather than on prose. It keeps
the human gates that *paper2code* omits, because faithfulness to the paper is a
research-judgment question no smoke test can settle.

---

## Inputs you need

- `<name>`: the method's short, lowercase, identifier-safe name (e.g. `proximal`).
- `<ClassName>`: PascalCase + `Method` suffix (e.g. `ProximalMethod`).
- The paper (PDF, arXiv link, or text).
- Public reference code, **if any** (strongly preferred — see Phase 0).
- The method's repo compatibility family:
`template_family: gradient_based | non_gradient`. This decides which existing
methods you read as implementation templates.
- The method's actual mechanism as free text, e.g. gradient optimization, graph
search, generative model, RL policy, MILP/action-set enumeration, prototype
projection, or whatever the paper specifies.

Create a scratch directory for planning artifacts:

```
experiment/<name>/_planning/
```

The three YAML checkpoints live there. They are working artifacts; at Phase 5 ask
the human whether to keep or delete them.

---

## The contract (read this before anything else)

Read `method/method_object.py` in full. A method MUST satisfy:

1. **The class** — `method/<name>/<name>.py`, decorated `@register("<name>")`
  (from `utils.registry`), subclassing `MethodObject`, with signature:
   In `__init__` you MUST set: `self._target_model`, `self._seed`,
   `self._device` (via `model.model_utils.resolve_device`), `self._need_grad`,
   `self._is_trained = False`, `self._desired_class`. Consume extra config keys
   as explicit keyword args; `del kwargs` after.
2. **Registration import** — one line in `method/__init__.py`:
  `from method.<name>.<name> import <ClassName>`. The `@register` decorator
   only runs when this module is imported, so **without this line `Experiment`
   silently cannot discover the method**. This is the most common silent
   failure; do it in Phase 2 and verify it.
3. **A smoke config** — `experiment/<name>/smoke_config.yaml`, cloned from an
  existing method's, with `method.name` (and method kwargs) swapped.
4. **A smoke driver** — `experiment/<name>/smoke.py`, cloned from an existing
  method's (they are near-identical; they add the repo root to `sys.path` and
   run `Experiment` on the config).
5. **Formatting** — `isort` then `black` on the new files only.

The two method bodies carry the real semantics:

- `fit(train_set)` — train if the method needs it; otherwise set
`self._is_trained = True` and return.
- `get_counterfactuals(factuals) -> pd.DataFrame` — return the **same rows
(count and index) and the same feature columns** as the input, steering toward
`self._desired_class`. Represent a **failed row with NaN feature values**. The
base class's `predict()` enforces row-count and column-set preservation and
raises if you violate them, then wraps failures to target `-1`.

Two compatibility rules from the repo:

- `model.device` and `method.device` in the config must match.
- A `gradient_based` method sets `self._need_grad = True` and requires a
differentiable target model exposing `forward()`. A `non_gradient` method
sets `self._need_grad = False`. This is a target-model-access distinction, not
a complete taxonomy of recourse algorithms.

Run a method's smoke check with either:

```bash
python experiment/<name>/smoke.py            # per-method driver
python main.py -p experiment/<name>/smoke_config.yaml   # generic entry (-p PATH)
```

---

## Phase -1 — Environment sanity (preparation)

Before any analysis, confirm the checkout is healthy enough to develop and
smoke-test in. This is cheap and keeps you from debugging the framework when you
meant to debug your method.

1. Confirm you are in a **source checkout**: the repo root contains `method/`,
  `experiment/`, `main.py`, and `utils/registry.py`. If not, you are in the wrong
   place (or want the installed library — see `use-recourse-bench`).
2. Confirm the package imports in-tree and the registries are populated:
  ```bash
   python -c "import recourse_bench as rb; \
   print('methods:', len(rb.list_methods())); \
   print('models:', len(rb.list_models()))"
  ```
   Both counts must be non-zero. If the import fails, fix the environment
   (dependencies, editable install) before continuing.
3. Confirm an **existing** method's smoke check runs end-to-end, e.g.
  `python experiment/wachter/smoke.py`. A green baseline proves that any failure
   later is your method, not the framework.
4. Note the Python version and whether you can install packages — you may need
  that for dependencies in Phase 0/2.

If any step fails, resolve the environment first. Do not start porting a paper
into a broken checkout.

---

## Phase 0 — Reference search (preparation)

Before reading the paper deeply, look for public reference code (authors' repo,
CARLA, other benchmarks). Capture what you find. **Porting** authors' core logic
is far more reliable than implementing math from scratch, because correctness is
inherited — so if reference code exists, the whole task changes shape and Phase 3
becomes adaptation, not derivation.

Record in `_planning/00_references.md`:

- links to any reference implementation(s) and the license,
- which file/function holds the paper-specific algorithmic core,
- whether the paper ships expected numbers you can later check against.

Self-check: state explicitly whether reference code exists. If none, say so —
that raises the review budget for Phase 4.

---

## Phase 1 — Understand before writing

Read, in this order:

1. `method/method_object.py` (the contract).
2. `utils/registry.py` (how `@register` resolves type via MRO and rejects
  duplicate names) and `method/__init__.py` (the import wiring).
3. **2–3 existing methods that match the new method's template family and I/O
  shape**:
  - `gradient_based` examples → `method/wachter/wachter.py`,
  `method/revise/revise.py`;
  - `non_gradient` examples → start with `method/gs/gs.py`,
  `method/face/face.py`, then choose closer local examples if the paper is
  generative, graph-based, action-set based, RL-style, or otherwise better
  matched by another existing method.
   Note how they set `_need_grad`, handle `seed_context`, build NaN failure
   rows, and map the DataFrame I/O onto their internal representation.
4. One smoke config, e.g. `experiment/wachter/smoke_config.yaml`, and its
  `smoke.py`.

Then write **two YAML checkpoints** (paper2code style):

`**_planning/01_algorithm_extraction.yaml`** — every algorithm, objective, and
equation, transcribed faithfully:

```yaml
method: <name>
template_family: gradient_based | non_gradient
mechanism: <specific algorithmic mechanism, free text>
objective: <the objective, policy, rule, or search criterion as written in the paper>
equations:
  - id: eq1
    latex_or_prose: <...>
    role: <what it computes>smo
constraints:
  - <feasibility / actionability / immutability constraints>
hyperparameters:
  - name: <...>
    paper_default: <...>
    config_key: <...>
randomness: <where stochasticity enters; must route through seed_context>
```

`**_planning/02_concept_analysis.yaml**` — how the paper maps onto our contract:

```yaml
needs_training: <true|false — does fit() do real work?>
target_model_requirements: <differentiable+forward()? black-box ok?>
desired_class_handling: <how the method steers toward _desired_class>
io_mapping:
  input: factuals DataFrame (feature columns only)
  internal: <tensor / numpy / authors' format>
  output: same rows+columns, NaN for failures
failure_mode: <when a row yields no counterfactual>
ambiguities:
  - <each place the paper is unclear and a judgment call will be needed>
open_questions:
  - <anything you need the human to confirm>
```

Self-check before the gate: are **all** algorithms/equations from the paper
present in `01`? Is every `MethodObject` field accounted for in `02`?

> **GATE 1 — Understanding.** Present `00_references.md`, `01_algorithm_extraction.yaml`,
> and `02_concept_analysis.yaml`, plus the ambiguities list. **STOP.** Get human
> confirmation that the understanding is correct before any code exists.

---

## Phase 2 — Scaffold (the known-good empty slot)

First write `**_planning/03_implementation_plan.yaml`** (paper2code style):

```yaml
files:
  - path: method/<name>/<name>.py
    contents: [<ClassName> with __init__, fit, get_counterfactuals; helper fns]
  - path: method/<name>/__init__.py        # if the method needs submodules
  - path: experiment/<name>/smoke_config.yaml
  - path: experiment/<name>/smoke.py
registration:
  edit: method/__init__.py
  line: "from method.<name>.<name> import <ClassName>"
dependencies: [<new pip deps, if any — flag for human>]
fit_plan: <bullet steps>
get_counterfactuals_plan: <bullet steps: build internal repr -> run method core -> map back -> NaN failures>
```

**New dependencies require explicit human approval — even between gates.** If the
method needs a pip package not already in the repo's requirements, do **not**
install it silently or add an import on the assumption it is present. List each
one in the plan's `dependencies` with the package, a pinned version, and why it is
needed, and get the human's OK before installing. This is a hard stop regardless
of where you are in the workflow. Prefer what the repo already depends on (numpy,
pandas, torch, scikit-learn) and reuse machinery from methods with the same
template family or a similar algorithmic mechanism. Record any approved addition
in the repo's requirements file so the method stays reproducible. In
non-interactive mode, treat an unapproved dependency like an unreviewed gate:
record it and stop rather than installing it.

Then scaffold:

1. Create `method/<name>/<name>.py` with a **correctly-shaped stub**: full
  `__init__` (setting every required field), `fit` that sets
   `self._is_trained = True`, and `get_counterfactuals` that raises
   `NotImplementedError`.
2. Add the import line to `method/__init__.py`, then **verify the method is
  publicly discoverable** through the registry:
   `<name>` must appear in `rb.list_methods()`. If it does not, the import line is
   missing or wrong and the `@register` decorator never ran — this is the most
   common silent failure. Run it from a fresh interpreter; a stale import can mask
   the problem.
3. Clone `smoke_config.yaml` and `smoke.py` from a method with the same template
  family or a similar config shape; swap `method.name` to `<name>`, set matching
   `method.device`/`model.device`, and set tiny budgets (low iteration/evaluation
   counts, small sample/split) for a fast check.
4. Run `python experiment/<name>/smoke.py` and confirm it fails **only** on the
  stub's `NotImplementedError`. This proves registration, import, config, and
   device wiring are all correct — the algorithm is the only thing missing.

Self-check: does the run reach `get_counterfactuals` and fail there (not on an
import, registry KeyError, device mismatch, or config error)?

> **GATE 2 — Scaffold.** Show the diff and the run output (the
> `NotImplementedError` traceback). **STOP** for the human to confirm the empty
> slot is wired correctly.

---

## Phase 3 — Implement

Fill in `fit` and `get_counterfactuals` per `03_implementation_plan.yaml`,
file-by-file, implementing **exactly what the paper specifies** — do not skip
unclear parts; surface them.

- **If reference code exists:** this is a *porting* task. Wrap the authors' core
algorithmic logic and map our DataFrame I/O onto theirs. Keep their objectives,
update rules, constraints, stopping criteria, and failure semantics intact.
- **If not:** implement from the paper, mirroring Phase 1 reference methods only
where their repo plumbing or algorithmic structure is actually relevant.
- Route **all** randomness through `utils.seed.seed_context(self._seed)`.
- Long loops: use logging and `tqdm`.
- Build NaN rows for failures exactly as the reference methods do; preserve the
input index and feature columns.

---

## Phase 4 — Validate

**Minimal contract check (run this before the metric checks).** A green smoke run
can still hide a method that left a required field unset or bypassed the base
class. With the constructed method object `m` and the input `factuals` DataFrame
(the smoke driver builds both), assert the contract directly:

```python
for attr in ("_target_model", "_seed", "_device", "_need_grad",
             "_is_trained", "_desired_class"):
    assert hasattr(m, attr), f"contract violation: __init__ never set {attr}"

cf = m.get_counterfactuals(factuals)
assert len(cf) == len(factuals) and cf.index.equals(factuals.index), "row mismatch"
assert list(cf.columns) == list(factuals.columns), "feature-column mismatch"
```

Then re-run the smoke config and check:

- **Shape:** output row count and feature columns equal the input's (the base
class will raise otherwise).
- **NaN handling:** failed rows are NaN; successful rows are finite and within
expected feature ranges.
- **Non-trivial metrics:** `validity` and `distance` are meaningful, not
degenerate (not all-fail, not zero-distance copies of the input).

A passing smoke test verifies **shape and plumbing, not faithfulness**. This gate
is where the real correctness check happens.

**Best-effort paper reproduction.** In addition to the smoke test, make a
best-effort attempt to reproduce the paper's *reported experimental results* —
**when the paper provides enough information to do so**. The smoke config is a
tiny health check; a reproduction targets the paper's actual numbers.

- Attempt it only when the paper reports concrete numbers **and** gives enough
setup to target them: the dataset, target model, sample/split, and the metric
definitions. If key details are missing (preprocessing, hyperparameters,
seeds, splits), do **not** guess or tune to force a match — record exactly what
was missing and skip the attempt.
- Build a reproduction config that matches the paper's setup as closely as the
repo allows — its dataset/model where registered, its metrics, and a comparable
sample — rather than the tiny smoke budget. Reuse an existing benchmark config
(`benchmark/configs/`) as the template, not the smoke config.
- Run it and record your numbers next to the paper's, with the gap and the exact
setup you used. Note any principled reasons for a gap (different split,
unavailable dataset variant, missing hyperparameter).
- This is **best-effort**, not the full reproduction workflow — that is the
separate `recoursebench-reproduction` skill. Do not expand scope here; report
what a single, honest attempt shows.

Write the outcome to `_planning/04_reproduction.md`: either the paper-vs-repro
comparison, or a short note that the paper lacks enough information to attempt it
(and what specifically was missing). **Report it at GATE 3.**

---

## Phase 4.5 — Paper-derived experiment tests

After the method passes smoke and MethodObject contract checks, use
`paper-experiment-tests` when the paper contains experimental results, behavioral
claims, ablations, benchmark tables, reference logs, or enough setup details to
define runnable reproduction checks.

Do not duplicate that workflow here. Hand off to `paper-experiment-tests` to:

- extract paper experiment targets,
- gather dataset/model/preprocessing/metric details,
- create runnable RecourseBench configs or pytest tests,
- run the implemented method against those tests,
- compare observed results to paper/reference results,
- write a reproduction report under `experiment/<name>/_paper_tests/`.

If reproduction is impossible, the companion skill should still record why the
status is `not_comparable` or `blocked`.

The output of `paper-experiment-tests` becomes evidence for **GATE 3 —
Faithfulness**, but does not replace human review.

---

> **GATE 3 — Faithfulness.** Present:
>
> - the validated objective and constraints **against the paper** (cite
> `01_algorithm_extraction.yaml`),
> - the smoke metrics,
> - the MethodObject contract check result,
> - the paper-number comparison (`04_reproduction.md`), if available,
> - the `paper-experiment-tests` reproduction report
> (`experiment/<name>/_paper_tests/05_reproduction_report.md`), if that skill
> was run,
> - the reproduction status: `exact_match`, `close_match`, `partial_match`,
> `mismatch`, `not_comparable`, or `blocked`,
> - the remaining ambiguity and judgment calls.
>
> **STOP** for the human to verify the implementation faithfully realizes the
> paper. Concentrate this review on the algorithmic core — the objective, update
> or decision rule, constraints, stopping criteria, and failure semantics — not
> the I/O plumbing.

---

## Phase 5 — Finalize

1. Format the new files only:
  ```bash
   isort method/<name>/ experiment/<name>/
   black method/<name>/ experiment/<name>/
  ```
2. Write a **judgment-call summary**: every place the paper was ambiguous and the
  choice you made (paper2code's per-phase self-check, surfaced for the human).
   This list is the human's targeted review checklist.
3. Ask the human whether to keep or delete `experiment/<name>/_planning/`. In
  non-interactive mode, **keep** the artifacts (they hold the unreviewed-gate
   record) and leave the decision for the human.
4. Confirm: class registered, import wired, smoke config + driver present, smoke
  run green, formatting applied, and the best-effort paper reproduction recorded
   in `_planning/04_reproduction.md` (or its absence justified there).

---

## Where automation stops, and why

"Did this code correctly implement the paper's method?" is a research-judgment
question. An agent can produce a method that runs, has the right shape, and
passes the smoke test while still being subtly wrong — an incorrect objective, a
mishandled constraint, an invalid update rule, or a bad stopping condition — and
no smoke test will catch it. Agents extract high-level algorithm steps well but
miss math-heavy implementation details, so the human review budget belongs on
the paper-specific algorithmic core, not the I/O plumbing. The smoothest path is
to prioritize methods with public reference code, point yourself at both the
paper and that code, and keep the review on the contract adaptation rather than
the algorithm.

Paper-derived tests (`paper-experiment-tests`) strengthen Gate 3 but still do not
prove full faithfulness: aggregate metrics can match the paper while the loss
terms, gradient signs, or constraints are wrong. Treat a `close_match` as
supporting evidence for the human's review, never as a substitute for it.

## Quick reference


| Step                          | Command                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| Run smoke (per-method driver) | `python experiment/<name>/smoke.py`                                                  |
| Run smoke (generic entry)     | `python main.py -p experiment/<name>/smoke_config.yaml`                              |
| Format new files              | `isort method/<name>/ experiment/<name>/ && black method/<name>/ experiment/<name>/` |
| Run paper-derived tests       | use `paper-experiment-tests`; outputs under `experiment/<name>/_paper_tests/`        |


Failed counterfactual rows are `NaN` feature values (target `-1` once wrapped).
`validity` and `distance` are the minimum useful smoke checks.