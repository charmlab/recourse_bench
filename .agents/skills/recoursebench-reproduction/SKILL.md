---
name: recoursebench-reproduction
description: Reproduce RecourseBench method checks. Use when Codex needs to run or explain RecourseBench reproduction workflows, especially minimum successful checks across all methods, scoped fallbacks, Slurm execution, status/log capture, and comparison with bundled reproduce_logs.txt files.
metadata:
  version: v0.5.0
---

# RecourseBench Reproduction

## Overview

Use this skill to reproduce RecourseBench method results from the repository root after the checkout and Python environment are available. Prefer the checked reproduction scripts, bundled configs, and reference logs over ad hoc command construction.

The default goal is to get at least one successful executable run for every method. For paper-scale or full-setting reproductions, use each method's `experiment/<method>/reproduce.py`, config files, and `experiment/<method>/reproduce_logs.txt` as the source of truth. Do not duplicate result targets in this skill; they live in the reference logs and previous run artifacts.

When the user asks broadly about RecourseBench verification, first distinguish
between these two workflows:

1. Reproduction checks, which compare method outputs with bundled reproduction
   references.
2. Smoke tests, which check executable health without claiming paper-result
   reproduction.

Use `$recoursebench-smoke-tests` when the requested scope is smoke-test health.

## Workflow

1. Resolve and enter the repository root from the current checkout:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
PYTHON="$REPO_ROOT/.venv/bin/python"
```

2. Use the repository virtual environment:

```bash
"$PYTHON" --version
```

When submitting Slurm jobs, export the dynamically resolved `REPO_ROOT` and
`PYTHON` values into the job script. Do not embed a user- or machine-specific
checkout path.

3. Create an obvious output directory for the run:

```bash
RUN_ROOT=experiment/repro_runs_$(date -u +%Y%m%d)_all_methods_min_success
mkdir -p "$RUN_ROOT"
```

4. Follow the Minimum Successful Checks table first. For each method, capture:

- stdout in `$RUN_ROOT/<method>.out`
- stderr in `$RUN_ROOT/<method>.err`
- status in `$RUN_ROOT/<method>.status`

The status file must include command, start/end time, duration, Python path/version, NumPy version, Torch version when importable, and `EXIT_CODE`.

5. Prefer local execution for short checks. Use Slurm for long checks, CUDA/PyTorch-heavy checks, or methods that have already taken tens of minutes locally.

6. After each successful run, compare output with the nearest bundled reference:

```text
experiment/<method>/reproduce_logs.txt
```

The bundled logs use the standardized section format documented under
Standardized Reference Logs. Do not infer a stronger reproduction level from
the fresh process exit alone. Compare the fresh observed values, covered
experiment scope, and scalar-result assessment with the corresponding fields
in the reference log.

If the standardized log has `No value available.` for a field, report that the
reference does not provide that evidence. If no useful observed result exists
for a smoke-only method, compare against the emitted run summary and mark the
result as smoke/scoped coverage.

7. Report:

- method
- command and log/status path
- exit code
- status (`Reproduced`, `Smoke OK`, `Scoped OK`, or `Failed`)
- duration
- key reproduced metrics or emitted summary fields
- failure reason for unsuccessful runs
- methods with no successful run at all
- any skill updates needed for future reliability

## Command Capture

Use a wrapper or equivalent shell pattern so every run has a complete status file. The command may be local or inside a Slurm script.

```bash
METHOD=method_name
CMD="$PYTHON experiment/path/reproduce.py --args"
OUT="$RUN_ROOT/${METHOD}.out"
ERR="$RUN_ROOT/${METHOD}.err"
STATUS="$RUN_ROOT/${METHOD}.status"

{
  echo "METHOD=$METHOD"
  echo "COMMAND=$CMD"
  echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "PYTHON=$PYTHON"
  "$PYTHON" --version
  "$PYTHON" - <<'PY'
import importlib
for name in ("numpy", "torch"):
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=unavailable ({exc})")
PY
} > "$STATUS"

START=$(date +%s)
bash -lc "$CMD" > "$OUT" 2> "$ERR"
EXIT_CODE=$?
END=$(date +%s)

{
  echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "DURATION_SECONDS=$((END - START))"
  echo "EXIT_CODE=$EXIT_CODE"
} >> "$STATUS"
```

## Standardized Reference Logs

Each `experiment/<method>/reproduce_logs.txt` is a structured text record with
these sections in this order:

```text
Level Badge
Reason
Method
Reproduction Summary
Reference Scope
Observed Result
Reproduction Assessment
Paper Result
Scalar Result Coverage
Reproducibility Criterion
Notes / Caveats
```

Sections may contain prose, bullet lists, tables, JSON-like output, or
`No value available.`. Treat section headings as the stable interface; do not
assume that section bodies are machine-readable JSON.

When comparing a fresh run:

1. Confirm that its dataset, model, method variant, target class, seeds, and
   bounded/full scope agree with `Reference Scope`.
2. Compare emitted metrics with `Observed Result` and `Paper Result`, keeping
   units and percentage-versus-fraction conventions unchanged.
3. Use `Scalar Result Coverage` and `Reproducibility Criterion` when populated.
   The standardized default criterion treats one
   `(dataset, model, metric)` value as a scalar result and uses:

```text
Delta(m, m_hat) = |m_hat - m| / max(|m|, |m_hat|, epsilon_0)
epsilon_0 = 1e-12
delta = 0.15
```

4. Preserve the log's `Level Badge` as reference metadata. A fresh scoped or
   smoke run does not independently establish that badge.
5. Explicitly distinguish missing evidence (`No value available.`) from a
   measured zero.

## Minimum Successful Checks

All commands assume the repository root. Set `PYTHON="$REPO_ROOT/.venv/bin/python"`
before running locally or through Slurm. The relative `.venv/bin/python`
spelling in the command table is shorthand for that resolved interpreter.

| Method | Minimum successful command | Scope and notes |
| --- | --- | --- |
| APAS | `.venv/bin/python experiment/apas/reproduce.py` | Requires working Gurobi WLS/network. Verify the license with a tiny model first. |
| Argumentative Ensembling | `.venv/bin/python experiment/arg_ensembling/reproduce.py --config experiment/arg_ensembling/config.yaml` | Long locally; prefer Slurm. |
| CCHVAE | `.venv/bin/python experiment/cchvae/reproduce.py -p experiment/cchvae/credit_cchvae_sklearn_logistic_regression_cchvae_reproduce.yaml` | Long locally; prefer Slurm. |
| CEMSP | `.venv/bin/python experiment/cemsp/reproduce.py --max-factuals 20` | Bounded successful check. |
| CFRL | `.venv/bin/python experiment/cfrl/reproduce.py` | Long locally; prefer Slurm. |
| CFVAE | `.venv/bin/python experiment/cfvae/reproduce.py --weights-dir experiment/cfvae/cfvae_weights/ --tolerance 1.0` | Requires extracted `experiment/cfvae/cfvae_weights/`. If full generation stalls, use the scoped fallback below. |
| CLUE | `.venv/bin/python experiment/clue/reproduce.py --bnn-art-path "$W/fc_BNN_NEW_ART_compas_models/state_dicts.pkl" --vae-art-path "$W/fc_preact_VAE_NEW(300)_ART_compas_models/theta_best.dat" --vaeac-art-path "$W/fc_preact_VAEAC_NEW_ART_compas_models/theta_best.dat" --vaeac-gt-path "$W/fc_preact_VAEAC_NEW_compas_models/theta_best.dat" --under-vaeac-gt-path "$W/fc_VAEAC_NEW_under_compas_models/theta_best.dat" --device cpu --output-dir "$RUN_ROOT/clue_compas_reproduce" --no-assert-paper` | Set `W` to the extracted CLUE `notebooks/saves` directory. Use scoped method smoke if the paper artifact path is too slow. |
| CoGS | `.venv/bin/python experiment/cogs/reproduce.py` | Local run is acceptable, but not sub-minute. |
| COLS | `.venv/bin/python experiment/cols/reproduce.py --config experiment/cols/reproduce_configs.yaml --profile paper --progress none` | Long locally; prefer Slurm. Avoid low classifier epoch overrides that filter out all factuals. |
| CVAS-PROJ | `.venv/bin/python experiment/cvas_proj/reproduce.py --device cpu --smoke` | Smoke check. |
| DiCE | `.venv/bin/python experiment/dice/reproduce.py --num-factuals 2 --num-boundary-samples 5 --ks 1 --settings DiverseCF --no-save` | Scoped check. Full 480-factual runs are long. |
| DiverseDist | `.venv/bin/python experiment/diverse_dist/reproduce.py` | Requires `alibi==0.9.6`. |
| FACE | `.venv/bin/python experiment/face/reproduce.py` | Full synthetic reproduction is suitable as the minimum check but can take hours. |
| GS | `.venv/bin/python experiment/gs/reproduce.py --mode smoke --factual-limit 5` | Smoke check. |
| LARR | `.venv/bin/python experiment/larr/reproduce.py` | Expect tens of minutes on shared CPU nodes. |
| MACE adult | `.venv/bin/python experiment/mace/reproduce.py --dataset adult --num-factuals 10` | Bounded check. |
| MACE credit | `.venv/bin/python experiment/mace/reproduce.py --dataset credit --num-factuals 10` | Bounded check. Do not use the unsupported `compas` option. |
| PROBE | `.venv/bin/python experiment/probe/reproduce.py --datasets compas_carla --models linear mlp --device cpu` | To cover both logged datasets, add `credit_carla`. |
| ProPlace | `.venv/bin/python experiment/proplace/reproduce.py --config experiment/proplace/config.yaml --max-factuals 5 --model-name linear --ensemble-count 2 --ensemble-epochs 5 --method-k 1` | Requires working Gurobi WLS/network. Bounded license-safe check. |
| RBR | `.venv/bin/python experiment/rbr/reproduce.py --current-config experiment/rbr/german_mlp_rbr_reproduce_current.yaml --future-config experiment/rbr/german_mlp_rbr_reproduce_future.yaml` | Prefer Slurm or a verified long local session. |
| ROAR | `.venv/bin/python experiment/roar/reproduce.py` | Local run is acceptable, but not sub-minute. |
| SNS | `.venv/bin/python experiment/sns/reproduce.py --max-factuals 5 --max-related-models 3` | Bounded check. Full config is longer: add `--config experiment/sns/config.yaml`. |
| TreX | `.venv/bin/python experiment/trex/reproduce.py --device cpu --wi-models 2 --lo-models 2 --pilot-model-count 2 --pilot-factual-limit 5 --factual-limit 5 --cf-steps 50 --trex-max-steps 50` | Tiny CPU check. Full check is `--device cpu` with defaults. |
| CLAPROAR | `.venv/bin/python experiment/claproar/smoke.py` | Smoke-only method. |
| CRUDS | `.venv/bin/python experiment/cruds/smoke.py` | Smoke-only method. |
| FeatureTweak | `.venv/bin/python experiment/feature_tweak/smoke.py` | Smoke-only method. |
| Gravitational | `.venv/bin/python experiment/gravitational/smoke.py` | Smoke-only method. |
| REVISE | `.venv/bin/python experiment/revise/smoke.py` | Smoke-only method. |
| Wachter | `.venv/bin/python experiment/wachter/smoke.py` | Smoke-only method. |

## Scoped Fallbacks

Use scoped fallbacks only when the minimum command is blocked by missing artifacts, license/network issues, or impractical runtime. Mark the result as `Scoped OK`, not `Reproduced`.

| Method | Fallback guidance |
| --- | --- |
| CFVAE | If the full `--tolerance 1.0` run stalls, run an available-weights subset such as `--available-only --datasets adult-age --methods BaseCVAE --max-factuals 10`. |
| CLUE | If the full paper artifact path is CPU-bound for hours, run a method-level smoke that loads the pretrained BNN/VAE and calls `ClueMethod.get_counterfactuals_clue` on one finalized COMPAS row with a very small step count. |
| COLS | Keep default classifier training and bound the search instead of lowering classifier epochs aggressively. |
| DiCE, SNS, TreX | Reduce factual counts, model counts, search steps, or boundary samples. Record the exact scoped arguments. |
| CLAPROAR, CRUDS, FeatureTweak, Gravitational, REVISE, Wachter | The smoke scripts are the intended minimum coverage unless a full reproduction script is added later. |

## Operational Guidance

- Choose CPU or GPU resources based on the method. Use GPU for
  CUDA/PyTorch-heavy runs when available; CPU is fine for lightweight tabular
  or smoke checks.
- Before Slurm submission, verify that the compute node can see `REPO_ROOT`
  and `PYTHON`. Treat an unavailable checkout or interpreter as infrastructure
  failure, not as a method result.
- For Gurobi methods (`APAS`, `ProPlace`), verify WLS/network before interpreting failures. If `token.gurobi.com` is unreachable, report it as an environment/license blocker.
- A local timeout is not a reproduction failure by itself. Prefer Slurm for long checks and inspect stderr/status before deciding the result.
- If a Slurm job exits immediately, inspect environment setup first, especially Python path, missing `torch`, missing pretrained artifacts, and required CLI arguments.
- For bounded or smoke runs, clearly report the selected scope and compare against the closest available reference log. Do not claim full reproduction from a scoped fallback.
- Set `W` to the available CLUE `notebooks/saves` directory. Do not assume a
  fixed artifact location.

## Reference Artifacts

The primary references are the per-method logs discovered in the checkout:

```text
experiment/<method>/reproduce_logs.txt
```

Treat prior generated run directories as optional diagnostic context only.
Never require or silently reuse a dated run directory.
