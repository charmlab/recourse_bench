---
name: recoursebench-smoke-tests
description: Run RecourseBench method smoke tests, save complete per-method results, and write a consolidated Markdown report. Use when Codex needs to check smoke-test health across all or selected experiment methods, preserve stdout/stderr/status evidence, summarize failures, or report branch readiness without claiming paper-level reproduction.
metadata:
  version: v0.3.0
---

# RecourseBench Smoke Tests

## Overview

Use this skill to run the repository's method smoke tests, save auditable
results, and create a single report that states exactly what passed, failed,
or was skipped. Prefer tests discovered from the current checkout over a
hard-coded method list.

Treat smoke tests as functionality checks only. Never describe a passing smoke
test as a full reproduction or as evidence that paper results were reproduced.

## Workflow

1. Resolve and enter the repository root from the current checkout:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
PYTHON="$REPO_ROOT/.venv/bin/python"
```

2. Check the branch, dirty state, and Python environment before running tests:

```bash
git status --short --branch
"$PYTHON" --version
```

Use `"$PYTHON"` for every smoke test. Do not silently use system Python.
Stop and report an environment blocker when that interpreter is missing.

3. Create one timestamped result directory:

```bash
OUTPUT_BASE="${OUTPUT_BASE:-smoke_test}"
RUN_ROOT="$OUTPUT_BASE/run_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"
```

Do not overwrite a previous run. Store all evidence and the final report under
this directory. Respect a user-supplied `OUTPUT_BASE`; do not assume a
machine-specific output location.

4. Discover smoke tests from the checkout instead of relying on a stale list:

```bash
find experiment -mindepth 2 -maxdepth 2 -type f -name smoke.py | sort \
  > "$RUN_ROOT/discovered_tests.txt"
```

For a full smoke pass, run every discovered file, including `experiment/toy`
when present. If the user requests selected methods, record the requested
scope and save the selected test list as `$RUN_ROOT/selected_tests.txt`.

5. Save run context before executing tests:

- `$RUN_ROOT/git_status.txt`: `git status --short --branch`
- `$RUN_ROOT/git_head.txt`: `git rev-parse HEAD`
- `$RUN_ROOT/environment.txt`: UTC start time, hostname, Python executable and
  version, plus NumPy and Torch versions when importable
- `$RUN_ROOT/discovered_tests.txt`: all smoke tests found
- `$RUN_ROOT/selected_tests.txt`: tests actually selected

6. Run tests sequentially unless the user explicitly requests parallel or
Slurm execution. Continue after failures so the report covers the full scope.

For each `experiment/<method>/smoke.py`, save:

- `<method>.out`: stdout
- `<method>.err`: stderr
- `<method>.status`: command, timestamps, duration, and exit code

Use this pattern or an equivalent wrapper:

```bash
while IFS= read -r TEST; do
  METHOD="$(basename "$(dirname "$TEST")")"
  CMD="$PYTHON $TEST"
  OUT="$RUN_ROOT/${METHOD}.out"
  ERR="$RUN_ROOT/${METHOD}.err"
  STATUS="$RUN_ROOT/${METHOD}.status"
  START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  START_SECONDS="$(date +%s)"

  {
    echo "METHOD=$METHOD"
    echo "TEST=$TEST"
    echo "COMMAND=$CMD"
    echo "START_UTC=$START_UTC"
  } > "$STATUS"

  "$PYTHON" "$TEST" > "$OUT" 2> "$ERR"
  EXIT_CODE=$?

  END_SECONDS="$(date +%s)"
  {
    echo "END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "DURATION_SECONDS=$((END_SECONDS - START_SECONDS))"
    echo "EXIT_CODE=$EXIT_CODE"
  } >> "$STATUS"
done < "$RUN_ROOT/selected_tests.txt"
```

Do not use `set -e` around the test loop. A single failure must not prevent
remaining smoke tests from running.

7. Inspect every status file and the relevant tail of stdout/stderr. Classify:

- `PASS`: exit code `0`
- `FAIL`: nonzero exit code caused by test or code behavior
- `BLOCKED`: test could not run because of missing dependency, artifact,
  license, unavailable hardware, or environment setup
- `SKIPPED`: deliberately excluded from the requested scope

Do not convert a nonzero exit into `PASS` because some output was produced.
Do not hide import errors, tracebacks, warnings relevant to correctness, or
environment blockers.

8. Compare each method's result with its checked-in reference log:

```text
$REPO_ROOT/experiment/<method>/<method>_smoke_log.txt
```

Review the new stdout, stderr, exit status, final metrics, and any traceback or
error against the reference log. Ignore expected volatile differences such as
timestamps, durations, temporary paths, and harmless formatting changes. If
the new run has the same completion/failure outcome and materially the same
metrics or diagnostic result as the reference log, record the comparison as
`OK`. A reference log that records a failure only makes the comparison `OK`;
it does not change the current run's status from `FAIL` or `BLOCKED` to
`PASS`.

Record `DIFFERENT` when the outcome, actionable error, or material result
differs. Record `NO REFERENCE` when the expected reference log is absent. Do
not claim that a result is `OK` without checking the method's reference log.

9. Write `$RUN_ROOT/REPORT.md` after all selected tests finish.

## Report Requirements

Start the report with:

- run directory
- UTC start and end time
- Git branch and commit
- whether the worktree was dirty
- Python executable and version
- requested scope and number of discovered/selected tests
- totals for `PASS`, `FAIL`, `BLOCKED`, and `SKIPPED`
- overall result: `PASS` only when every selected test passed; otherwise
  `ISSUES FOUND`

Include a result table:

```text
| Method | Status | Reference comparison | Exit code | Duration (s) | Evidence | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
```

For `Evidence`, link to the relative `.out`, `.err`, and `.status` files.
For `Reference comparison`, report `OK`, `DIFFERENT`, or `NO REFERENCE` and
link to `../<method>/<method>_smoke_log.txt` when it exists. When the
comparison is `OK`, explicitly say that the new result is the same as the
reference-log result.
For failures and blockers, summarize the first actionable root cause, not only
the final traceback line.

Add these sections after the table:

1. `Failures and Blockers`: diagnosis for every non-passing selected test.
2. `Reference Comparisons`: explain every `DIFFERENT` or `NO REFERENCE`
   result and state that matching results are `OK`.
3. `Warnings and Caveats`: dirty-worktree effects, optional dependency
   warnings, hardware limitations, or scoped execution.
4. `Artifacts`: list the discovered/selected manifests, environment capture,
   per-method logs, statuses, and report.

If every selected test passes, explicitly state that no smoke-test failures
were observed. Still retain all result files.

## Completion Rules

- Do not stop after launching jobs; wait for all selected tests to reach a
  terminal state.
- Do not delete or replace prior smoke-run directories.
- Do not modify method code merely to make a test pass unless the user also
  requested fixes.
- Report the exact result directory in the final response.
- Mention failed or blocked methods in the final response, even when the full
  details are already in `REPORT.md`.
