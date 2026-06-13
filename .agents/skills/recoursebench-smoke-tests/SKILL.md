---
name: recoursebench-smoke-tests
description: Run RecourseBench method smoke tests, capture per-method evidence, and summarize branch readiness. Use when Codex needs to check smoke-test health across all or selected experiment methods without claiming paper-level reproduction.
metadata:
  version: v0.5.0
---

# RecourseBench Smoke Tests

## Overview

Use this skill to run RecourseBench smoke tests from the current checkout and
produce an auditable result directory. Smoke tests are functionality checks;
do not describe them as full paper reproduction.

Prefer the repository's checked-in smoke scripts and reference logs over ad hoc
commands. Do not modify method code just to make a smoke test pass unless the
user explicitly asks for fixes.

For CPU versus GPU execution, use the device recorded in the method's matching
reference log, `experiment/<method>/<method>_smoke_log.txt` (`device: cpu` or
`device: cuda`). If the matching log is absent or does not identify a device,
use CPU by default. Record the selected device and any unavailable-device
blocker in the report.

## Workflow

1. Confirm the checkout and environment:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
git status --short --branch
PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python)"
"$PYTHON" --version
```

If the Python environment is missing or incomplete, follow the installation
steps in `README.md`. Do not install optional method dependencies unless the
selected smoke test needs them.

2. Create one timestamped output directory:

```bash
OUTPUT_BASE="${OUTPUT_BASE:-smoke_test}"
RUN_ROOT="$OUTPUT_BASE/run_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_ROOT"
```

Never overwrite a previous run. Store all logs, manifests, context files, and
the final report under `RUN_ROOT`.

3. Discover and select tests from the checkout:

```bash
find experiment -mindepth 2 -maxdepth 2 -type f -name smoke.py | sort \
  > "$RUN_ROOT/discovered_tests.txt"
cp "$RUN_ROOT/discovered_tests.txt" "$RUN_ROOT/selected_tests.txt"
```

For a scoped run, write only the requested methods to `selected_tests.txt` and
record the requested scope in the report.

4. Capture run context before execution:

- `git_status.txt`: `git status --short --branch`
- `git_head.txt`: `git rev-parse HEAD`
- `environment.txt`: UTC time, hostname, Python executable/version, and
  importable NumPy/Torch versions, plus whether CPU or GPU execution was used
- `discovered_tests.txt` and `selected_tests.txt`

5. Run selected smoke tests sequentially unless the user asks for parallel or
Slurm execution. Continue after failures.

Before running each method, inspect its matching reference log and configure
the smoke test for the same CPU or GPU device. Different methods in one run may
therefore use different devices.

For each `experiment/<method>/smoke.py`, save:

- `<method>.out`: stdout
- `<method>.err`: stderr
- `<method>.status`: command, start/end time, duration, and exit code

Do not use `set -e` around the loop; one failing method must not hide later
results.

6. Classify each selected test:

- `PASS`: exit code `0`
- `FAIL`: test ran and exited nonzero
- `BLOCKED`: missing dependency, artifact, hardware, license, or environment
- `SKIPPED`: deliberately outside the requested scope

Do not convert a nonzero exit into `PASS` because partial output exists.

7. Compare each result with its checked-in reference log when present:

```text
experiment/<method>/<method>_smoke_log.txt
```

Record `OK` when the outcome and material diagnostics match the reference log,
`DIFFERENT` when they do not, and `NO REFERENCE` when the file is absent. A
reference log that records a failure can make the comparison `OK`, but it does
not make the current run pass.

8. Write `REPORT.md` in `RUN_ROOT`.

## Report Format

Start with:

- run directory, UTC start/end time, branch, commit, and dirty-worktree status
- Python executable/version
- requested scope and discovered/selected test counts
- totals for `PASS`, `FAIL`, `BLOCKED`, and `SKIPPED`
- overall result: `PASS` only when every selected test passed; otherwise
  `ISSUES FOUND`

Include this table:

```text
| Method | Status | Reference | Exit | Duration (s) | Evidence | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
```

Link evidence to the relative `.out`, `.err`, and `.status` files. For failures
and blockers, summarize the first actionable cause, not only the last traceback
line.

Then add concise sections:

- `Failures and Blockers`
- `Reference Comparisons`
- `Warnings and Caveats`
- `Artifacts`

## Completion Rules

- Wait for all selected tests to reach a terminal state.
- Preserve prior smoke-run directories.
- Report the exact result directory in the final response.
- Mention failed or blocked methods in the final response.
