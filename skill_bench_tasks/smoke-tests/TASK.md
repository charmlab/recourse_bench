# Task: Run RecourseBench Smoke Tests

Follow the `recoursebench-smoke-tests` skill exactly to run the smoke tests for all discovered methods, save auditable results, and write a consolidated Markdown report.

## Requirements

- Follow every step of the `recoursebench-smoke-tests` skill without omissions.
- Discover smoke tests dynamically from the repository rather than using a hardcoded list.
- Run all discovered smoke tests and continue after any failure so the report covers the full scope.
- Save all results under the default output directory (`smoke_test/run_<timestamp>/`).
- Write `REPORT.md` in that directory with the full result table, failures/blockers section, reference comparisons, and artifacts list.
- Include per-method `.out`, `.err`, and `.status` files.
- Save context files: `git_status.txt`, `git_head.txt`, `environment.txt`, `discovered_tests.txt`, `selected_tests.txt`.

## Success Criteria

A run is considered successful when:
1. `smoke_test/run_*/REPORT.md` exists and contains a complete result table.
2. Per-method log files (`.out`, `.err`, `.status`) exist for every tested method.
3. All required context files are present.
4. The report's overall result line is either `PASS` or `ISSUES FOUND` — not absent.

Do not modify method code to force a passing result. Report the exact outcome.
