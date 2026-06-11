# Task: Run RecourseBench Reproduction Checks

Follow the `recoursebench-reproduction` skill exactly to run the minimum successful reproduction check for every method, capture results, and produce a summary report.

## Requirements

- Use the `recoursebench-smoke-tests` skill first to resolve the Python environment; do not reinstall if it already satisfies requirements.
- Follow the Minimum Successful Checks table in the `recoursebench-reproduction` skill — run the listed command for each method.
- Capture stdout, stderr, and a status file for each method under the run directory.
- After all runs, compare each result against the bundled `experiment/<method>/reproduce_logs.txt` reference using the comparison rules in the skill.
- Write a final report that states, for each method: command, exit code, status (Reproduced / Smoke OK / Scoped OK / Failed), duration, key reproduced metrics, and reference comparison outcome.

## Success Criteria

A run is considered successful when:
1. A run directory exists with per-method `.out`, `.err`, and `.status` files.
2. At least one method reached a terminal state and was compared against its reference log.
3. The final report documents every method attempted, including failures and blockers.

Do not claim full reproduction from a scoped fallback. Report the exact result.
