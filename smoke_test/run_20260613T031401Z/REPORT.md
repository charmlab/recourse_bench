# RecourseBench Smoke Test Report

- Run directory: `smoke_test/run_20260613T031401Z`
- UTC start/end: `2026-06-13T03:14:01Z` to `2026-06-13T13:44:00Z`
- Branch: `feature/(experiment)_smoke-tests`
- Commit: `39e54ef589f1772de17e046094cfdd9ac9ddfebb`
- Dirty worktree: `yes`
- Python: `/u501/hoyarhos/recourse_bench/.venv/bin/python` (`Python 3.12.4`)
- Requested scope: `all discovered smoke tests`
- Discovered/selected tests: `29` / `29`
- Totals: PASS `29`, FAIL `0`, BLOCKED `0`, SKIPPED `0`
- Overall result: `PASS`

The compatible base requirements were installed or confirmed from
`requirements.txt`, and `pip check` reported no broken requirements.
`alibi==0.9.6` was excluded because `requirements.txt` explicitly documents
that it conflicts with the pinned NumPy version. The `diverse_dist` smoke test
passed without it.

| Method | Status | Reference | Exit | Duration (s) | Evidence | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| apas | PASS | OK | 0 | 35 | [out](apas.out), [err](apas.err), [status](apas.status) | Exact final metric match. |
| arg_ensembling | PASS | OK | 0 | 9 | [out](arg_ensembling.out), [err](arg_ensembling.err), [status](arg_ensembling.status) | Exact final metric match. |
| cchvae | PASS | OK | 0 | 12 | [out](cchvae.out), [err](cchvae.err), [status](cchvae.status) | Exact final metric match. |
| cemsp | PASS | OK | 0 | 7 | [out](cemsp.out), [err](cemsp.err), [status](cemsp.status) | Exact final metric match. |
| cfrl | PASS | DIFFERENT | 0 | 16 | [out](cfrl.out), [err](cfrl.err), [status](cfrl.status) | Completed, but final metrics differ. |
| cfvae | PASS | DIFFERENT | 0 | 28 | [out](cfvae.out), [err](cfvae.err), [status](cfvae.status) | Completed, but final metrics differ. |
| claproar | PASS | OK | 0 | 46 | [out](gpu_retry_20260613T134000Z/claproar.out), [err](gpu_retry_20260613T134000Z/claproar.err), [status](gpu_retry_20260613T134000Z/claproar.status) | CUDA retry; exact final metric match. |
| clue | PASS | DIFFERENT | 0 | 165 | [out](clue.out), [err](clue.err), [status](clue.status) | Completed, but final metrics differ. |
| cogs | PASS | OK | 0 | 13 | [out](cogs.out), [err](cogs.err), [status](cogs.status) | Exact final metric match. |
| cols | PASS | OK | 0 | 109 | [out](cols.out), [err](cols.err), [status](cols.status) | Exact final metric match. |
| cruds | PASS | DIFFERENT | 0 | 84 | [out](gpu_retry_20260613T134000Z/cruds.out), [err](gpu_retry_20260613T134000Z/cruds.err), [status](gpu_retry_20260613T134000Z/cruds.status) | CUDA retry; small final distance drift. |
| cvas_proj | PASS | OK | 0 | 8 | [out](cvas_proj.out), [err](cvas_proj.err), [status](cvas_proj.status) | Exact final metric match. |
| dice | PASS | DIFFERENT | 0 | 111 | [out](dice.out), [err](dice.err), [status](dice.status) | Tiny final distance drift. |
| diverse_dist | PASS | OK | 0 | 7 | [out](diverse_dist.out), [err](diverse_dist.err), [status](diverse_dist.status) | Exact final metric match. |
| face | PASS | OK | 0 | 9 | [out](face.out), [err](face.err), [status](face.status) | Exact final metric match. |
| feature_tweak | PASS | OK | 0 | 21 | [out](feature_tweak.out), [err](feature_tweak.err), [status](feature_tweak.status) | Exact final metric match. |
| gravitational | PASS | OK | 0 | 46 | [out](gpu_retry_20260613T134000Z/gravitational.out), [err](gpu_retry_20260613T134000Z/gravitational.err), [status](gpu_retry_20260613T134000Z/gravitational.status) | CUDA retry; exact final metric match. |
| gs | PASS | OK | 0 | 23 | [out](gs.out), [err](gs.err), [status](gs.status) | Exact final metric match. |
| larr | PASS | OK | 0 | 40 | [out](larr.out), [err](larr.err), [status](larr.status) | Exact final metric match. |
| mace | PASS | OK | 0 | 30 | [out](mace.out), [err](mace.err), [status](mace.status) | Exact final metric match. |
| probe | PASS | DIFFERENT | 0 | 10 | [out](probe.out), [err](probe.err), [status](probe.status) | Completed, but validity and distances differ. |
| proplace | PASS | OK | 0 | 7 | [out](proplace.out), [err](proplace.err), [status](proplace.status) | Exact final metric match. |
| rbr | PASS | DIFFERENT | 0 | 14 | [out](rbr.out), [err](rbr.err), [status](rbr.status) | One-unit last-digit distance drift. |
| revise | PASS | DIFFERENT | 0 | 102 | [out](gpu_retry_20260613T134000Z/revise.out), [err](gpu_retry_20260613T134000Z/revise.err), [status](gpu_retry_20260613T134000Z/revise.status) | CUDA retry; small final distance drift. |
| roar | PASS | DIFFERENT | 0 | 15 | [out](roar.out), [err](roar.err), [status](roar.status) | Completed, but final distances differ. |
| sns | PASS | OK | 0 | 9 | [out](sns.out), [err](sns.err), [status](sns.status) | Exact final metric match. |
| toy | PASS | OK | 0 | 5 | [out](toy.out), [err](toy.err), [status](toy.status) | Exact final metric match. |
| trex | PASS | OK | 0 | 10 | [out](trex.out), [err](trex.err), [status](trex.status) | Exact final metric match. |
| wachter | PASS | OK | 0 | 42 | [out](gpu_retry_20260613T134000Z/wachter.out), [err](gpu_retry_20260613T134000Z/wachter.err), [status](gpu_retry_20260613T134000Z/wachter.status) | CUDA retry; exact final metric match. |

## Failures and Blockers

- No failures or blockers remain.
- `claproar`, `cruds`, `gravitational`, `revise`, and `wachter` were initially
  blocked in the CPU-only allocation, then passed in Slurm job `1451959`.
  The retry requested `gpu:jimmygpu:1` in partition `JIMMY`, ran on
  `watgpu108`, and used an NVIDIA RTX 6000 Ada Generation GPU.

## Reference Comparisons

- OK: `20`
- DIFFERENT: `9`
- NO REFERENCE: `0`
- Successful runs with material metric differences: `cfrl`, `cfvae`, `clue`,
  `probe`, and `roar`.
- Successful runs with small numeric drift: `cruds`, `dice`, `rbr`, and
  `revise`.

## Warnings and Caveats

- This is a functionality smoke test, not a full paper reproduction.
- The worktree was dirty before installation and testing. No method source code
  was modified during this run.
- The initial Slurm allocation had no visible GPU. CUDA-reference methods were
  not silently run on CPU; they were retried with a requested GPU on
  `watgpu108`.
- The optional `alibi==0.9.6` dependency is incompatible with the repository's
  pinned `numpy==2.4.2`; the repository's `diverse_dist` implementation still
  passed its smoke test without `alibi`.

## Artifacts

- [Installation log](install.log)
- [Environment](environment.txt)
- [Git status](git_status.txt) and [commit](git_head.txt)
- [Discovered tests](discovered_tests.txt) and [selected tests](selected_tests.txt)
- Per-method `.out`, `.err`, and `.status` files in this directory
- CUDA retry: [job ID](gpu_retry_20260613T134000Z/job_id.txt),
  [environment](gpu_retry_20260613T134000Z/environment.txt),
  [Slurm stdout](gpu_retry_20260613T134000Z/slurm-1451959.out),
  [Slurm stderr](gpu_retry_20260613T134000Z/slurm-1451959.err), and
  [submission script](gpu_retry_20260613T134000Z/run_gpu_retry.sbatch)
