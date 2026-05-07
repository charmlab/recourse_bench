# Benchmark Batch 2x2x5

This folder contains a reproducible benchmark batch with:
- 2 datasets: `toydata`, `breast_cancer`
- 2 models: `linear`, `mlp`
- 5 methods: `toy`, `wachter`, `gs`, `face`, `claproar`
- evaluations: `validity` + `distance` (`l0`, `l1`, `l2`, `linf`) + `ynn` + `runtime_seconds`

## Config Files
Per-experiment YAMLs are in `configs/` (20 files total).

## Run
Use Python 3.12 environment:

```powershell
& python experiment\benchmark_batch_2x2x5\run_batch.py
```

Optional arguments:
- `--config-dir <path>`
- `--output-csv <path>`
- `--stop-on-error`

## Output
By default the consolidated CSV is written to:

`results/benchmark_batch_2x2x5/benchmark_results.csv`

The CSV includes one row per YAML run, plus status/error info when a run fails.
