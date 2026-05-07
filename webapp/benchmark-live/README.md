# Benchmark Live Site (CSV-Only)

This dashboard ranks recourse methods from precomputed experiment rows in:

- `data/default_results.csv`

and compatibility rules generated from:

- `benchmark/configs/methods/*.yml`

The page does not run experiments. It only filters and ranks rows already in the CSV.

## Run Locally
From repo root:

```powershell
venv12\Scripts\python.exe webapp\benchmark-live\sync_data.py
venv12\Scripts\python.exe -m http.server 8000 --directory webapp\benchmark-live
```

Open:

- `http://127.0.0.1:8000/`

## Expected CSV Columns
The dashboard accepts either the original live-site aggregate schema or the newer benchmark runner schema.

Required filter columns in the aggregate schema:
- `dataset`
- `model`
- `method`

Equivalent benchmark runner columns:
- `dataset_name`
- `model_name`
- `method_name`

Metrics used by ranking:
- `validity` (maximize)
- `distance_l2` (minimize)
- `distance_l0` (minimize)
- `ynn` or `knn_5` (maximize)
- `runtime_seconds` (minimize)

Optional:
- `status` (if present, only rows with `success` or `completed` are used)
- `row_type` (when summary rows are present, the dashboard ranks summary rows)
- `run_duration_seconds` (used as runtime when `runtime_seconds` is absent)

## Refresh Live-Site Data
When `benchmark/configs` or `benchmark/results/default_results.csv` changes, regenerate the live-site data files:

```powershell
venv12\Scripts\python.exe webapp\benchmark-live\sync_data.py
```

This refreshes:
- `webapp/benchmark-live/data/compatibility.json`
- `webapp/benchmark-live/data/default_results.csv`
