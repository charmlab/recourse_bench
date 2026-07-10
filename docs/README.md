# RecourseBench MCP server (optional)

A small, **local-first** [Model Context Protocol](https://modelcontextprotocol.io)
server that lets coding agents use RecourseBench through a few controlled tools.
It is a thin wrapper over the canonical public API and is **not** the primary way
to use the library.

For direct use, call the Python API — the MCP server is only for agent access:

```python
import recourse_bench as rb

metrics = rb.run(config)
metrics = rb.run_config_file(path)
rb.list_datasets(); rb.list_methods(); ...
```

## Install

The server needs the optional `mcp` extra (the core library installs fine
without it):

```bash
pip install -e ".[mcp]"
```

Or, from the published package:

```bash
pip install "recourse-bench[mcp]"
```

## Run

Over stdio (the default and only transport):

```bash
recourse-bench-mcp            # console script
# or
python -m recourse_bench.mcp_server
```

`recourse-bench-mcp --help` shows the (minimal) options.

### Example local MCP client config

For a stdio MCP client (e.g. Claude Desktop), add a server entry like:

```json
{
  "mcpServers": {
    "recourse-bench": {
      "command": "recourse-bench-mcp"
    }
  }
}
```

If the console script is not on `PATH`, use the module form instead:

```json
{
  "mcpServers": {
    "recourse-bench": {
      "command": "python",
      "args": ["-m", "recourse_bench.mcp_server"]
    }
  }
}
```

## Tools

| Tool | Input | Returns |
| --- | --- | --- |
| `list_recourse_components` | `kind`: one of `datasets`, `preprocess`, `models`, `methods`, `evaluations` | `{kind, components, count}` |
| `get_example_config` | `dataset="toydata"`, `model="linear"`, `method="wachter"`, `seed=7`, `quick=true` | `{valid, config, notes}` (or `{valid: false, errors, available}`) |
| `validate_recourse_config` | `config: dict` | `{valid, errors, warnings}` — structure only, no run |
| `run_recourse_experiment` | `config: dict` | `{ok, metrics, columns, provenance}` (or structured error) |
| `run_recourse_config_file` | `path: str` (under cwd/repo only) | same as `run_recourse_experiment` |
| `run_smoke_test` | `method: str`, `compare_to_baselines=false`, `baseline_path=None`, `metrics=None`, `tolerances=None` | `{ok, method, config_path, metrics, columns, provenance, runtime_seconds}` (+ `baseline_comparison` when comparing) |
| `run_benchmark_pack` | `pack="small"`, `method=None` | `{ok, pack, method, results:[...], warnings}` |

All component names returned/accepted come straight from the `rb.list_*`
registries, so they never drift from what the library actually supports.

### `run_smoke_test` — single-method health check

Runs **one** method's smoke config
(`experiment/<method>/smoke_config.yaml`) exactly once through the public
`rb.run(...)` API, after validating it. It is a lightweight *health check* — is
this method wired up and producing metrics? — **not** a benchmark. It never runs
sweeps and never runs more than one method.

Returned fields: `ok`, `method`, `config_path`, `metrics` (list of JSON-safe
rows), `columns`, `provenance`, `runtime_seconds`, and a structured `error`/`errors`
on failure.

```text
"Check if Wachter works."
  -> run_smoke_test(method="wachter")
```

#### `compare_to_baselines` — smoke-level regression check

When `compare_to_baselines=true`, the smoke run above is *also* diffed against a
saved smoke result. This is a **smoke-level regression check** ("does Wachter
still match the number we saved?"), not a full benchmark regression suite.

* Baseline discovery: an explicit `baseline_path` (validated for safety) wins;
  otherwise the tool auto-discovers a method-specific baseline by convention,
  first found of:
  * `experiment/<method>/<method>_smoke_log.txt` — the canonical baseline,
    written by `experiment/run_smoke.py`; its trailing lines are the
    pandas-printed metrics table.
  * `results/baselines/<method>_smoke.json`
  * `benchmark/results/baselines/<method>_smoke.json`
  * `experiment/<method>/baseline_result.json`
* A baseline may be a **smoke log** (`.txt`/`.log`, parsed from the trailing
  metrics table) or **JSON** (`.json`, either a flat `{metric: value}` mapping or
  the smoke-tool output shape `{"metrics": [ {...} ], ...}`).
* Because logged values are rounded (pandas prints ~6 significant digits), the
  default `1e-6` tolerance is chosen to absorb that rounding for a deterministic
  method; tighten or loosen it per metric via `tolerances`.
* Metrics compared: `metrics` if given, else the intersection of numeric metric
  columns. **Runtime is never compared** unless you list it in `metrics`.
* Tolerances: `tolerances` if given (per-metric, plus an optional `"default"`
  key), else a small absolute tolerance of `1e-6`.
* A missing baseline is a **warning**, not an error (`passed: null`).
* Baseline files are **never written or updated** by this tool.

The result gains a `baseline_comparison` block:

```json
{
  "baseline_comparison": {
    "enabled": true,
    "baseline_path": "results/baselines/wachter_smoke.json",
    "passed": true,
    "current":  {"validity": 0.82, "distance_l1": 1.31},
    "baseline": {"validity": 0.82, "distance_l1": 1.31},
    "diffs":    {"validity": 0.0,  "distance_l1": 0.0},
    "tolerances": {"validity": 1e-6, "distance_l1": 1e-6},
    "failures": [],
    "warnings": []
  }
}
```

```text
"Check if Wachter still matches our saved smoke result."
  -> run_smoke_test(method="wachter", compare_to_baselines=True)

"Check Wachter against a specific baseline."
  -> run_smoke_test(method="wachter", compare_to_baselines=True,
                    baseline_path="results/baselines/wachter_smoke.json")
```

### `run_benchmark_pack` — bounded benchmark pack

Runs a **bounded** benchmark *pack*: a small suite at
`benchmark/configs/suites/<pack>.yaml`. Each `(dataset, model, method)` run is
**composed on top of `benchmark/configs/base.yaml`** (merging the `datasets/`,
`models/` and `methods/` component configs, exactly like `benchmark/run.py`),
then validated and run once via `rb.run(...)`. Harness-only keys are dropped,
`device` is forced to `cpu`, and the test split is bounded (`pack.split_sample`)
so each run stays fast. The shipped `small` pack is toydata/german on `linear`
with `gs`/`dice`.

* At most `10` runs per pack (excess is capped with a warning); one run each;
  **no sweeps, no multi-seed suites**.
* With `method` set, the method must be registered; it then **replaces** the
  suite's method grid — each `(dataset, model)` is run with that method (using
  its `benchmark/configs/methods/<method>.yaml` if present, else its registered
  defaults).
* A run that is **incompatible** (per a method's `compatibility` allow-lists),
  invalid, or fails is **skipped with a structured warning** — one bad run never
  crashes the whole pack.
* Add a new pack by dropping a `benchmark/configs/suites/<name>.yaml` next to
  `small.yaml`; no code changes needed.

```text
"Run the small benchmark."
  -> run_benchmark_pack(pack="small")

"Compare my new method to the small benchmark."
  -> run_benchmark_pack(pack="small", method="my_new_method")
```

## Safety notes

- **No arbitrary Python execution.** Tools only call the public `rb.*` API with
  validated configs; there is no eval/exec surface.
- **No expensive sweeps.** `run_recourse_experiment` runs exactly one config.
  `get_example_config` defaults to small CPU smoke settings (`quick=true`).
  `run_smoke_test` runs exactly one method's smoke config. `run_benchmark_pack`
  runs a bounded pack (≤ 10 single-run configs); it does **not** run the full
  benchmark suite, multi-seed sweeps, or `benchmark/configs/suites/default.yaml`.
- **Restricted baseline reads.** `run_smoke_test` baseline comparison only reads
  `.json`/`.txt`/`.log` files under repo-local result directories (`results/`,
  `benchmark/results/`, `experiment/`); paths escaping the repo or those
  directories are rejected with a warning. There is no arbitrary file read, and
  baselines are never written or overwritten.
- **CPU-first defaults.** Example configs use `device: "cpu"`.
- **JSON-serialized results.** Metrics are converted from pandas/numpy to plain
  JSON; `NaN`/`inf` become `null`. Raw DataFrames are never returned.
- **Path restriction.** `run_recourse_config_file` only reads files under the
  current working directory or the repository root; absolute paths elsewhere are
  rejected.
- **Validation before run.** `run_recourse_experiment` validates structure and
  component names first and refuses to run an invalid config.

For anything beyond this controlled surface (adding methods, sweeps,
reproductions, internals), use the library and repo directly.

## Verify

A lightweight check that exercises the tool logic without an MCP client:

```bash
python examples/test_mcp_tools.py  # or: pytest examples/test_mcp_tools.py
```

This covers component discovery, config validation, a tiny experiment run,
`run_smoke_test` (with baseline match / out-of-tolerance / missing-baseline /
unsafe-path cases), and `run_benchmark_pack` (discovery, method injection, and
safe failures) — all without a live MCP client.

## Limitations / intentionally not supported

- No arbitrary Python/shell execution and no arbitrary file reads.
- `run_smoke_test` is a single-method health check; it never runs multiple
  methods, and its baseline comparison is a **smoke-level regression check, not a
  full benchmark regression suite**. It does not write or update baselines.
- `run_benchmark_pack` runs only a bounded pack (≤ 10 single-run configs, CPU
  defaults). Full suites and multi-seed sweeps are out of scope by design; run
  those from the repo directly (`benchmark/run.py`).
- stdio transport only.
- `run_recourse_config_file` path policy is intentionally conservative (cwd/repo
  only). Loosen deliberately if you need to read configs from elsewhere.
