# Reproduction Log Fields

Each method stores its reproduction record in:

```text
experiment/<method>/reproduce_logs.txt
```

When the initial repository does not contain a log, create one using only
information available in that method's initial `experiment/<method>/` files.
Do not use run output or another external source. If those files contain no
recorded result, use `No value available.` for the result fields.


All log files use the same fields in the order documented below. The format
standardizes where information appears; it does not require every field to
have a value.

## Method Coverage

`Smoke` and `Reproduce` indicate whether the initial repository contains
`smoke.py` or `test_<method_name>_reproduce.py` for the method. `Observed Result` and
`Paper Result` indicate whether the corresponding log field contains a
recorded value. The table uses only `Yes` and `No`.

| Method | Smoke | Reproduce | Observed Result | Paper Result |
| --- | --- | --- | --- | --- |
| APAS | No | Yes | Yes | Yes |
| Argumentative Ensembling | No | Yes | Yes | Yes |
| CCHVAE | No | Yes | Yes | No |
| CEMSP | No | Yes | Yes | Yes |
| CFRL | No | Yes | Yes | Yes |
| CFVAE | No | Yes | Yes | Yes |
| CLAPROAR | Yes | No | No | No |
| CLUE | No | Yes | Yes | Yes |
| CoGS | No | Yes | Yes | Yes |
| COLS | No | Yes | Yes | Yes |
| CRUDS | Yes | No | No | No |
| CVAS-PROJ | No | Yes | Yes | Yes |
| DiCE | No | Yes | Yes | Yes |
| DiverseDist | No | Yes | Yes | Yes |
| FACE | No | Yes | Yes | No |
| FeatureTweak | Yes | No | No | No |
| Gravitational | Yes | No | No | No |
| GS | No | Yes | Yes | Yes |
| LARR | No | Yes | Yes | No |
| MACE | No | Yes | No | Yes |
| PROBE | No | Yes | Yes | Yes |
| ProPlace | No | Yes | Yes | Yes |
| RBR | No | Yes | Yes | Yes |
| REVISE | Yes | No | No | No |
| ROAR | No | Yes | Yes | No |
| SNS | No | Yes | Yes | Yes |
| TreX | No | Yes | Yes | Yes |
| Wachter | Yes | No | No | No |

No method has both `smoke.py` and `test_<method_name>_reproduce.py`, and no listed method has
neither file.

## Fields

### Level Badge

The assigned reproduction level and its short label, for example:

```text
Level 2: Partial Reproduction
```

Use `No value available.` when no level has been assigned.

### Reason

The explanation for the level badge. It should state why the available
results support that level and identify important limitations.

### Method

The method name and, when already documented, a short description of the
implementation, dataset, or experiment being reproduced.

### Reproduction Summary

A high-level summary of the reproduction. This may include aggregate
statistics, execution details, the tested configuration, or a brief account
of what was reproduced.

### Reference Scope

The exact scope used for comparison, such as:

- dataset
- model
- target table, figure, or experiment
- configuration or hyperparameters
- number of runs, folds, factuals, or seeds
- whether the run is full, bounded, scoped, or smoke-only

This field prevents a limited run from being interpreted as full-paper
coverage.

### Observed Result

Results recorded from the reproduction run or existing reproduction log.
These may be scalar metrics, tables, structured output, or qualitative
observations.

Keep values at their recorded precision. Do not infer, recalculate, or add
results merely to fill this field.

### Reproduction Assessment

The existing evaluation of whether the observed results reproduce the paper
or reference results under the stated criterion. This may identify matches,
deviations, or results that fall within or outside a tolerance.

Do not create a new assessment from the observed and paper values merely to
fill this field. Use `No value available.` when the source log contains no
assessment.

### Paper Result

The paper, notebook, reference repository, configuration, or other documented
target used for comparison. This field may also contain an existing comparison
between observed and reference results.

The source of the reference should be named when it is already known.

### Scalar Result Coverage

Counts describing the scalar comparison set, commonly:

- `R`: number of scalar reference results considered
- `R_rep`: number reproduced within the stated tolerance
- `R - R_rep`: number outside the tolerance

Do not derive these counts unless the log already contains them or the
reproduction procedure explicitly produces them.

### Reproducibility Criterion

The rule used to decide whether a result was reproduced. This may define:

- the comparison unit
- an error or distance formula
- numerical-stability constants
- a fixed tolerance
- any coverage requirement

### Notes / Caveats

Interpretation details, known limitations, environment constraints, missing
artifacts, unsupported settings, or other qualifications that do not belong
in the primary result fields.

### Next Steps

Remaining work required to improve coverage, investigate deviations, or
support a stronger reproduction claim.

## Missing Values

When a field has no recorded value, write:

```text
No value available.
```

Do not search other files for a replacement value unless that work is
explicitly requested. Do not invent, infer, calculate, or summarize new
information solely to populate a missing field.

Existing text such as `EMPTY` may be retained when it is part of the original
recorded content.

## Content Preservation

When standardizing a log:

1. Do not delete or rewrite existing factual content.
2. Rename legacy headings to the closest standard field.
3. Move content under the appropriate field when necessary.
4. Preserve numeric values and their precision.
5. Preserve reasons, caveats, comparison text, and execution details.
6. Add only a field heading and `No value available.` when the field is absent.

Common legacy heading mappings include:

| Legacy heading | Standard field |
| --- | --- |
| `Summary` | `Reproduction Summary` |
| `Reproduction Results` | `Observed Result` |
| `Reference setting` or `Reference scope` | `Reference Scope` |
| `Observed Result Highlights` | `Observed Result` |
| `Paper Comparison` | `Paper Result` |
| `Paper / reference comparison` | `Paper Result` |
| `Paper / repo comparison` | `Paper Result` |
| `Paper / notebook comparison` | `Paper Result` |

## Template

```text
Level Badge
No value available.

Reason
No value available.

Method
No value available.

Reproduction Summary
No value available.

Reference Scope
No value available.

Observed Result
No value available.

Reproduction Assessment
No value available.

Paper Result
No value available.

Scalar Result Coverage
No value available.

Reproducibility Criterion
No value available.

Notes / Caveats
No value available.

Next Steps
No value available.
```
