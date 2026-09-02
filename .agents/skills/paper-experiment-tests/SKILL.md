---
name: paper-experiment-tests
description: Use when extracting the experimental results a recourse method's OWN source paper reports (e.g. the ROAR paper for the roar method) into a single reproduction_report.json in RecourseBench's schema, then comparing it against the method's committed report. Nothing is executed — only the paper's numbers are recorded.
metadata:
  version: v0.5.0
---

# Paper-Derived Experiment Report

Read the recourse method's **own source paper** and record every number it reports for
that method in one JSON file, using the repo's `reproduction_report.json` schema. Then
compare that file against the method's committed report and write the comparison.

**Nothing is run.** You produce only `original` — the paper's values. `reproduced` and
`delta` come from an actual run and stay `null`.

Operate on a source checkout of `recourse_bench`; all paths are repo-relative and
commands run from the repo root.

---

## Output — exactly two files

```text
experiment/<method>/_paper_tests/reproduction_report.json   # the paper's numbers
experiment/<method>/_paper_tests/notes.md                   # every gap and judgement call
experiment/<method>/_paper_tests/comparison.md              # written by the comparator
```

Produce nothing else — no configs, no runner, no coverage report, no logs, and never
touch `experiment/<method>/reproduction_report.json`.

---

## Which paper

The paper that **introduced this method** — the ROAR paper for `roar`, the DiCE paper
for `dice`. Not the RecourseBench paper, not a survey, not a citing method's paper.

To find it: a citation/DOI/arXiv id in `method/<method>/` docstrings or a README; then
method docs or the registry entry; then web search (registered name + "algorithmic
recourse" / "counterfactual explanation", preferring arXiv); then the method's own
reference implementation. Cross-check that its datasets, metrics, and method
description match. If several papers plausibly fit, or the id resolves to a different
method, **stop and ask**. If none is found, say so and stop.

**Do not read the repo's answers.** `experiment/<method>/reproduction_report.json` is
the answer key you are scored against; `reproduce_logs.txt`, existing
`*_reproduce.yaml`, and stored `paper_targets` are copies of it. Never open any of
them — every number must be your own read of the paper.

---

## Extracting a number faithfully

- **Pin the exact coordinate**: `table N, row <label>, col <label>`, not "the paper's
validity". The method's *default configuration* maps to exactly one cell.
- **Record the operating point.** A metric conditioned on a knob (invalidation rate,
σ², k, β/γ, threshold) is a different result at a different setting — put the knob in
`configuration`.
- **Check absolute vs. relative, and units.** A number that matches no row is often
the other representation (%-improvement vs. absolute, ℓ0/ℓ1/ℓ2/ℓ∞).
- **Prefer table > text > figure.** Reference code or released logs are a fallback, not
a paper value.
- **Never guess.** A value the paper does not state is `null` plus a line in
`notes.md` — never a plausible-looking number. An invented value is worse than a gap:
it is indistinguishable from a real extraction and silently corrupts the comparison.
- **Mind blind spots.** One-pass PDF extraction misses figure-only and appendix-only
tables; "the paper reports no number" ≠ "my pass didn't reach it". Say which it is.

## Take every experiment

Record **every experiment the paper presents for this method** — main results,
ablations, sensitivity sweeps, appendix tables, figure-only results — one
`experiments` entry each. Sweep the whole paper before writing: main tables, ablation
sections, appendices, supplementary material.

An experiment you can only partly read still gets its entry. Put `null` in whatever
field the paper does not give you and note it — a dataset you cannot map to a
registered name, a metric reported only as a bar in a figure, a knob whose setting the
caption omits. Never drop an experiment for being incomplete, and never pad one to look
complete.

For every `null`, `notes.md` gets a line saying which experiment and field, and which
kind of gap it is:

```markdown
- `german_credit.sns_l2_cost` — figure 4 only, no numeric value in the text
- `adult_ablation.configuration.dataset` — paper says "a subsampled Adult"; no
  registered dataset matches, and the subsampling is not described
- `compas.validity` — appendix table referenced as Table 7 but absent from the PDF
```

Keep `notes.md` to these lines plus, if useful, a short paragraph on how you settled
the paper's identity. It is a record of gaps, not an essay.

---

## The JSON

Two top-level keys, nothing more. `reproduction_metadata` is run bookkeeping
(timestamps, absolute paths, run limits) — omit it entirely.

```json
{
  "paper_id": "<method>_<short-slug>",
  "experiments": {
    "<dataset>[_<condition>...]": {
      "configuration": {
        "dataset": "<registered dataset>",
        "display_name": "<paper's name for it>",
        "<operating point param>": "<value>"
      },
      "metrics": {
        "<metric_name>": { "original": <paper value>, "reproduced": null, "delta": null }
      }
    }
  }
}
```

Rules:

- A `metrics` entry has **exactly** `original`, `reproduced`, `delta`. `original` is
the paper's number, or `null` when the paper reports the metric but you cannot read a
value — noted in `notes.md`. `reproduced` and `delta` are always `null`.
- Anything else in `configuration` may be `null` too, `dataset` included, when the
paper does not pin it down. Note every one.
- **No run-only diagnostics.** The committed reports carry metrics with
`"original": null` — `baseline_accuracy`, `main_model_test_auc`, cache counters —
because a run emitted them, not because the paper reports them. Add a metric only if
the paper presents it; a `null` means "the paper reports this and I could not read the
number", never "a run would produce this".
- One `experiments` entry per paper experiment; one `metrics` entry per reported
scalar. Mean ± std is two entries (`cost_mean`, `cost_std`), not one string.
- Keep the paper's units and precision: `100.0` if it prints percent, `0.36` not `0.4`.
Record the choice in `notes.md` whenever the paper prints percentages — a report that
stores the same metric as a fraction disagrees by ~0.99, which is a representational
difference, not a misread number, and the note is what tells the two apart.

**Experiment keys** — lowercase `snake_case`, leading with the paper's own name for the
setting, then the conditions distinguishing this row from its siblings:
`german_credit`, `correction_linear_l1`, `diabetes_norm_1_opt_1`. That lead token must
appear in `configuration` — usually the dataset, but ROAR names its experiments by
shift (`correction`) while `configuration.dataset` is `german_roar`, which is fine.
`configuration.dataset` holds the registered dataset name when one exists.

**Metric names** — reuse the repo's vocabulary where the paper's scalar means the same
thing; the comparator matches by name, so a synonym scores as a miss:

```text
validity  delta_validity  cost_mean/_std  time_mean/_std  l1  feasibility_mean  avg_bound
current_validity_mean/_std  future_validity_mean/_std  m1_validity_mean/_std  m2_validity_mean/_std
k_distance_mean/_std  k_diversity_mean/_std  set_distance_sum_mean/_std  set_distance_max_mean/_std
lambda_mean/_std
```

Prefix a variant when the paper's row compares variants (`base_l2_cost` vs
`sns_l2_cost`); otherwise use a bare `snake_case` name of the paper's metric.

---

## Compare

Only after the JSON is final:

```bash
python experiment/compare_paper_tests_to_gt.py <method>
```

It validates the shape, aligns experiments and metrics against the committed report,
writes `comparison.md`, and prints extraction accuracy, coverage over paper-valued
metrics, and every mismatch. It already forgives one difference automatically —
**decimal truncation**: a value within one unit of the paper's last printed decimal
(`1.38` for a run's `1.389183`) is scored `printed`, not `mismatch`, so those need no
action beyond noticing them.

**Classify every surviving mismatch in `notes.md`.** A raw delta does not say whether
the extraction was wrong. Before treating a mismatch as a real disagreement, check the
two systematic causes and, when one explains it, write a `notes.md` line saying which:

- **Scale.** The same quantity on a different unit — validity as `100` (percent) vs
`1.0` (fraction), a cost as a percentage-improvement vs an absolute. The tell is a
constant ratio (×100, ×1/100) across a whole metric or experiment. This is a
representational choice, not a misread; note which unit the paper prints and which the
ground truth stores, and **do not rescale your value to match** — that would hide a
convention the human needs to decide on.
- **Decimal truncation** beyond what the tool already forgives — e.g. the paper prints
a coarser precision than one unit covers, or a `±0.00` std whose true value is nonzero.
Note it as a precision artifact, not an error.

A line per classified mismatch, e.g.:

```markdown
- `adult.validity` — scale: paper prints 100 (percent), ground truth stores 1.0
  (fraction); values agree up to the ×100 factor
- `seizure.time_std` — truncation: paper prints ±0.00, run measured ±0.004
```

What is left after removing scale and truncation is the genuine finding: a number the
extraction and the ground truth actually disagree on. Leave those unexplained in
`notes.md` and flagged in `comparison.md`.

**Never edit the JSON after running the comparison.** It shows the answers; revising
your extraction to match — including rescaling to erase a scale difference — turns a
measurement into a copy. Fix genuine format errors (`--schema-only` catches those
before the comparison) and leave the numbers alone; a mismatch is a finding.
