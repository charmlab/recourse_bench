# Getting started

## Install

Install from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple \
    recourse-bench
```

Or install from a source checkout:

```bash
pip install -r requirements.txt
```

## Minimal config

RecourseBench experiments are configured as data. A config chooses a dataset,
preprocessing, target model, recourse method, and evaluation metrics.

```yaml
name: toy_linear_wachter
seed: 7
dataset:
  name: toydata
preprocess:
  - name: scale
    scaling: standardize
    range: true
  - name: split
    split: 0.25
model:
  name: linear
  device: cpu
method:
  name: wachter
  device: cpu
  desired_class: 1
evaluation:
  - name: validity
  - name: distance
```

## Run

From the command line:

```bash
recourse-bench -p experiment/toy/smoke_config.yaml
```

From Python:

```python
import yaml
from pathlib import Path
import recourse_bench as rb

config = yaml.safe_load(Path("experiment/toy/smoke_config.yaml").read_text())
metrics = rb.run(config)
print(metrics.to_string(index=False))
```

Use the registry helpers to discover valid component names:

```python
rb.list_datasets()
rb.list_models()
rb.list_methods()
rb.list_evaluations()
```

`rb.run(config)` trains the target model, fits the recourse method, generates
counterfactuals, and returns a one-row metrics dataframe.
