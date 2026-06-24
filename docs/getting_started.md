# Getting started

## Install

```bash
conda create -n recoursebench python=3.12
conda activate recoursebench
pip install -r requirements.txt
# Optional: only needed for some methods (e.g. diverse_dist). Downgrades numpy.
pip install alibi==0.9.6
```

## Run an experiment from the CLI

After installation the `recourse-bench` entry point runs any experiment config:

```bash
recourse-bench -p experiment/toy/smoke_config.yaml
```

From a source checkout, `python experiments.py -p <config>` is equivalent.

## A minimal config

An experiment is fully described by a config dictionary (usually YAML). A single
top-level `seed` is propagated to every component that does not set its own.

```yaml
name: credit_linear_wachter_smoke
seed: 7                       # propagated to dataset/model/method/preprocess
logger:
  level: INFO
  path: ./logs/credit_linear_wachter_smoke.log
caching:
  path: ./cache/
dataset:
  name: credit
preprocess:
  - name: balance
    strategy: downsample
  - name: encode
    encoding: onehot
  - name: scale
    scaling: normalize
  - name: split
    split: 0.3
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

`finalize` is appended automatically if omitted, and must be the last
preprocessing step. `model.device` and `method.device` must match.

## The Python API

Import the package as `rb`. The quickest path is `rb.run(config)`:

```python
import yaml
from pathlib import Path
import recourse_bench as rb

config = yaml.safe_load(Path("experiment/toy/smoke_config.yaml").read_text())
metrics = rb.run(config)
print(metrics.to_string(index=False))
print(metrics.attrs["provenance"])     # library version, config hash, seed, ...
```

### Construct components by name

Each registered component is reachable through a named namespace; the attribute
*is* the class, so call it to construct an instance:

```python
model  = rb.models.linear(seed=7)
method = rb.methods.wachter(target_model=model, seed=7, desired_class=1)
data   = rb.datasets.credit()

rb.list_methods()      # discover what's available
```

See {doc}`reference/using` for the full list of namespaces.

### Full control with `Experiment`

When you also want the trained model, the counterfactuals, or provenance, use
the {class}`~experiments.Experiment` class:

```python
experiment = rb.Experiment(config)
metrics = experiment.run()

model = experiment.target_model()       # the trained classifier
cfs   = experiment.counterfactuals()    # the generated counterfactuals
train = experiment.trainset()
```

Configuration problems raise {class}`~utils.exceptions.ConfigError` (a subclass
of {class}`~utils.exceptions.RecourseBenchError`) instead of exiting the
process, so they can be caught programmatically.

## What `run()` does

`Experiment.run()` performs preprocessing, resolves the train/test datasets,
trains the target model, fits the recourse method, generates counterfactuals,
and concatenates evaluation outputs into one metrics dataframe. Failed
counterfactual rows are represented with `NaN` feature values and target `-1`.
