# RecourseBench
## Code for _RecourseBench: A Modular Framework for Reproducible Algorithmic Recourse Evaluation_

## Install
Install PyTorch for your machine first, then RecourseBench. Which torch build
you pick dominates the install size, so it is a deliberate step rather than
something this project decides for you.

```bash
conda create -n recoursebench python=3.12
conda activate recoursebench

# 1. PyTorch — pick ONE (skip on macOS/Windows, where PyPI wheels are CPU-only)
pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cpu     # ~500 MB
pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu126   # ~4.5 GB, needs an NVIDIA GPU

# 2. RecourseBench and the rest
pip install -r requirements.txt
# If you want to run diverse_dist, then override install alibi:
pip install alibi==0.9.6
```

See <https://pytorch.org/get-started/locally/> for ROCm and other CUDA versions.
Use `--index-url`, not `--extra-index-url` — the latter leaves pip resolving
across both PyPI and the PyTorch index with no guarantee which one a given
package comes from.

`requirements.txt` installs every solver backend, which is what you want for a
source checkout or to reproduce the full benchmark. Installing the package
itself keeps the solvers optional:

```bash
pip install recourse_bench                 # core, no solver backends
pip install recourse_bench[all-methods]    # + every solver backend
pip install recourse_bench[gurobi,lime]    # + just the backends you need
```

| Extra | Package | Methods |
| --- | --- | --- |
| `gurobi` | gurobipy (needs a Gurobi license) | `apas`, `proplace` |
| `z3` | z3-solver | `cemsp` |
| `smt` | PySMT | `mace` |
| `asp` | clingo | `arg_ensembling` |
| `cvx` | cvxpy | `cvas_proj` |
| `art` | adversarial-robustness-toolbox | `sns`, `trex` |
| `lime` | lime | `roar`, `larr` |

Every method stays registered and listed by `rb.list_methods()` whether or not
its backend is installed — constructing one without it raises
`MissingDependencyError` naming the extra to install.

**On Linux, skipping the torch step above gives you the CUDA build**: PyPI's
default `torch` wheel pulls ~3.1 GB of `nvidia-*` wheels, taking the install from
~500 MB to ~4.5 GB. If you do not have an NVIDIA GPU, that is ~4 GB you will
never use.

## Where the cache goes

Trained models and other derived artifacts are cached outside the working
directory, under (in order) `$RECOURSE_BENCH_CACHE`,
`$XDG_CACHE_HOME/recourse_bench`, or `~/.cache/recourse_bench`. Set
`caching.path` in a config to keep a run's cache alongside its outputs instead.

## Documentation

Full documentation — a getting-started guide, an extension guide, and a
numpy-style API reference for the public classes and functions (arguments,
return values, and examples) — is published at
<https://charmlab.github.io/recourse_bench/docs/>.

To build the site locally:

```bash
pip install -e ".[docs]"      # or: pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
# open docs/_build/html/index.html
```

## Supported Components
### Datasets

`adult`, `adult_cfrl`, `adult_cfvae`, `adult_cogs`, `boston_housing`, `breast_cancer`, `compas`, `compas_carla`, `compas_clue`, `credit`, `credit_cchvae`, `diabetes`, `german`, `german_roar`, `german_sns`, `hepatitis`, `news_popularity`, `synthetic_face`, `toy_data`.

Each dataset lives under `dataset/<name>/` and should provide an offline data file plus metadata YAML. Dataset objects are initialized as raw, non-encoded, non-scaled datasets; encoding, scaling, splitting, freezing, ... belong to preprocessing.

### Preprocess

`balance`, `encode`, `scale`, `reorder`, `split`, `finalize`.

Typical usage is a pipeline such as `balance -> encode -> scale -> split -> finalize`, but you can always add customed ones. If `finalize` is omitted from a config, `Experiment` appends it automatically. `finalize` must be the last preprocessing step.

### Target Models

`linear`, `mlp`, `mlp_bayesian`, `random_forest`, `sklearn_logistic_regression`.

Torch models expose `forward()` for gradient-based methods. Tree/sklearn models generally do not support differentiable recourse. `model.device` and `method.device` must match.

### Methods

`apas`, `arg_ensembling`, `cchvae`, `cemsp`, `cfrl`, `cfvae`, `claproar`, `clue`, `cogs`, `cols`, `cruds`, `cvas_proj`, `dice`, `diverse_dist`, `face`, `feature_tweak`, `gravitational`, `gs`, `larr`, `mace`, `probe`, `proplace`, `rbr`, `revise`, `roar`, `sns`, `toy`, `trex`, `wachter`.

Method compatibility depends on the paper and implementation: some methods require differentiable models, some require sklearn/tree models, and some prefer to use method-specific datasets or artifacts. Recommand to check the method's YAML and script under `experiment/<method>/` before changing model or preprocessing choices.

### Evaluation

`validity`, `distance`, `constraint`, `knn`, `example`.

Evaluation receives the finalized factual dataset and the counterfactual dataset returned by `MethodObject.predict()`. Failed counterfactual rows are represented with `NaN` feature values and target `-1`.

## Writing New Components

For users, the framework has five extensible object types. New classes should be registered, and imported from the corresponding package `__init__.py` so `Experiment` can discover them. If it is across different abstract object types, the registered name can be the same.

### DatasetObject

Implement `dataset/<name>/<name>.py` and `dataset/<name>/<name>.yaml`. Put data files under `dataset/<name>/xxx'.

```python
from dataset.dataset_object import DatasetObject
from utils.registry import register


@register("custom")
class CustomDataset(DatasetObject):
    def __init__(self, path: str = "./dataset/custom/", **kwargs):
        self._rawdf = self._read_df(path)
        for key, value in self._read_attrs(path).items():
            setattr(self, key, value)
        # Save more setting args and do whatever you want
        ...

    def _read_df(self, path: str):
        ...
```

Dataset metadata should include at least `name`, `target_column`, `raw_feature_type`, `raw_feature_mutability`, and `raw_feature_actionability`. The initialized dataframe should be raw and include the target column. Later on you can get df with `get()` and flag/metadata with `attr()`, and clone it with `clone()` (the clone will be automatically unfreezed so you can do more modifications).

### PreProcessObject

Implement `preprocess/<name>.py` when the behavior is not covered by common preprocessing.

```python
from preprocess.preprocess_object import PreProcessObject
from utils.registry import register


@register("custom")
class CustomPreProcess(PreProcessObject):
    def __init__(self, seed: int | None = None, **kwargs):
        self._seed = seed
        # Save more setting args and do whatever you want
        ...

    def transform(self, input: DatasetObject) -> DatasetObject:
        df = input.snapshot()
        ...
        input.update("preprocess_flag", True, df=df)
        input.update("metadata_name", metadata_content)
        return input
```

When modifying _rawdf, only use DatasetObject's `snapshot()` to get a copy of df and `update()` to save the change! Use `update()` to set a unique flag to prevent accidental double application, or just to save some metadata.

### ModelObject

Implement `model/<name>/<name>.py`. You can put auxiliary files under `model/<name>/`.

```python
from model.model_object import ModelObject, process_nan
from utils.registry import register


@register("custom")
class CustomModel(ModelObject):
    def __init__(self, seed: int | None = None, device: str = "cpu", **kwargs):
        self._seed = seed
        self._device = device
        self._need_grad = True  # Or False
        self._is_trained = False
        self._model = ...
        # Save more setting args and do whatever you want
        ...

    def fit(self, train_set: DatasetObject):
        ...
        self._is_trained = True

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        ...

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        ...  # Or raise RuntimeError("This model is not differentiable")
```

By default, inferencing interfaces (with DatasetObject) such as `predict()` and `predict_proba()` are inherited batching wrappers, but you can still overwrite them for better control of the metadata.

### MethodObject

Implement `method/<name>/<name>.py`. You can put auxiliary files under `method/<name>/`.

```python
from method.method_object import MethodObject
from utils.registry import register


@register("custom")
class CustomMethod(MethodObject):
    def __init__(self, target_model: ModelObject, seed: int | None = None, device: str = "cpu", desired_class: int | str | None = None, **kwargs):
        self._target_model = target_model
        self._seed = seed
        self._device = device
        self._desired_class = desired_class
        self._need_grad = True  # Or False
        self._is_trained = False  # Or True if it does not need training at all
        # Please check everything, including model type compatibility, model device compatibility and more...
        # Save more setting args and do whatever you want
        ...

    def fit(self, train_set: DatasetObject):
        ...
        self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        ...
```

`get_counterfactuals()` receives a feature dataframe and must return a dataframe with the same rows and feature columns. Rows with no valid counterfactual should be filled with `NaN`. By default, inferencing interface (with DatasetObject) `predict()` wraps this output into a frozen counterfactual `DatasetObject`, but you can still overwrite it for better control of the metadata.

### EvaluationObject

Implement `evaluation/<name>.py`.

```python
from evaluation.evaluation_object import EvaluationObject
from utils.registry import register


@register("custom")
class CustomEvaluation(EvaluationObject):
    def __init__(self, **kwargs):
        # Save more setting args and do whatever you want
        ...

    def evaluate(self, factuals: DatasetObject, counterfactuals: DatasetObject) -> pd.DataFrame:
        ...
```

Return a one-row dataframe with stable, descriptive column names. If the counterfactual dataset has `evaluation_filter` (to filter out ignored rows), apply it before performing evaluations.

## Running Experiments

After installing the package, the generic CLI entry point is `recourse-bench`:

```bash
recourse-bench -p experiment/toy/smoke_config.yaml
```

From a source checkout, `experiments.py` is equivalent:

```bash
python experiments.py -p experiment/toy/smoke_config.yaml
```

`main.py` is equivalent:

```bash
python main.py -p experiment/toy/toydata_linear_toy.yaml
```

A minimal config has this structure:

```yaml
name: credit_linear_wachter_smoke
logger:
  level: INFO
  path: ./logs/credit_linear_wachter_smoke.log
caching:
  path: ./cache/
dataset:
  name: credit
preprocess:
  - name: balance
    seed: 7
    strategy: downsample
  - name: encode
    seed: 7
    encoding: onehot
  - name: scale
    seed: 7
    scaling: normalize
  - name: split
    seed: 7
    split: 0.3
model:
  name: linear
  seed: 7
  device: cpu
method:
  name: wachter
  seed: 7
  device: cpu
  desired_class: 1
evaluation:
  - name: validity
  - name: distance
```

`Experiment.run()` performs preprocessing, resolves train/test datasets, trains the target model, fits the recourse method, generates counterfactuals, and concatenates evaluation outputs into one metrics dataframe.

Python API usage:

```python
from pathlib import Path

import yaml

from experiments import Experiment

config = yaml.safe_load(Path("experiment/wachter/credit_linear_wachter_smoke.yaml").read_text())
metrics = Experiment(config).run()
print(metrics.to_string(index=False))
```

Or you can arrange components like LEGO blocks.

## Reproduction and Smoke Scripts

Method-specific scripts live in `experiment/<method>/`. Smoke scripts are short functionality checks; reproduction scripts usually follow paper-specific protocols and may run for much longer.

Some reproduction scripts implement additional paper logic and should be invoked directly.

| Method | Entry point | Notes |
| --- | --- | --- |
| `apas` | `python experiment/apas/reproduce.py` | Uses `--config`, default `experiment/apas/config.yaml`. |
| `arg_ensembling` | `python experiment/arg_ensembling/reproduce.py` | Uses `--config`, default `experiment/arg_ensembling/config.yaml`. |
| `cchvae` | `python experiment/cchvae/reproduce.py` | Uses `-p/--path`, default credit CCHVAE YAML. |
| `cemsp` | `python experiment/cemsp/reproduce.py` | Optional `--max-factuals`. |
| `cfrl` | `python experiment/cfrl/reproduce.py` | Builds its reproduction config in the script. |
| `cfvae` | `python experiment/cfvae/reproduce.py --weights-dir <dir>` | Requires CFVAE reproduction weights. |
| `claproar` | `python experiment/claproar/smoke.py` | Uses `-p/--path`. |
| `clue` | `python experiment/clue/reproduce.py --bnn-art-path <pth> --vae-art-path <pth> --vaeac-art-path <pth> --vaeac-gt-path <pth> --under-vaeac-gt-path <pth>` | Requires pretrained CLUE artifacts. |
| `cogs` | `python experiment/cogs/reproduce.py` | Optional `--max-factuals`. |
| `cols` | `python experiment/cols/reproduce.py` | Optional `--max-runs`, `--max-factuals`, and method/profile overrides. |
| `cruds` | `python experiment/cruds/smoke.py` | Uses `-p/--path`. |
| `cvas_proj` | `python experiment/cvas_proj/reproduce.py` | Uses current/future configs; `--smoke` is available. |
| `dice` | `python experiment/dice/reproduce.py` | Optional `--assert-paper`, `--autorepro`, `--num-factuals`. |
| `diverse_dist` | `python experiment/diverse_dist/reproduce.py` | Uses `--config`, default `experiment/diverse_dist/config.yaml`. |
| `face` | `python experiment/face/reproduce.py` | Runs built-in FACE synthetic-data reproduction variants. |
| `feature_tweak` | `python experiment/feature_tweak/smoke.py` | Uses `-p/--path`. |
| `gravitational` | `python experiment/gravitational/smoke.py` | Uses `-p/--path`. |
| `gs` | `python experiment/gs/reproduce.py` | Supports `--mode smoke` and `--factual-limit`. |
| `larr` | `python experiment/larr/reproduce.py` | Runs built-in LARR reproduction. |
| `mace` | `python experiment/mace/reproduce.py` | Supports `--dataset`, `--norm`, `--num-factuals`, and `--strict`. |
| `probe` | `python experiment/probe/reproduce.py` | Uses `-p/--path`. |
| `proplace` | `python experiment/proplace/reproduce.py` | Optional `--max-factuals` and `--row-limit`. |
| `rbr` | `python experiment/rbr/reproduce.py` | Uses current/future German configs. |
| `revise` | `python experiment/revise/smoke.py` | Uses `-p/--path`. |
| `roar` | `python experiment/roar/reproduce.py` | Uses current/future German configs. |
| `sns` | `python experiment/sns/reproduce.py` | Optional `--max-factuals`, `--max-related-models`. |
| `trex` | `python experiment/trex/reproduce.py` | Supports device, split, tau, and factual/model-count overrides. |
| `wachter` | `python experiment/wachter/smoke.py` | Uses `-p/--path`. |

Long reproduction jobs write logs under `logs/` and cache intermediate artifacts under `cache/`, according to their configs.

## Development Conventions

- Add narrowly scoped components; avoid modifying shared abstractions unless the current design cannot express.
- Keep datasets offline available and raw at initialization.
- Put method-specific reproduction scripts under `experiment/<method>/`.
- Register every component with `@register("<name>")` and import it in the relevant `__init__.py`.
- Use `utils.seed.seed_context` around random operations.
- Use logging and `tqdm` for long training or search loops.
- Prefer existing preprocessing, target model, and evaluation objects before adding new ones.
- Format finalized code with `isort` and `black`.
