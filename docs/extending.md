# Extending the framework

The framework has five extensible component types. A new class subclasses the
relevant base, is decorated with {func}`~utils.registry.register`, and is
imported from its package `__init__.py` so the registry discovers it. The same
registered name may be reused across different component types.

See the {doc}`reference/index` for the full contract of each base class.

## Dataset

Implement `dataset/<name>/<name>.py` and `dataset/<name>/<name>.yaml`. The
metadata should include at least `name`, `target_column`, `raw_feature_type`,
`raw_feature_mutability`, and `raw_feature_actionability`. The initialized
dataframe must be raw (non-encoded, non-scaled) and include the target column.

```python
from recourse_bench.dataset.dataset_object import DatasetObject
from recourse_bench.utils.registry import register


@register("custom")
class CustomDataset(DatasetObject):
    def __init__(self, path: str = "./dataset/custom/", **kwargs):
        self._rawdf = self._read_df(path)
        for key, value in self._read_attrs(path).items():
            setattr(self, key, value)

    def _read_df(self, path: str):
        ...
```

## Preprocess

When modifying the dataframe, only use
{meth}`~dataset.dataset_object.DatasetObject.snapshot` to get a copy and
{meth}`~dataset.dataset_object.DatasetObject.update` to write it back (also set
a unique flag to guard against double application).

```python
from recourse_bench.preprocess.preprocess_object import PreProcessObject
from recourse_bench.utils.registry import register


@register("custom")
class CustomPreProcess(PreProcessObject):
    def __init__(self, seed: int | None = None, **kwargs):
        self._seed = seed

    def transform(self, input):
        df = input.snapshot()
        ...
        input.update("preprocess_flag", True, df=df)
        return input
```

## Target model

```python
from recourse_bench.model.model_object import ModelObject, process_nan
from recourse_bench.utils.registry import register


@register("custom")
class CustomModel(ModelObject):
    def __init__(self, seed=None, device="cpu", **kwargs):
        self._seed, self._device = seed, device
        self._need_grad = True
        self._is_trained = False

    def fit(self, trainset):
        ...
        self._is_trained = True

    @process_nan()
    def get_prediction(self, X, proba=True):
        ...

    def forward(self, X):
        ...  # or raise RuntimeError for non-differentiable models
```

## Method

`get_counterfactuals()` receives a feature dataframe and must return a dataframe
with the same rows and feature columns; rows with no valid counterfactual are
filled with `NaN`.

```python
from recourse_bench.method.method_object import MethodObject
from recourse_bench.utils.registry import register


@register("custom")
class CustomMethod(MethodObject):
    def __init__(self, target_model, seed=None, device="cpu", desired_class=None, **kwargs):
        self._target_model = target_model
        self._seed, self._device, self._desired_class = seed, device, desired_class
        self._need_grad = True
        self._is_trained = False

    def fit(self, trainset):
        ...
        self._is_trained = True

    def get_counterfactuals(self, factuals):
        ...
```

## Evaluation

Return a one-row dataframe with stable, descriptive column names. If the
counterfactual dataset has an `evaluation_filter`, apply it first.

```python
from recourse_bench.evaluation.evaluation_object import EvaluationObject
from recourse_bench.utils.registry import register


@register("custom")
class CustomEvaluation(EvaluationObject):
    def __init__(self, **kwargs):
        ...

    def evaluate(self, factuals, counterfactuals):
        ...
```
