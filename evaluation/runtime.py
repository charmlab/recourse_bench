from __future__ import annotations

import pandas as pd

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from utils.registry import register


@register("runtime")
class RuntimeEvaluation(EvaluationObject):
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def _resolve_runtime_series(
        counterfactuals: DatasetObject, expected_index: pd.Index
    ) -> pd.Series:
        raw_runtime = counterfactuals.attr("runtime_seconds")
        if isinstance(raw_runtime, pd.Series):
            runtime_series = raw_runtime.astype("float64")
        elif isinstance(raw_runtime, pd.DataFrame):
            if raw_runtime.shape[1] == 0:
                raise ValueError("runtime_seconds must contain at least one column")
            runtime_series = raw_runtime.iloc[:, 0].astype("float64")
        else:
            raise TypeError("runtime_seconds must be a pandas Series or DataFrame")

        if runtime_series.shape[0] != expected_index.shape[0]:
            raise ValueError("runtime_seconds length must match factual row count")
        return runtime_series.loc[expected_index]

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        (
            factual_features,
            _counterfactual_features,
            evaluation_mask,
            _success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)

        try:
            runtime_series = self._resolve_runtime_series(
                counterfactuals, factual_features.index
            )
        except AttributeError:
            return pd.DataFrame([{"runtime_seconds": float("nan")}])

        selected_runtime = runtime_series.loc[evaluation_mask.to_numpy()]
        if selected_runtime.shape[0] == 0:
            runtime_seconds = float("nan")
        else:
            runtime_seconds = float(selected_runtime.mean())

        return pd.DataFrame([{"runtime_seconds": runtime_seconds}])
