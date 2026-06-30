from __future__ import annotations

import pandas as pd

from recourse_bench.dataset.dataset_object import DatasetObject
from recourse_bench.evaluation.evaluation_object import EvaluationObject
from recourse_bench.evaluation.evaluation_utils import (
    distance,
    resolve_evaluation_inputs,
    resolve_restore_mode,
    restore_features,
)
from recourse_bench.utils.registry import register


@register("distance")
class DistanceEvaluation(EvaluationObject):
    @staticmethod
    def _resolve_metrics(metrics: list[str] | None) -> list[str]:
        resolved_metrics = [
            metric.lower() for metric in (metrics or ["l0", "l1", "l2", "linf"])
        ]
        invalid = [
            metric
            for metric in resolved_metrics
            if metric not in {"l0", "l1", "l2", "linf"}
        ]
        if invalid:
            raise ValueError(f"Unsupported distance metrics: {invalid}")
        return resolved_metrics

    def __init__(
        self,
        metrics: list[str] | None = None,
        restore_categorical: bool | str = False,
        restore_numerical: bool | str = False,
        **kwargs,
    ):
        self._metrics = self._resolve_metrics(metrics)
        (
            self._binarize_categorical,
            self._restore_mode,
        ) = resolve_restore_mode(restore_categorical, restore_numerical)

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        (
            factual_features,
            counterfactual_features,
            evaluation_mask,
            success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)

        selected_mask = evaluation_mask & success_mask
        results: dict[str, float] = {}
        if selected_mask.sum() == 0:
            for metric in self._metrics:
                results[f"distance_{metric}"] = float("nan")
            return pd.DataFrame([results])

        factual_success = factual_features.loc[selected_mask.to_numpy()]
        counterfactual_success = counterfactual_features.loc[selected_mask.to_numpy()]
        if self._restore_mode != "none":
            factual_features, counterfactual_features = restore_features(
                factuals, counterfactuals, mode=self._restore_mode
            )
            factual_success = factual_features.loc[selected_mask.to_numpy()]
            counterfactual_success = counterfactual_features.loc[
                selected_mask.to_numpy()
            ]

        for metric in self._metrics:
            values = distance(
                factual_success,
                counterfactual_success,
                metric,
                binarize_list=(
                    [
                        column
                        for column, feature_type in factual_success.attrs.get(
                            "raw_feature_type", {}
                        ).items()
                        if str(feature_type).lower() == "categorical"
                    ]
                    if self._binarize_categorical
                    else []
                ),
            )
            results[f"distance_{metric}"] = float(values.mean().item())

        return pd.DataFrame([results])
