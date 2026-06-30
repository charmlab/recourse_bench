from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from recourse_bench.dataset.dataset_object import DatasetObject
from recourse_bench.evaluation.evaluation_object import EvaluationObject
from recourse_bench.evaluation.evaluation_utils import (
    distance,
    resolve_evaluation_inputs,
    resolve_restore_mode,
    restore_features,
    to_float_tensor,
)
from recourse_bench.utils.registry import register


@register("ynn")
class YnnEvaluation(EvaluationObject):
    def __init__(
        self,
        k: int = 5,
        restore_categorical: bool | str = False,
        restore_numerical: bool | str = False,
        **kwargs,
    ):
        self._k = int(k)
        if self._k < 1:
            raise ValueError("k must be >= 1")
        (
            self._binarize_categorical,
            self._restore_mode,
        ) = resolve_restore_mode(restore_categorical, restore_numerical)

    @staticmethod
    def _resolve_index_series(
        counterfactuals: DatasetObject,
        flag: str,
        expected_index: pd.Index,
    ) -> pd.Series:
        raw_values = counterfactuals.attr(flag)
        if isinstance(raw_values, pd.Series):
            series = raw_values.astype("int64")
        elif isinstance(raw_values, pd.DataFrame):
            if raw_values.shape[1] == 0:
                raise ValueError(f"{flag} must contain at least one column")
            series = raw_values.iloc[:, 0].astype("int64")
        else:
            raise TypeError(f"{flag} must be a pandas Series or DataFrame")

        if series.shape[0] != expected_index.shape[0]:
            raise ValueError(f"{flag} length must match factual row count")
        return series.loc[expected_index]

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
        if selected_mask.sum() == 0:
            return pd.DataFrame([{"ynn": float("nan")}])

        try:
            factual_prediction_index = self._resolve_index_series(
                counterfactuals,
                "factual_prediction_index",
                factual_features.index,
            )
            target_prediction_index = self._resolve_index_series(
                counterfactuals,
                "target_prediction_index",
                factual_features.index,
            )
        except AttributeError:
            return pd.DataFrame([{"ynn": float("nan")}])

        selected_counterfactuals = counterfactual_features.loc[selected_mask.to_numpy()]
        selected_targets = target_prediction_index.loc[selected_mask.to_numpy()]

        if self._restore_mode != "none":
            counterfactuals_clone = counterfactuals.clone()
            counterfactuals_clone._rawdf = pd.concat(
                [counterfactual_features, counterfactuals.get(target=True)], axis=1
            )
            counterfactuals_clone.freeze()
            factual_features, counterfactual_features = restore_features(
                factuals, counterfactuals_clone, mode=self._restore_mode
            )
            selected_counterfactuals = counterfactual_features.loc[
                selected_mask.to_numpy()
            ]

        if self._binarize_categorical:
            binarize_list = [
                column
                for column, feature_type in factual_features.attrs.get(
                    "raw_feature_type", {}
                ).items()
                if str(feature_type).lower() == "categorical"
            ]
            distances = torch.stack(
                [
                    distance(
                        factual_features,
                        pd.DataFrame(
                            [row.to_dict()] * factual_features.shape[0],
                            index=factual_features.index,
                            columns=factual_features.columns,
                        ),
                        "l2",
                        binarize_list=binarize_list,
                    )
                    for _, row in selected_counterfactuals.iterrows()
                ]
            )
            factual_count = factual_features.shape[0]
        else:
            factual_tensor = to_float_tensor(factual_features)
            selected_counterfactual_tensor = to_float_tensor(selected_counterfactuals)
            distances = torch.cdist(selected_counterfactual_tensor, factual_tensor, p=2)
            factual_count = factual_tensor.shape[0]
        k = min(self._k, factual_count)
        if k < 1:
            return pd.DataFrame([{"ynn": float("nan")}])

        knn_indices = torch.topk(distances, k=k, largest=False).indices.cpu().numpy()
        factual_pred_values = factual_prediction_index.to_numpy(dtype=np.int64)
        target_values = selected_targets.to_numpy(dtype=np.int64)

        scores = []
        for row_idx, neighbors in enumerate(knn_indices):
            neighbor_predictions = factual_pred_values[neighbors]
            score = float((neighbor_predictions == target_values[row_idx]).mean())
            scores.append(score)

        ynn = float(np.mean(scores)) if scores else float("nan")
        return pd.DataFrame([{"ynn": ynn}])
