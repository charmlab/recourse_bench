from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs, to_float_tensor
from utils.registry import register


@register("ynn")
class YnnEvaluation(EvaluationObject):
    def __init__(self, k: int = 5, **kwargs):
        self._k = int(k)
        if self._k < 1:
            raise ValueError("k must be >= 1")

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

        factual_tensor = to_float_tensor(factual_features)
        selected_counterfactuals = counterfactual_features.loc[selected_mask.to_numpy()]
        selected_counterfactual_tensor = to_float_tensor(selected_counterfactuals)
        selected_targets = target_prediction_index.loc[selected_mask.to_numpy()]

        k = min(self._k, factual_tensor.shape[0])
        if k < 1:
            return pd.DataFrame([{"ynn": float("nan")}])

        distances = torch.cdist(selected_counterfactual_tensor, factual_tensor, p=2)
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
