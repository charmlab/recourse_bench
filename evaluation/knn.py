from __future__ import annotations

import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import (
    resolve_evaluation_inputs,
    resolve_ref_df,
    to_float_tensor,
)
from utils.registry import register


@register("knn")
class KnnEvaluation(EvaluationObject):
    def __init__(self, refset: DatasetObject, k: int = 5, **kwargs):
        if refset is None:
            raise ValueError("refset must not be None")
        if int(k) < 1:
            raise ValueError("k must be >= 1")

        self._refset = refset
        self._k = int(k)

    def set_refset(self, refset: DatasetObject) -> None:
        if refset is None:
            raise ValueError("refset must not be None")
        self._refset = refset

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        (
            _,
            counterfactual_features,
            evaluation_mask,
            success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)

        selected_mask = evaluation_mask & success_mask
        result_column = f"knn_{self._k}"
        if selected_mask.sum() == 0:
            return pd.DataFrame([{result_column: float("nan")}])

        ref_df = resolve_ref_df(self._refset)
        ref_features = ref_df.loc[:, ref_df.columns != self._refset.target_column]
        if ref_features.shape[0] == 0:
            raise ValueError("refset must contain at least one row")
        if list(ref_features.columns) != list(counterfactual_features.columns):
            raise ValueError(
                "refset features must match counterfactual feature columns"
            )

        counterfactual_success = counterfactual_features.loc[selected_mask.to_numpy()]
        ref_tensor = to_float_tensor(ref_features)
        counterfactual_tensor = to_float_tensor(counterfactual_success)

        neighbor_count = min(self._k, ref_tensor.shape[0])
        distances = torch.cdist(counterfactual_tensor, ref_tensor, p=2)
        nearest = torch.topk(distances, k=neighbor_count, largest=False, dim=1).values
        value = nearest.mean(dim=1).mean().item()
        return pd.DataFrame([{result_column: float(value)}])
