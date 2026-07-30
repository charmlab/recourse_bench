from __future__ import annotations

import warnings

import pandas as pd
import torch

from recourse_bench.dataset.dataset_object import DatasetObject
from recourse_bench.evaluation.evaluation_object import EvaluationObject
from recourse_bench.evaluation.evaluation_utils import (
    distance,
    resolve_evaluation_inputs,
    resolve_ref_df,
    resolve_restore_mode,
    restore_features,
    to_float_tensor,
)
from recourse_bench.utils.registry import register


@register("knn")
class KnnEvaluation(EvaluationObject):
    def __init__(
        self,
        ref_set: DatasetObject | None = None,
        k: int = 5,
        restore_categorical: bool | str = False,
        restore_numerical: bool | str = False,
        **kwargs,
    ):
        if ref_set is None and "refset" in kwargs:
            warnings.warn(
                "The 'refset' argument was renamed to 'ref_set'; the old name "
                "still works but will be removed.",
                DeprecationWarning,
                stacklevel=2,
            )
            ref_set = kwargs.pop("refset")
        if ref_set is None:
            raise ValueError("ref_set must not be None")
        if int(k) < 1:
            raise ValueError("k must be >= 1")

        self._ref_set = ref_set
        self._k = int(k)
        (
            self._binarize_categorical,
            self._restore_mode,
        ) = resolve_restore_mode(restore_categorical, restore_numerical)

    def set_ref_set(self, ref_set: DatasetObject) -> None:
        if ref_set is None:
            raise ValueError("ref_set must not be None")
        self._ref_set = ref_set

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

        ref_df = resolve_ref_df(self._ref_set)
        ref_features = ref_df.loc[:, ref_df.columns != self._ref_set.target_column]
        if ref_features.shape[0] == 0:
            raise ValueError("ref_set must contain at least one row")
        if list(ref_features.columns) != list(counterfactual_features.columns):
            raise ValueError(
                "ref_set features must match counterfactual feature columns"
            )

        counterfactual_success = counterfactual_features.loc[selected_mask.to_numpy()]
        if self._restore_mode != "none":
            counterfactuals_clone = counterfactuals.clone()
            counterfactuals_clone._rawdf = pd.concat(
                [counterfactual_features, counterfactuals.get(target=True)], axis=1
            )
            counterfactuals_clone.freeze()
            ref_features, counterfactual_features = restore_features(
                self._ref_set, counterfactuals_clone, mode=self._restore_mode
            )
            counterfactual_success = counterfactual_features.loc[
                selected_mask.to_numpy()
            ]

        if self._binarize_categorical:
            binarize_list = [
                column
                for column, feature_type in ref_features.attrs.get(
                    "raw_feature_type", {}
                ).items()
                if str(feature_type).lower() == "categorical"
            ]
            distances = torch.stack(
                [
                    distance(
                        ref_features,
                        pd.DataFrame(
                            [row.to_dict()] * ref_features.shape[0],
                            index=ref_features.index,
                            columns=ref_features.columns,
                        ),
                        "l2",
                        binarize_list=binarize_list,
                    )
                    for _, row in counterfactual_success.iterrows()
                ]
            )
        else:
            ref_tensor = to_float_tensor(ref_features)
            counterfactual_tensor = to_float_tensor(counterfactual_success)
            distances = torch.cdist(counterfactual_tensor, ref_tensor, p=2)

        neighbor_count = min(self._k, ref_features.shape[0])
        nearest = torch.topk(distances, k=neighbor_count, largest=False, dim=1).values
        value = nearest.mean(dim=1).mean().item()
        return pd.DataFrame([{result_column: float(value)}])
