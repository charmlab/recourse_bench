from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch

from model.model_object import ModelObject


def _to_feature_dataframe(
    values: pd.DataFrame | np.ndarray | torch.Tensor,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.loc[:, list(feature_names)].copy(deep=True)

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    if array.ndim == 1:
        array = array.reshape(1, -1)
    return pd.DataFrame(array, columns=list(feature_names))


def _predict_label_indices(
    target_model: ModelObject,
    feature_names: Sequence[str],
    X: pd.DataFrame | np.ndarray | torch.Tensor,
) -> np.ndarray:
    features = _to_feature_dataframe(X, feature_names)
    probabilities = target_model.get_prediction(features, proba=True)
    if isinstance(probabilities, torch.Tensor):
        return probabilities.detach().cpu().numpy().argmax(axis=1)
    return np.asarray(probabilities).argmax(axis=1)


def _resolve_target_indices(
    target_model: ModelObject,
    original_prediction: np.ndarray,
    desired_class: int | str | None,
) -> np.ndarray:
    class_to_index = target_model.get_class_to_index()
    if desired_class is not None:
        if desired_class not in class_to_index:
            raise ValueError("desired_class is invalid for the trained target model")
        return np.full(
            shape=original_prediction.shape,
            fill_value=int(class_to_index[desired_class]),
            dtype=np.int64,
        )

    if len(class_to_index) != 2:
        raise ValueError(
            "desired_class=None is supported for binary classification only"
        )
    return 1 - original_prediction.astype(np.int64, copy=False)


def validate_counterfactuals(
    target_model: ModelObject,
    factuals: pd.DataFrame,
    candidates: pd.DataFrame,
    desired_class: int | str | None = None,
) -> pd.DataFrame:
    if list(candidates.columns) != list(factuals.columns):
        candidates = candidates.reindex(columns=factuals.columns)
    candidates = candidates.copy(deep=True)

    if candidates.shape[0] != factuals.shape[0]:
        raise ValueError("Candidates must preserve the number of factual rows")

    valid_rows = ~candidates.isna().any(axis=1)
    if not bool(valid_rows.any()):
        return candidates

    original_prediction = _predict_label_indices(
        target_model=target_model,
        feature_names=factuals.columns,
        X=factuals,
    )
    target_prediction = _resolve_target_indices(
        target_model=target_model,
        original_prediction=original_prediction,
        desired_class=desired_class,
    )

    candidate_prediction = _predict_label_indices(
        target_model=target_model,
        feature_names=factuals.columns,
        X=candidates.loc[valid_rows],
    )
    success_mask = pd.Series(False, index=candidates.index, dtype=bool)
    success_mask.loc[valid_rows] = (
        candidate_prediction.astype(np.int64, copy=False)
        == target_prediction[valid_rows.to_numpy()]
    )
    candidates.loc[~success_mask, :] = np.nan
    return candidates
