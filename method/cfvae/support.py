from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from preprocess.preprocess_utils import dataset_has_attr, resolve_feature_metadata

TorchModelTypes = (LinearModel, MlpModel)


def ensure_supported_target_model(
    target_model: ModelObject,
    supported_types: Sequence[type[ModelObject]],
    method_name: str,
) -> None:
    if isinstance(target_model, tuple(supported_types)):
        return
    if isinstance(target_model, ModelObject) and bool(
        getattr(target_model, "_need_grad", False)
    ):
        return

    supported_names = ", ".join(cls.__name__ for cls in supported_types)
    raise TypeError(
        f"{method_name} supports target models [{supported_names}] only, "
        f"received {target_model.__class__.__name__}"
    )


@dataclass
class FeatureGroups:
    feature_names: list[str]
    continuous: list[str]
    binary_scalar: list[str]
    onehot_groups: list[list[str]]
    binary_value_ranges: dict[str, tuple[float, float]]


def resolve_feature_groups(dataset: DatasetObject) -> FeatureGroups:
    feature_df = dataset.get(target=False)
    feature_names = list(feature_df.columns)
    feature_type, _, _ = resolve_feature_metadata(dataset)

    if any("_therm_" in feature_name for feature_name in feature_names):
        raise ValueError("CfvaeMethod does not support thermometer-encoded features")

    onehot_groups: list[list[str]] = []
    if dataset_has_attr(dataset, "encoding"):
        encoding = dataset.attr("encoding")
        for encoded_columns in encoding.values():
            if not isinstance(encoded_columns, list):
                continue
            group = [column for column in encoded_columns if column in feature_names]
            if len(group) <= 1:
                continue
            if not all("_cat_" in column for column in group):
                raise ValueError(
                    "CfvaeMethod supports onehot-encoded categorical features only"
                )
            onehot_groups.append(group)

    onehot_feature_names = {
        feature_name for group in onehot_groups for feature_name in group
    }
    continuous: list[str] = []
    binary_scalar: list[str] = []
    binary_value_ranges: dict[str, tuple[float, float]] = {}

    for feature_name in feature_names:
        feature_kind = str(feature_type[feature_name]).lower()
        if feature_kind == "numerical":
            series = feature_df[feature_name].astype("float64")
            if float(series.min()) < -1e-6 or float(series.max()) > 1.0 + 1e-6:
                raise ValueError(
                    "CfvaeMethod expects normalized continuous features in [0, 1]"
                )
            continuous.append(feature_name)
            continue

        if feature_kind == "categorical":
            raise ValueError(
                "CfvaeMethod requires fully numeric features; apply encode(onehot) first"
            )
        if feature_kind != "binary":
            raise ValueError(f"Unsupported feature type for CFVAE: {feature_kind}")
        if feature_name in onehot_feature_names:
            continue

        unique_values = sorted(
            pd.Series(feature_df[feature_name])
            .dropna()
            .astype("float64")
            .unique()
            .tolist()
        )
        if len(unique_values) != 2:
            raise ValueError(
                "CfvaeMethod expects scalar binary features to contain exactly two values"
            )

        low_value = float(unique_values[0])
        high_value = float(unique_values[1])
        if low_value == high_value:
            raise ValueError("Scalar binary feature values must not be identical")

        binary_scalar.append(feature_name)
        binary_value_ranges[feature_name] = (low_value, high_value)

    return FeatureGroups(
        feature_names=feature_names,
        continuous=continuous,
        binary_scalar=binary_scalar,
        onehot_groups=onehot_groups,
        binary_value_ranges=binary_value_ranges,
    )


def resolve_continuous_ranges(
    dataset: DatasetObject,
    continuous_features: Sequence[str],
) -> dict[str, float]:
    stored_ranges: dict[str, float] = {}
    if dataset_has_attr(dataset, "range"):
        candidate_ranges = dataset.attr("range")
        if isinstance(candidate_ranges, dict):
            stored_ranges = candidate_ranges

    feature_df = dataset.get(target=False)
    resolved_ranges: dict[str, float] = {}
    for feature_name in continuous_features:
        feature_range = stored_ranges.get(feature_name)
        if feature_range is None:
            series = feature_df[feature_name].astype("float64")
            feature_range = float(series.max() - series.min())
        feature_range = float(feature_range)
        if feature_range <= 0.0:
            feature_range = 1.0
        resolved_ranges[feature_name] = feature_range
    return resolved_ranges


def predict_label_indices(
    target_model: ModelObject,
    X: pd.DataFrame,
) -> np.ndarray:
    probabilities = target_model.get_prediction(X, proba=True)
    if isinstance(probabilities, torch.Tensor):
        return probabilities.detach().cpu().numpy().argmax(axis=1)
    return np.asarray(probabilities).argmax(axis=1)


def resolve_target_indices(
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

    original_prediction = predict_label_indices(target_model, factuals)
    target_prediction = resolve_target_indices(
        target_model=target_model,
        original_prediction=original_prediction,
        desired_class=desired_class,
    )

    candidate_prediction = predict_label_indices(
        target_model, candidates.loc[valid_rows]
    )
    success_mask = pd.Series(False, index=candidates.index, dtype=bool)
    success_mask.loc[valid_rows] = (
        candidate_prediction.astype(np.int64, copy=False)
        == target_prediction[valid_rows.to_numpy()]
    )
    candidates.loc[~success_mask, :] = np.nan
    return candidates
