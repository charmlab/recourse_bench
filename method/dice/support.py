from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from method.probe.utils import (
    CategoricalGroup,
    build_thermometer_patterns,
    infer_binary_feature_indices,
    infer_categorical_groups,
)
from model.model_object import ModelObject
from preprocess.preprocess_utils import dataset_has_attr, resolve_feature_metadata


def ensure_model_supports_gradients(
    target_model: ModelObject,
    method_name: str,
) -> None:
    if bool(getattr(target_model, "_need_grad", False)):
        return
    raise TypeError(f"{method_name} requires target_model._need_grad == True")


def ensure_binary_classifier(target_model: ModelObject, method_name: str) -> None:
    class_to_index = target_model.get_class_to_index()
    if len(class_to_index) != 2:
        raise ValueError(f"{method_name} supports binary classification only")


def to_feature_dataframe(
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


@dataclass
class FeatureGroups:
    feature_names: list[str]
    continuous: list[str]
    categorical: list[str]
    binary: list[str]
    immutable: list[str]
    mutable: list[str]
    mutable_mask: np.ndarray


def resolve_feature_groups(dataset: DatasetObject) -> FeatureGroups:
    feature_df = dataset.get(target=False)
    feature_names = list(feature_df.columns)
    feature_type, feature_mutability, feature_actionability = resolve_feature_metadata(
        dataset
    )

    continuous: list[str] = []
    categorical: list[str] = []
    binary: list[str] = []
    immutable: list[str] = []
    mutable: list[str] = []

    for feature_name in feature_names:
        feature_kind = str(feature_type[feature_name]).lower()
        is_mutable = bool(feature_mutability[feature_name])
        actionability = str(feature_actionability[feature_name]).lower()

        if feature_kind == "numerical":
            continuous.append(feature_name)
        else:
            categorical.append(feature_name)
            if feature_kind == "binary":
                binary.append(feature_name)

        if (not is_mutable) or actionability in {"none", "same"}:
            immutable.append(feature_name)
        else:
            mutable.append(feature_name)

    mutable_mask = np.array(
        [feature_name in set(mutable) for feature_name in feature_names], dtype=bool
    )
    return FeatureGroups(
        feature_names=feature_names,
        continuous=continuous,
        categorical=categorical,
        binary=binary,
        immutable=immutable,
        mutable=mutable,
        mutable_mask=mutable_mask,
    )


class RecourseModelAdapter:
    def __init__(self, target_model: ModelObject, feature_names: Sequence[str]):
        self._target_model = target_model
        self._feature_names = list(feature_names)
        class_to_index = target_model.get_class_to_index()
        self._index_to_class = {
            index: class_value for class_value, index in class_to_index.items()
        }
        self.classes_ = np.array(
            [self._index_to_class[index] for index in sorted(self._index_to_class)]
        )

    def get_ordered_features(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> pd.DataFrame:
        return to_feature_dataframe(X, self._feature_names)

    def predict_proba(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if isinstance(X, torch.Tensor):
            return differentiable_predict_proba(self._target_model, X)

        features = self.get_ordered_features(X)
        prediction = self._target_model.get_prediction(features, proba=True)
        if isinstance(prediction, torch.Tensor):
            return prediction.detach().cpu().numpy()
        return np.asarray(prediction)

    def predict(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if isinstance(X, torch.Tensor):
            probabilities = differentiable_predict_proba(self._target_model, X)
            return probabilities.argmax(dim=1)

        features = self.get_ordered_features(X)
        prediction = self._target_model.get_prediction(features, proba=False)
        if isinstance(prediction, torch.Tensor):
            label_indices = prediction.detach().cpu().numpy().argmax(axis=1)
        else:
            label_indices = np.asarray(prediction).argmax(axis=1)
        return np.asarray([self._index_to_class[int(index)] for index in label_indices])

    def predict_label_indices(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        if isinstance(X, torch.Tensor):
            probabilities = differentiable_predict_proba(self._target_model, X)
            return probabilities.argmax(dim=1).detach().cpu().numpy()

        features = self.get_ordered_features(X)
        probabilities = self._target_model.get_prediction(features, proba=True)
        if isinstance(probabilities, torch.Tensor):
            return probabilities.detach().cpu().numpy().argmax(axis=1)
        return np.asarray(probabilities).argmax(axis=1)


def differentiable_predict_proba(
    target_model: ModelObject,
    X: torch.Tensor,
) -> torch.Tensor:
    ensure_model_supports_gradients(target_model, "differentiable_predict_proba")
    logits = target_model(X.to(target_model._device))
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Target model forward pass must return a torch.Tensor")

    output_activation = getattr(target_model, "_output_activation_name", "softmax")
    output_activation = str(output_activation).lower()
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    if output_activation == "sigmoid":
        positive_probability = torch.sigmoid(logits)
        return torch.cat([1.0 - positive_probability, positive_probability], dim=1)
    return torch.softmax(logits, dim=1)


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

    adapter = RecourseModelAdapter(target_model, factuals.columns)
    original_prediction = adapter.predict_label_indices(factuals)
    target_prediction = resolve_target_indices(
        target_model=target_model,
        original_prediction=original_prediction,
        desired_class=desired_class,
    )

    candidate_prediction = adapter.predict_label_indices(candidates.loc[valid_rows])
    success_mask = pd.Series(False, index=candidates.index, dtype=bool)
    success_mask.loc[valid_rows] = (
        candidate_prediction.astype(np.int64, copy=False)
        == target_prediction[valid_rows.to_numpy()]
    )
    candidates.loc[~success_mask, :] = np.nan
    return candidates


def get_encoding_map(dataset: DatasetObject) -> dict[str, list[str]] | None:
    if not dataset_has_attr(dataset, "encoding"):
        return None
    raw_encoding = dataset.attr("encoding")
    return {
        str(feature_name): [str(column) for column in columns]
        for feature_name, columns in raw_encoding.items()
    }


def resolve_categorical_groups(
    dataset: DatasetObject,
    feature_names: Sequence[str],
) -> list[CategoricalGroup]:
    encoding_map = get_encoding_map(dataset)
    return infer_categorical_groups(feature_names, encoding_map)


def resolve_binary_feature_value_map(
    train_features: pd.DataFrame,
    feature_names: Sequence[str],
    feature_type: Mapping[str, str],
    categorical_groups: Sequence[CategoricalGroup],
) -> dict[int, np.ndarray]:
    binary_indices = infer_binary_feature_indices(
        feature_names,
        feature_type,
        categorical_groups,
    )
    value_map: dict[int, np.ndarray] = {}
    for feature_index in binary_indices:
        feature_name = feature_names[feature_index]
        unique_values = sorted(pd.Index(train_features[feature_name].unique()).tolist())
        if len(unique_values) > 2:
            raise ValueError(
                f"Binary feature {feature_name} has more than two observed values"
            )
        value_map[feature_index] = np.asarray(unique_values, dtype=np.float32)
    return value_map


def compute_feature_bounds(
    train_features: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    lower_bounds = (
        train_features.loc[:, list(feature_names)]
        .min(axis=0)
        .to_numpy(dtype=np.float32)
    )
    upper_bounds = (
        train_features.loc[:, list(feature_names)]
        .max(axis=0)
        .to_numpy(dtype=np.float32)
    )
    return lower_bounds, upper_bounds


def compute_feature_weights(
    train_features: pd.DataFrame,
    feature_names: Sequence[str],
    continuous_feature_names: Sequence[str],
    feature_weights: str | Mapping[str, float],
) -> np.ndarray:
    if isinstance(feature_weights, Mapping):
        return np.asarray(
            [
                float(feature_weights.get(feature_name, 1.0))
                for feature_name in feature_names
            ],
            dtype=np.float32,
        )

    feature_weights_name = str(feature_weights).lower()
    if feature_weights_name != "inverse_mad":
        raise ValueError(
            "DiceMethod supports feature_weights='inverse_mad' or dict[str, float] only"
        )

    continuous_set = set(continuous_feature_names)
    resolved_weights: list[float] = []
    for feature_name in feature_names:
        if feature_name not in continuous_set:
            resolved_weights.append(1.0)
            continue
        series = train_features[feature_name].astype("float64")
        median = float(series.median())
        mad = float(np.median(np.abs(series.to_numpy() - median)))
        if mad <= 0.0:
            resolved_weights.append(1.0)
        else:
            resolved_weights.append(round(1.0 / mad, 2))
    return np.asarray(resolved_weights, dtype=np.float32)


def compute_sparsity_thresholds(
    train_features: pd.DataFrame,
    continuous_feature_names: Sequence[str],
    quantile: float,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for feature_name in continuous_feature_names:
        series = train_features[feature_name].astype("float64")
        median = float(series.median())
        deviations = np.abs(series.to_numpy() - median)
        mad = float(np.median(deviations))
        non_zero_deviations = deviations[deviations > 0.0]
        if non_zero_deviations.size == 0:
            threshold = mad
        else:
            quantile_value = float(np.quantile(non_zero_deviations, quantile))
            if mad <= 0.0:
                threshold = quantile_value
            else:
                threshold = min(mad, quantile_value)
        thresholds[feature_name] = float(max(threshold, 0.0))
    return thresholds


def project_binary_features(
    instance: torch.Tensor,
    binary_feature_value_map: Mapping[int, np.ndarray],
) -> torch.Tensor:
    if not binary_feature_value_map:
        return instance

    squeeze_output = False
    if instance.ndim == 1:
        instance = instance.unsqueeze(0)
        squeeze_output = True

    projected = instance.clone()
    for feature_index, raw_values in binary_feature_value_map.items():
        allowed_values = torch.as_tensor(
            raw_values,
            dtype=projected.dtype,
            device=projected.device,
        )
        distances = torch.abs(
            projected[:, feature_index].unsqueeze(1) - allowed_values.unsqueeze(0)
        )
        nearest_indices = distances.argmin(dim=1)
        projected[:, feature_index] = allowed_values.index_select(0, nearest_indices)

    if squeeze_output:
        return projected.squeeze(0)
    return projected


def project_categorical_groups(
    instance: torch.Tensor,
    categorical_groups: Sequence[CategoricalGroup],
    tie_random: bool = False,
) -> torch.Tensor:
    if not categorical_groups:
        return instance

    squeeze_output = False
    if instance.ndim == 1:
        instance = instance.unsqueeze(0)
        squeeze_output = True

    projected = instance.clone()
    for group in categorical_groups:
        group_indices = list(group.indices)
        group_values = projected[:, group_indices]

        if group.encoding == "thermometer":
            patterns = build_thermometer_patterns(
                len(group_indices),
                device=projected.device,
                dtype=projected.dtype,
            )
            squared_distance = torch.sum(
                (group_values.unsqueeze(1) - patterns.unsqueeze(0)) ** 2,
                dim=2,
            )
            best_pattern = squared_distance.argmin(dim=1)
            projected[:, group_indices] = patterns.index_select(0, best_pattern)
            continue

        winners = []
        for row_index in range(group_values.shape[0]):
            row = group_values[row_index]
            max_value = torch.max(row)
            winner_indices = torch.nonzero(
                torch.isclose(row, max_value),
                as_tuple=False,
            ).squeeze(1)
            if winner_indices.numel() == 1 or not tie_random:
                winners.append(int(winner_indices[0].item()))
            else:
                random_choice = torch.randint(
                    low=0,
                    high=winner_indices.numel(),
                    size=(1,),
                    device=projected.device,
                ).item()
                winners.append(int(winner_indices[random_choice].item()))

        group_projection = torch.zeros_like(group_values)
        row_indices = torch.arange(group_values.shape[0], device=projected.device)
        group_projection[
            row_indices, torch.tensor(winners, device=projected.device)
        ] = 1.0
        projected[:, group_indices] = group_projection

    if squeeze_output:
        return projected.squeeze(0)
    return projected


def project_discrete_feature_space(
    instance: torch.Tensor,
    categorical_groups: Sequence[CategoricalGroup],
    binary_feature_value_map: Mapping[int, np.ndarray],
    tie_random: bool = False,
) -> torch.Tensor:
    projected = project_categorical_groups(
        instance,
        categorical_groups=categorical_groups,
        tie_random=tie_random,
    )
    projected = project_binary_features(
        projected,
        binary_feature_value_map=binary_feature_value_map,
    )
    return projected


def deduplicate_counterfactuals(counterfactuals: pd.DataFrame) -> pd.DataFrame:
    if counterfactuals.empty:
        return counterfactuals.copy(deep=True)
    rounded = counterfactuals.copy(deep=True)
    numeric_columns = rounded.select_dtypes(include=["number"]).columns
    rounded.loc[:, numeric_columns] = rounded.loc[:, numeric_columns].round(6)
    rounded = rounded.drop_duplicates(ignore_index=True)
    return rounded
