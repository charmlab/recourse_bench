from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from method.feature_tweak.support import (
    FeatureTweakContext,
    build_feature_tweak_context,
    project_candidate_features,
)
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from preprocess.preprocess_utils import resolve_feature_metadata

TorchReviseModelTypes = (LinearModel, MlpModel)


def ensure_supported_target_model(
    target_model: ModelObject,
    method_name: str,
) -> None:
    if isinstance(target_model, TorchReviseModelTypes):
        return
    raise TypeError(
        f"{method_name} supports target models [LinearModel, MlpModel] only, "
        f"received {target_model.__class__.__name__}"
    )


def ensure_binary_classifier(target_model: ModelObject, method_name: str) -> None:
    class_to_index = target_model.get_class_to_index()
    if len(class_to_index) != 2:
        raise ValueError(f"{method_name} supports binary classification only")


@dataclass(frozen=True)
class ReviseFeatureContext:
    dataset_name: str
    feature_names: list[str]
    target_column: str
    mutable_mask: np.ndarray
    categorical_groups: list[list[int]]
    thermometer_groups: list[list[int]]
    binary_feature_indices: list[int]
    legality_context: FeatureTweakContext


def _resolve_dataset_name(dataset: DatasetObject) -> str:
    try:
        dataset_name = dataset.attr("name")
    except AttributeError:
        dataset_name = dataset.__class__.__name__.lower()
    return str(dataset_name)


def build_revise_feature_context(trainset: DatasetObject) -> ReviseFeatureContext:
    feature_df = trainset.get(target=False)
    feature_names = list(feature_df.columns)
    feature_type, feature_mutability, feature_actionability = resolve_feature_metadata(
        trainset
    )

    mutable_mask = np.array(
        [
            bool(feature_mutability[feature_name])
            and str(feature_actionability[feature_name]).lower() not in {"none", "same"}
            for feature_name in feature_names
        ],
        dtype=bool,
    )
    if not mutable_mask.any():
        raise ValueError("ReviseMethod requires at least one mutable feature")

    try:
        encoding_map = trainset.attr("encoding")
    except AttributeError:
        encoding_map = {}

    categorical_groups: list[list[int]] = []
    thermometer_groups: list[list[int]] = []
    grouped_features: set[str] = set()

    for encoded_columns in encoding_map.values():
        active_columns = [
            str(column) for column in encoded_columns if str(column) in feature_names
        ]
        if len(active_columns) <= 1:
            continue

        if all("_cat_" in column for column in active_columns):
            categorical_groups.append(
                [feature_names.index(column) for column in active_columns]
            )
            grouped_features.update(active_columns)
            continue

        if all("_therm_" in column for column in active_columns):
            thermometer_groups.append(
                [feature_names.index(column) for column in active_columns]
            )
            grouped_features.update(active_columns)
            continue

        raise ValueError(
            "ReviseMethod supports onehot or thermometer categorical encodings only "
            "after finalize"
        )

    for feature_name in feature_names:
        if str(feature_type[feature_name]).lower() == "categorical":
            raise ValueError(
                "ReviseMethod requires categorical features to be encoded before "
                f"finalize; unsupported feature: {feature_name}"
            )

    binary_feature_indices = [
        index
        for index, feature_name in enumerate(feature_names)
        if str(feature_type[feature_name]).lower() == "binary"
        and feature_name not in grouped_features
    ]

    return ReviseFeatureContext(
        dataset_name=_resolve_dataset_name(trainset),
        feature_names=feature_names,
        target_column=trainset.target_column,
        mutable_mask=mutable_mask,
        categorical_groups=categorical_groups,
        thermometer_groups=thermometer_groups,
        binary_feature_indices=binary_feature_indices,
        legality_context=build_feature_tweak_context(trainset),
    )


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


def differentiable_predict_proba(
    target_model: ModelObject,
    X: torch.Tensor,
) -> torch.Tensor:
    if not getattr(target_model, "_is_trained", False):
        raise RuntimeError("Target model is not trained")
    model = getattr(target_model, "_model", None)
    if model is None:
        raise RuntimeError("Target model network is unavailable")

    raw_logits = model(X.to(target_model._device))
    if raw_logits.ndim == 1:
        raw_logits = raw_logits.unsqueeze(1)

    output_activation = getattr(target_model, "_output_activation_name", "softmax")
    output_activation = str(output_activation).lower()
    if output_activation == "sigmoid":
        if raw_logits.shape[1] != 1:
            raise ValueError(
                "Sigmoid target models must expose a single-logit output layer"
            )
        positive_probability = torch.sigmoid(raw_logits)
        return torch.cat([1.0 - positive_probability, positive_probability], dim=1)
    return torch.softmax(raw_logits, dim=1)


def _encode_targets(
    target_series: pd.Series,
    class_to_index: dict[int | str, int],
) -> pd.Series:
    encoded_values: list[int] = []
    for value in target_series.tolist():
        normalized_value = value
        if isinstance(value, (int, np.integer)):
            normalized_value = int(value)
        elif (
            isinstance(value, (float, np.floating))
            and not pd.isna(value)
            and float(value).is_integer()
        ):
            normalized_value = int(value)

        if normalized_value not in class_to_index:
            raise ValueError(
                "Training labels do not align with the fitted target model classes"
            )
        encoded_values.append(int(class_to_index[normalized_value]))

    return pd.Series(
        encoded_values,
        index=target_series.index.copy(),
        name=target_series.name,
        dtype="int64",
    )


class ReviseTargetModelAdapter:
    def __init__(
        self,
        target_model: ModelObject,
        feature_context: ReviseFeatureContext,
        trainset: DatasetObject,
    ):
        self._target_model = target_model
        self.feature_context = feature_context.legality_context
        self.feature_input_order = list(feature_context.feature_names)
        self._mutable_mask = feature_context.mutable_mask.copy()
        self.backend = "pytorch"
        self.device = str(target_model._device)

        train_features = trainset.get(target=False).loc[:, self.feature_input_order]
        target_series = trainset.get(target=True).iloc[:, 0]
        encoded_target = _encode_targets(
            target_series=target_series,
            class_to_index=target_model.get_class_to_index(),
        )
        train_df = pd.concat(
            [train_features, encoded_target.rename(feature_context.target_column)],
            axis=1,
        )

        self.data = SimpleNamespace(
            target=feature_context.target_column,
            df=train_df,
            categorical_groups=[
                list(group) for group in feature_context.categorical_groups
            ],
            thermometer_groups=[
                list(group) for group in feature_context.thermometer_groups
            ],
            binary_feature_indices=list(feature_context.binary_feature_indices),
        )

    def get_ordered_features(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> pd.DataFrame:
        return to_feature_dataframe(X, self.feature_input_order)

    def get_mutable_mask(self) -> np.ndarray:
        return self._mutable_mask.copy()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return differentiable_predict_proba(self._target_model, X)

    def predict_proba(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if isinstance(X, torch.Tensor):
            return self.forward(X)

        features = self.get_ordered_features(X)
        prediction = self._target_model.get_prediction(features, proba=True)
        if isinstance(prediction, torch.Tensor):
            return prediction.detach().cpu().numpy()
        return np.asarray(prediction)

    def predict(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        probabilities = self.predict_proba(X)
        if isinstance(probabilities, torch.Tensor):
            return probabilities.argmax(dim=1)
        return np.asarray(probabilities).argmax(axis=1)


def check_counterfactuals(
    mlmodel: ReviseTargetModelAdapter,
    counterfactuals: list[np.ndarray] | pd.DataFrame,
    factuals: pd.DataFrame,
    desired_class: int,
) -> pd.DataFrame:
    feature_names = list(mlmodel.feature_input_order)
    factual_features = mlmodel.get_ordered_features(factuals).reindex(
        index=factuals.index,
        columns=feature_names,
    )

    if isinstance(counterfactuals, list):
        if counterfactuals:
            candidate_array = np.asarray(counterfactuals, dtype="float64")
            df_counterfactuals = pd.DataFrame(
                candidate_array,
                index=factual_features.index.copy(),
                columns=feature_names,
            )
        else:
            df_counterfactuals = factual_features.copy(deep=True)
    else:
        df_counterfactuals = mlmodel.get_ordered_features(counterfactuals)
        if df_counterfactuals.shape[0] != factual_features.shape[0]:
            raise ValueError("Counterfactuals must preserve factual row count")
        df_counterfactuals = df_counterfactuals.reindex(
            index=factual_features.index,
            columns=feature_names,
        )

    if df_counterfactuals.shape[0] != factual_features.shape[0]:
        raise ValueError("Counterfactuals must preserve factual row count")

    projected_array = df_counterfactuals.to_numpy(dtype="float64", copy=True)
    factual_array = factual_features.to_numpy(dtype="float64", copy=False)
    for row_index in range(projected_array.shape[0]):
        invalid_mask = ~np.isfinite(projected_array[row_index, :])
        if bool(invalid_mask.any()):
            projected_array[row_index, invalid_mask] = factual_array[
                row_index, invalid_mask
            ]
        projected_array[row_index, :] = project_candidate_features(
            candidate=projected_array[row_index, :],
            factual=factual_array[row_index, :],
            context=mlmodel.feature_context,
        )

    df_counterfactuals = pd.DataFrame(
        projected_array,
        index=factual_features.index.copy(),
        columns=feature_names,
    )

    valid_rows = ~df_counterfactuals.isna().any(axis=1)
    if bool(valid_rows.any()):
        valid_index = df_counterfactuals.index[valid_rows.to_numpy()]
        predicted_labels = mlmodel.predict(df_counterfactuals.loc[valid_index])
        if isinstance(predicted_labels, torch.Tensor):
            predicted_labels = predicted_labels.detach().cpu().numpy()
        success_mask = np.asarray(predicted_labels, dtype=np.int64).reshape(-1) == int(
            desired_class
        )
        failed_index = valid_index[~success_mask]
        if len(failed_index) > 0:
            df_counterfactuals.loc[failed_index, :] = np.nan

    return df_counterfactuals.reindex(
        index=factual_features.index,
        columns=feature_names,
    )
