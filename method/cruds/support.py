from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from preprocess.preprocess_utils import resolve_feature_metadata

TorchCrudsModelTypes = (LinearModel, MlpModel)


def ensure_supported_target_model(
    target_model: ModelObject,
    method_name: str,
) -> None:
    if isinstance(target_model, TorchCrudsModelTypes):
        return
    raise TypeError(
        f"{method_name} supports target models [LinearModel, MlpModel] only, "
        f"received {target_model.__class__.__name__}"
    )


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


@dataclass
class CrudsFeatureContext:
    dataset_name: str
    feature_names: list[str]
    target_column: str
    mutable_mask: np.ndarray
    categorical_groups: list[list[int]]
    binary_feature_indices: list[int]


def _resolve_dataset_name(dataset: DatasetObject) -> str:
    try:
        dataset_name = dataset.attr("name")
    except AttributeError:
        dataset_name = dataset.__class__.__name__.lower()
    return str(dataset_name)


def build_cruds_feature_context(trainset: DatasetObject) -> CrudsFeatureContext:
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
        raise ValueError("CrudsMethod requires at least one mutable feature")

    try:
        encoding_map = trainset.attr("encoding")
    except AttributeError:
        encoding_map = {}

    categorical_groups: list[list[int]] = []
    grouped_features: set[str] = set()
    for encoded_columns in encoding_map.values():
        active_columns = [
            column for column in encoded_columns if column in feature_names
        ]
        if len(active_columns) <= 1:
            continue
        if not all("_cat_" in column for column in active_columns):
            raise ValueError(
                "CrudsMethod supports onehot categorical groups only after finalize"
            )

        group_indices = [feature_names.index(column) for column in active_columns]
        categorical_groups.append(group_indices)
        grouped_features.update(active_columns)

    for feature_name in feature_names:
        feature_kind = str(feature_type[feature_name]).lower()
        if feature_kind == "categorical":
            raise ValueError(
                "CrudsMethod requires categorical features to be onehot encoded "
                f"before finalize; unsupported feature: {feature_name}"
            )

    binary_feature_indices = [
        index
        for index, feature_name in enumerate(feature_names)
        if str(feature_type[feature_name]).lower() == "binary"
        and feature_name not in grouped_features
    ]

    return CrudsFeatureContext(
        dataset_name=_resolve_dataset_name(trainset),
        feature_names=feature_names,
        target_column=trainset.target_column,
        mutable_mask=mutable_mask,
        categorical_groups=categorical_groups,
        binary_feature_indices=binary_feature_indices,
    )


def _encode_targets(
    target_series: pd.Series,
    class_to_index: dict[int | str, int],
) -> pd.Series:
    encoded_values: list[int] = []
    for value in target_series.tolist():
        normalized_value = value
        if (
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


class CrudsTargetModelAdapter:
    def __init__(
        self,
        target_model: ModelObject,
        feature_context: CrudsFeatureContext,
        trainset: DatasetObject,
    ):
        self._target_model = target_model
        self.feature_input_order = list(feature_context.feature_names)
        self._mutable_mask = feature_context.mutable_mask.copy()

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
            binary_feature_indices=list(feature_context.binary_feature_indices),
        )

    def get_ordered_features(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> pd.DataFrame:
        return to_feature_dataframe(X, self.feature_input_order)

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
        probabilities = self.predict_proba(X)
        if isinstance(probabilities, torch.Tensor):
            return probabilities.argmax(dim=1)
        return np.asarray(probabilities).argmax(axis=1)

    def get_mutable_mask(self) -> np.ndarray:
        return self._mutable_mask.copy()


def check_counterfactuals(
    mlmodel: CrudsTargetModelAdapter,
    counterfactuals: list | pd.DataFrame,
    factuals_index: pd.Index,
    desired_class: int,
) -> pd.DataFrame:
    if isinstance(counterfactuals, list):
        df_cfs = pd.DataFrame(
            np.asarray(counterfactuals),
            columns=mlmodel.feature_input_order,
            index=factuals_index.copy(),
        )
    else:
        df_cfs = counterfactuals.copy(deep=True)
        if df_cfs.shape[0] != len(factuals_index):
            raise ValueError(
                "Counterfactual rows must match the number of factual rows"
            )
        df_cfs.index = factuals_index.copy()

    df_cfs = df_cfs.loc[:, mlmodel.feature_input_order].copy(deep=True)
    predicted = np.asarray(mlmodel.predict(df_cfs)).reshape(-1)
    df_cfs[mlmodel.data.target] = predicted
    df_cfs.loc[predicted != int(desired_class), :] = np.nan
    return df_cfs
