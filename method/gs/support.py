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
from model.randomforest.randomforest import RandomForestModel
from preprocess.preprocess_utils import resolve_feature_metadata

TorchModelTypes = (LinearModel, MlpModel)
BlackBoxModelTypes = (LinearModel, MlpModel, RandomForestModel)


def ensure_supported_target_model(
    target_model: ModelObject,
    supported_types: Sequence[type[ModelObject]],
    method_name: str,
) -> None:
    if isinstance(target_model, tuple(supported_types)):
        return

    supported_names = ", ".join(cls.__name__ for cls in supported_types)
    raise TypeError(
        f"{method_name} supports target models [{supported_names}] only, "
        f"received {target_model.__class__.__name__}"
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
        if isinstance(X, torch.Tensor) and isinstance(
            self._target_model, TorchModelTypes
        ):
            return differentiable_predict_proba(self._target_model, X)

        features = self.get_ordered_features(X)
        prediction = self._target_model.get_prediction(features, proba=True)
        if isinstance(prediction, torch.Tensor):
            return prediction.detach().cpu().numpy()
        return np.asarray(prediction)

    def predict(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if isinstance(X, torch.Tensor) and isinstance(
            self._target_model, TorchModelTypes
        ):
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
        if isinstance(X, torch.Tensor) and isinstance(
            self._target_model, TorchModelTypes
        ):
            probabilities = differentiable_predict_proba(self._target_model, X)
            return probabilities.argmax(dim=1).detach().cpu().numpy()

        features = self.get_ordered_features(X)
        prediction = self._target_model.get_prediction(features, proba=False)
        if isinstance(prediction, torch.Tensor):
            return prediction.detach().cpu().numpy().argmax(axis=1)
        return np.asarray(prediction).argmax(axis=1)


def differentiable_predict_proba(
    target_model: ModelObject,
    X: torch.Tensor,
) -> torch.Tensor:
    ensure_supported_target_model(
        target_model,
        TorchModelTypes,
        "differentiable_predict_proba",
    )
    model = getattr(target_model, "_model", None)
    if model is None:
        raise RuntimeError("Target model has not been initialized")

    device = target_model._device
    logits = model(X.to(device))
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
