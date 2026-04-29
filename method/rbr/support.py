from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from model.randomforest.randomforest import RandomForestModel

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


class RecourseModelAdapter:
    def __init__(self, target_model: ModelObject, feature_names: Sequence[str]):
        self._target_model = target_model
        self._feature_names = list(feature_names)

    def get_ordered_features(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> pd.DataFrame:
        return to_feature_dataframe(X, self._feature_names)

    def predict(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        features = self.get_ordered_features(X)
        return self._target_model.get_prediction(features, proba=True)

    def _to_feature_tensor(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(X, pd.DataFrame):
            array = X.loc[:, self._feature_names].to_numpy(dtype=np.float32, copy=False)
            tensor = torch.tensor(array, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            tensor = X.detach().to(dtype=torch.float32)
        else:
            tensor = torch.tensor(np.asarray(X), dtype=torch.float32)

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self._target_model._device)

    def predict_label_indices(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        if isinstance(self._target_model, TorchModelTypes):
            features = self._to_feature_tensor(X)
            model = self._target_model._model
            if model is None:
                raise RuntimeError("Target model is not trained")
            with torch.no_grad():
                model.eval()
                logits = model(features)
                if self._target_model._output_activation_name == "sigmoid":
                    if logits.ndim == 1:
                        logits = logits.unsqueeze(1)
                    logits = torch.cat([-logits, logits], dim=1)
            return logits.detach().cpu().numpy().argmax(axis=1)

        probabilities = self.predict(X)
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


def resolve_onehot_feature_indices(
    dataset: DatasetObject,
    feature_names: Sequence[str],
) -> list[list[int]]:
    if not hasattr(dataset, "encoding"):
        return []

    encoding = dataset.attr("encoding")
    onehot_groups: list[list[int]] = []
    for source_feature, encoded_columns in encoding.items():
        if not isinstance(source_feature, str) or not isinstance(encoded_columns, list):
            continue
        expected_prefix = f"{source_feature}_cat_"
        if len(encoded_columns) < 2:
            continue
        if not all(
            isinstance(column, str) and column.startswith(expected_prefix)
            for column in encoded_columns
        ):
            continue

        group_indices = [
            int(feature_names.index(column))
            for column in encoded_columns
            if column in feature_names
        ]
        if len(group_indices) == len(encoded_columns):
            onehot_groups.append(group_indices)
    return onehot_groups


def apply_onehot_constraints(
    values: np.ndarray | Sequence[float],
    onehot_feature_indices: Sequence[Sequence[int]],
) -> np.ndarray:
    constrained = np.asarray(values, dtype=np.float32).copy()
    for group in onehot_feature_indices:
        group_indices = list(group)
        if len(group_indices) == 0:
            continue
        if len(group_indices) == 1:
            constrained[group_indices[0]] = np.float32(
                np.round(constrained[group_indices[0]])
            )
            continue

        best_local_index = int(np.argmax(constrained[group_indices]))
        constrained[group_indices] = 0.0
        constrained[group_indices[best_local_index]] = 1.0
    return constrained
