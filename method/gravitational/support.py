from __future__ import annotations

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

TorchGravitationalModelTypes = (LinearModel, MlpModel)


def ensure_supported_target_model(
    target_model: ModelObject,
    method_name: str,
) -> None:
    if isinstance(target_model, TorchGravitationalModelTypes):
        return
    raise TypeError(
        f"{method_name} supports target models [LinearModel, MlpModel] only, "
        f"received {target_model.__class__.__name__}"
    )


def ensure_binary_classifier(target_model: ModelObject, method_name: str) -> None:
    class_to_index = target_model.get_class_to_index()
    if len(class_to_index) != 2:
        raise ValueError(f"{method_name} supports binary classification only")


def build_gravitational_feature_context(
    trainset: DatasetObject,
) -> FeatureTweakContext:
    return build_feature_tweak_context(trainset)


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


class GravitationalTargetModelAdapter:
    def __init__(
        self,
        target_model: ModelObject,
        feature_context: FeatureTweakContext,
        target_column: str,
        train_features: pd.DataFrame | None = None,
        train_labels: pd.Series | np.ndarray | None = None,
    ):
        self._target_model = target_model
        self.feature_context = feature_context
        self.feature_input_order = list(feature_context.feature_names)
        self.data = SimpleNamespace(target=target_column)
        self.backend = "pytorch"
        self.device = str(target_model._device)
        self.train_features = (
            None
            if train_features is None
            else train_features.loc[:, self.feature_input_order].copy(deep=True)
        )
        if train_labels is None:
            self.train_labels = None
        elif isinstance(train_labels, pd.Series):
            self.train_labels = train_labels.copy(deep=True)
        else:
            self.train_labels = np.asarray(train_labels).copy()

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


def check_counterfactuals(
    mlmodel: GravitationalTargetModelAdapter,
    counterfactuals: list | pd.DataFrame,
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
            array = np.asarray(counterfactuals, dtype="float64")
            df_counterfactuals = pd.DataFrame(
                array,
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

    projected_array = df_counterfactuals.to_numpy(dtype="float64", copy=True)
    factual_array = factual_features.to_numpy(dtype="float64", copy=False)
    for row_index in range(projected_array.shape[0]):
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
