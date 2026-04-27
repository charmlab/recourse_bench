from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject

TorchClaproarModelTypes = (LinearModel, MlpModel)


def ensure_supported_target_model(
    target_model: ModelObject,
    method_name: str,
) -> None:
    if isinstance(target_model, TorchClaproarModelTypes):
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


class ClaproarTargetModelAdapter:
    def __init__(
        self,
        target_model: ModelObject,
        feature_names: Sequence[str],
        target_column: str,
    ):
        self._target_model = target_model
        self.feature_input_order = list(feature_names)
        self.data = SimpleNamespace(target=target_column)

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
    mlmodel: ClaproarTargetModelAdapter,
    counterfactuals: list | pd.DataFrame,
    factuals_index: pd.Index,
    negative_label: int = 0,
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

    df_cfs[mlmodel.data.target] = np.asarray(mlmodel.predict_proba(df_cfs)).argmax(
        axis=1
    )
    df_cfs.loc[df_cfs[mlmodel.data.target] == negative_label, :] = np.nan
    return df_cfs
