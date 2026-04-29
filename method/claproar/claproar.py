from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.claproar.model import ClaPROAR
from method.claproar.support import (
    ClaproarTargetModelAdapter,
    ensure_binary_classifier,
    ensure_supported_target_model,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from model.model_utils import resolve_device
from utils.registry import register
from utils.seed import seed_context


def _as_label_array(predictions: object) -> np.ndarray:
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    return np.asarray(predictions, dtype=np.int64).reshape(-1)


@register("claproar")
class ClaproarMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = 1,
        individual_cost_lambda: float = 0.1,
        external_cost_lambda: float = 0.1,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-4,
        **kwargs,
    ):
        del kwargs
        ensure_supported_target_model(target_model, "ClaproarMethod")

        self._target_model = target_model
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = True
        self._is_trained = False
        self._desired_class = desired_class

        self._individual_cost_lambda = float(individual_cost_lambda)
        self._external_cost_lambda = float(external_cost_lambda)
        self._learning_rate = float(learning_rate)
        self._max_iter = int(max_iter)
        self._tol = float(tol)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self._max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self._tol < 0:
            raise ValueError("tol must be >= 0")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ClaproarMethod.fit()")

        with seed_context(self._seed):
            ensure_binary_classifier(self._target_model, "ClaproarMethod")
            class_to_index = self._target_model.get_class_to_index()
            if (
                self._desired_class is not None
                and self._desired_class not in class_to_index
            ):
                raise ValueError(
                    "desired_class is invalid for the trained target model"
                )

            train_features = trainset.get(target=False)
            try:
                train_features.loc[:, :].to_numpy(dtype="float32")
            except ValueError as error:
                raise ValueError(
                    "ClaproarMethod requires finalized numeric input features"
                ) from error

            self._feature_names = list(train_features.columns)
            self._adapter = ClaproarTargetModelAdapter(
                target_model=self._target_model,
                feature_names=self._feature_names,
                target_column=trainset.target_column,
            )
            if self._desired_class is None:
                self._claproar_by_target = {
                    target_class: ClaPROAR(
                        mlmodel=self._adapter,
                        device=self._device,
                        individual_cost_lambda=self._individual_cost_lambda,
                        external_cost_lambda=self._external_cost_lambda,
                        learning_rate=self._learning_rate,
                        max_iter=self._max_iter,
                        tol=self._tol,
                        target_class=target_class,
                    )
                    for target_class in (0, 1)
                }
            else:
                self._claproar = ClaPROAR(
                    mlmodel=self._adapter,
                    device=self._device,
                    individual_cost_lambda=self._individual_cost_lambda,
                    external_cost_lambda=self._external_cost_lambda,
                    learning_rate=self._learning_rate,
                    max_iter=self._max_iter,
                    tol=self._tol,
                    target_class=int(class_to_index[self._desired_class]),
                )
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        if self._desired_class is not None:
            counterfactuals = self._claproar.get_counterfactuals(factuals=factuals)
            return counterfactuals.loc[:, self._feature_names].copy(deep=True)

        ordered_factuals = self._adapter.get_ordered_features(factuals)
        if ordered_factuals.empty:
            return ordered_factuals.loc[:, self._feature_names].copy(deep=True)

        predicted_labels = _as_label_array(self._adapter.predict(ordered_factuals))
        target_labels = 1 - predicted_labels
        counterfactuals = pd.DataFrame(
            np.nan,
            index=ordered_factuals.index.copy(),
            columns=self._feature_names,
        )

        for target_class, claproar in self._claproar_by_target.items():
            subset = ordered_factuals.loc[target_labels == target_class]
            if subset.empty:
                continue
            subset_counterfactuals = claproar.get_counterfactuals(factuals=subset)
            counterfactuals.loc[subset.index, self._feature_names] = (
                subset_counterfactuals.reindex(
                    index=subset.index,
                    columns=self._feature_names,
                )
            )

        return counterfactuals.loc[:, self._feature_names].copy(deep=True)
