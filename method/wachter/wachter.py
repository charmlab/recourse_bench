from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.method_object import MethodObject
from method.wachter.model import Wachter
from method.wachter.support import (
    WachterTargetModelAdapter,
    build_wachter_feature_context,
    ensure_binary_classifier,
    ensure_supported_target_model,
)
from model.model_object import ModelObject
from model.model_utils import resolve_device
from utils.registry import register
from utils.seed import seed_context


@register("wachter")
class WachterMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = 1,
        feature_cost: list[float] | None = None,
        learning_rate: float = 0.01,
        lambda_: float = 0.01,
        max_iter: int = 1000,
        max_minutes: float = 0.5,
        norm: int = 1,
        clamp: bool = True,
        loss_type: str = "BCE",
        **kwargs,
    ):
        del kwargs
        ensure_supported_target_model(target_model, "WachterMethod")

        self._target_model = target_model
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = True
        self._is_trained = False
        self._desired_class = desired_class

        self._feature_cost = (
            None
            if feature_cost is None
            else np.asarray(feature_cost, dtype=np.float32).reshape(-1)
        )
        self._learning_rate = float(learning_rate)
        self._lambda_param = float(lambda_)
        self._max_iter = int(max_iter)
        self._max_minutes = float(max_minutes)
        self._norm = int(norm)
        self._clamp = bool(clamp)
        self._loss_type = str(loss_type).upper()

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._desired_class is None:
            raise ValueError("WachterMethod requires desired_class to be set")
        if self._learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self._lambda_param < 0:
            raise ValueError("lambda_ must be >= 0")
        if self._max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self._max_minutes <= 0:
            raise ValueError("max_minutes must be > 0")
        if self._norm < 1:
            raise ValueError("norm must be >= 1")
        if self._loss_type not in {"BCE", "MSE"}:
            raise ValueError("loss_type must be 'BCE' or 'MSE'")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for WachterMethod.fit()")
        if not getattr(self._target_model, "_is_trained", False):
            raise RuntimeError("Target model must be trained before method.fit()")

        with seed_context(self._seed):
            ensure_binary_classifier(self._target_model, "WachterMethod")
            class_to_index = self._target_model.get_class_to_index()
            if self._desired_class not in class_to_index:
                raise ValueError(
                    "desired_class is invalid for the trained target model"
                )

            train_features = trainset.get(target=False)
            try:
                train_features.loc[:, :].to_numpy(dtype="float32")
            except ValueError as error:
                raise ValueError(
                    "WachterMethod requires finalized numeric input features"
                ) from error

            self._feature_context = build_wachter_feature_context(trainset)
            self._feature_names = list(self._feature_context.feature_names)
            if self._feature_cost is not None and self._feature_cost.shape[0] != len(
                self._feature_names
            ):
                raise ValueError(
                    "feature_cost length must match the finalized feature count"
                )

            self._desired_class_index = int(class_to_index[self._desired_class])
            self._adapter = WachterTargetModelAdapter(
                target_model=self._target_model,
                feature_context=self._feature_context,
            )

            if self._loss_type == "BCE":
                y_target = [1.0, 0.0]
                y_target[self._desired_class_index] = 1.0
                y_target[1 - self._desired_class_index] = 0.0
            else:
                y_target = [1.0] if self._desired_class_index == 1 else [-1.0]

            hyperparams = {
                "feature_cost": (
                    None if self._feature_cost is None else self._feature_cost.copy()
                ),
                "lr": self._learning_rate,
                "lambda_": self._lambda_param,
                "n_iter": self._max_iter,
                "t_max_min": self._max_minutes,
                "norm": self._norm,
                "clamp": self._clamp,
                "loss_type": self._loss_type,
                "y_target": y_target,
            }

            self._wachter = Wachter(
                mlmodel=self._adapter,
                hyperparams=hyperparams,
            )
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        with seed_context(self._seed):
            counterfactuals = self._wachter.get_counterfactuals(factuals=factuals)
        return counterfactuals.reindex(
            index=factuals.index,
            columns=factuals.columns,
        ).copy(deep=True)
