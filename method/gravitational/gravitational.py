from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.gravitational.model import Gravitational
from method.gravitational.support import (
    GravitationalTargetModelAdapter,
    build_gravitational_feature_context,
    ensure_binary_classifier,
    ensure_supported_target_model,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from model.model_utils import resolve_device
from utils.registry import register
from utils.seed import seed_context


def _encode_targets(
    target_series: pd.Series,
    class_to_index: dict[int | str, int],
) -> pd.Series:
    encoded_values: list[int] = []
    for value in target_series.tolist():
        normalized_value = value
        if isinstance(value, (np.integer, int)):
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


@register("gravitational")
class GravitationalMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = 1,
        prediction_loss_lambda: float = 1.0,
        original_dist_lambda: float = 0.5,
        grav_penalty_lambda: float = 1.5,
        learning_rate: float = 0.01,
        num_steps: int = 100,
        scheduler_step_size: int = 100,
        scheduler_gamma: float = 0.5,
        x_center: list[float] | None = None,
        **kwargs,
    ):
        del kwargs
        ensure_supported_target_model(target_model, "GravitationalMethod")

        self._target_model = target_model
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = True
        self._is_trained = False
        self._desired_class = desired_class

        self._prediction_loss_lambda = float(prediction_loss_lambda)
        self._original_dist_lambda = float(original_dist_lambda)
        self._grav_penalty_lambda = float(grav_penalty_lambda)
        self._learning_rate = float(learning_rate)
        self._num_steps = int(num_steps)
        self._scheduler_step_size = int(scheduler_step_size)
        self._scheduler_gamma = float(scheduler_gamma)
        self._x_center_override = (
            None
            if x_center is None
            else np.asarray(x_center, dtype=np.float32).reshape(-1)
        )

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._desired_class is None:
            raise ValueError("GravitationalMethod requires desired_class to be set")
        if self._prediction_loss_lambda < 0:
            raise ValueError("prediction_loss_lambda must be >= 0")
        if self._original_dist_lambda < 0:
            raise ValueError("original_dist_lambda must be >= 0")
        if self._grav_penalty_lambda < 0:
            raise ValueError("grav_penalty_lambda must be >= 0")
        if self._learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self._num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self._scheduler_step_size < 1:
            raise ValueError("scheduler_step_size must be >= 1")
        if self._scheduler_gamma <= 0:
            raise ValueError("scheduler_gamma must be > 0")

    def _compute_default_x_center(
        self,
        train_features: pd.DataFrame,
        encoded_target: pd.Series,
        desired_class_index: int,
    ) -> np.ndarray:
        mask = encoded_target == int(desired_class_index)
        if bool(mask.any()):
            x_center = (
                train_features.loc[mask.to_numpy()]
                .mean(axis=0)
                .to_numpy(dtype=np.float32)
            )
        else:
            predicted_labels = self._adapter.predict(train_features)
            if isinstance(predicted_labels, np.ndarray):
                predicted_array = predicted_labels.reshape(-1)
            else:
                predicted_array = predicted_labels.detach().cpu().numpy().reshape(-1)
            target_mask = predicted_array == int(desired_class_index)
            if np.asarray(target_mask).sum() > 0:
                x_center = (
                    train_features.loc[target_mask]
                    .mean(axis=0)
                    .to_numpy(dtype=np.float32)
                )
            else:
                x_center = train_features.mean(axis=0).to_numpy(dtype=np.float32)

        return np.nan_to_num(x_center, nan=0.0, posinf=1e6, neginf=-1e6)

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for GravitationalMethod.fit()")
        if not getattr(self._target_model, "_is_trained", False):
            raise RuntimeError("Target model must be trained before method.fit()")

        with seed_context(self._seed):
            ensure_binary_classifier(self._target_model, "GravitationalMethod")
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
                    "GravitationalMethod requires finalized numeric input features"
                ) from error

            self._feature_context = build_gravitational_feature_context(trainset)
            self._feature_names = list(self._feature_context.feature_names)
            self._desired_class_index = int(class_to_index[self._desired_class])
            ordered_train_features = train_features.loc[:, self._feature_names].copy(
                deep=True
            )
            encoded_target = _encode_targets(
                trainset.get(target=True).iloc[:, 0],
                class_to_index=class_to_index,
            )

            self._adapter = GravitationalTargetModelAdapter(
                target_model=self._target_model,
                feature_context=self._feature_context,
                target_column=trainset.target_column,
                train_features=ordered_train_features,
                train_labels=encoded_target,
            )

            if self._x_center_override is not None:
                if self._x_center_override.shape[0] != len(self._feature_names):
                    raise ValueError(
                        "x_center length must match the finalized feature count"
                    )
                x_center = self._x_center_override.copy()
            else:
                x_center = self._compute_default_x_center(
                    train_features=ordered_train_features,
                    encoded_target=encoded_target,
                    desired_class_index=self._desired_class_index,
                )

            hyperparams = {
                "prediction_loss_lambda": self._prediction_loss_lambda,
                "original_dist_lambda": self._original_dist_lambda,
                "grav_penalty_lambda": self._grav_penalty_lambda,
                "learning_rate": self._learning_rate,
                "num_steps": self._num_steps,
                "target_class": self._desired_class_index,
                "scheduler_step_size": self._scheduler_step_size,
                "scheduler_gamma": self._scheduler_gamma,
            }

            self._gravitational = Gravitational(
                mlmodel=self._adapter,
                hyperparams=hyperparams,
                x_center=x_center,
            )
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        with seed_context(self._seed):
            counterfactuals = self._gravitational.get_counterfactuals(factuals=factuals)
        return counterfactuals.loc[:, self._feature_names].copy(deep=True)
