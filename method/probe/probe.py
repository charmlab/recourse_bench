from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from method.method_object import MethodObject
from method.probe.utils import (
    infer_binary_feature_indices,
    infer_categorical_groups,
    probe_optimize,
)
from method.wachter.support import validate_counterfactuals
from model.model_object import ModelObject
from model.model_utils import logits_to_prediction
from preprocess.preprocess_utils import resolve_feature_metadata
from utils.registry import register
from utils.seed import seed_context


@register("probe")
class ProbeMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = 0,
        device: str = "cpu",
        desired_class: int | str | None = None,
        feature_cost: list[float] | None = None,
        lr: float = 0.001,
        lambda_: float = 0.01,
        n_iter: int = 1000,
        max_minutes: float | None = None,
        t_max_min: float | None = None,
        norm: int | float = 1,
        clamp: bool = True,
        loss_type: str = "MSE",
        y_target: int | float | list[float] | tuple[float, ...] = (1.0,),
        binary_cat_features: bool = True,
        noise_variance: float = 0.01,
        invalidation_target: float = 0.45,
        inval_target_eps: float = 0.005,
        **kwargs,
    ):
        feature_costs = kwargs.pop("feature_costs", None)
        if feature_cost is None and feature_costs is not None:
            feature_cost = feature_costs

        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = True
        self._is_trained = False
        self._desired_class = desired_class

        resolved_max_minutes = max_minutes if max_minutes is not None else t_max_min
        if resolved_max_minutes is None:
            resolved_max_minutes = 0.5

        self._feature_cost = list(feature_cost) if feature_cost is not None else None
        self._lr = float(lr)
        self._lambda = float(lambda_)
        self._n_iter = int(n_iter)
        self._max_minutes = float(resolved_max_minutes)
        self._norm = norm
        self._clamp = bool(clamp)
        self._loss_type = str(loss_type).upper()
        self._y_target = self._normalize_y_target(y_target)
        self._binary_cat_features = bool(binary_cat_features)
        self._noise_variance = float(noise_variance)
        self._invalidation_target = float(invalidation_target)
        self._inval_target_eps = float(inval_target_eps)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._lr <= 0:
            raise ValueError("lr must be > 0")
        if self._lambda < 0:
            raise ValueError("lambda_ must be >= 0")
        if self._n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        if self._max_minutes <= 0:
            raise ValueError("max_minutes must be > 0")
        if self._loss_type not in {"BCE", "MSE"}:
            raise ValueError("loss_type must be either 'BCE' or 'MSE'")
        if self._noise_variance <= 0:
            raise ValueError("noise_variance must be > 0")
        if not 0.0 <= self._invalidation_target <= 1.0:
            raise ValueError("invalidation_target must be between 0 and 1")
        if self._inval_target_eps < 0:
            raise ValueError("inval_target_eps must be >= 0")

    @staticmethod
    def _normalize_y_target(
        y_target: int | float | list[float] | tuple[float, ...] | np.ndarray,
    ) -> np.ndarray:
        if isinstance(y_target, (int, float)):
            target_array = np.asarray([float(y_target)], dtype=np.float32)
        else:
            target_array = np.asarray(y_target, dtype=np.float32).reshape(-1)

        if target_array.size == 1:
            positive_target = float(target_array[0])
            if not 0.0 <= positive_target <= 1.0:
                raise ValueError("Scalar y_target must be between 0 and 1")
            return np.asarray(
                [1.0 - positive_target, positive_target],
                dtype=np.float32,
            )

        if target_array.size == 2:
            if not np.all(np.isfinite(target_array)):
                raise ValueError("y_target must contain finite values")
            if np.any(target_array < 0.0) or np.any(target_array > 1.0):
                raise ValueError("y_target values must be between 0 and 1")
            return target_array.astype(np.float32, copy=False)

        raise ValueError("y_target must contain one or two numeric values")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ProbeMethod.fit()")

        with seed_context(self._seed):
            features = trainset.get(target=False)
            try:
                feature_array = features.to_numpy(dtype="float32")
            except ValueError as error:
                raise ValueError(
                    "ProbeMethod requires fully numeric input features"
                ) from error

            if not self._target_model._need_grad:
                raise ValueError("ProbeMethod requires a gradient-enabled target model")

            raw_model = getattr(self._target_model, "_model", None)
            if not isinstance(raw_model, torch.nn.Module):
                raise ValueError(
                    "ProbeMethod currently supports torch-based target models only"
                )

            output_activation = str(
                getattr(self._target_model, "_output_activation_name", "")
            ).lower()
            if output_activation not in {"softmax", "sigmoid"}:
                raise ValueError(
                    "ProbeMethod requires a target model with softmax or sigmoid outputs"
                )

            self._class_to_index = self._target_model.get_class_to_index()
            if len(self._class_to_index) != 2:
                raise ValueError(
                    "ProbeMethod currently supports binary classification only"
                )
            if (
                self._desired_class is not None
                and self._desired_class not in self._class_to_index
            ):
                raise ValueError(
                    "desired_class is invalid for the trained target model"
                )

            try:
                encoding_map = trainset.attr("encoding")
            except AttributeError:
                encoding_map = None

            feature_type, _, _ = resolve_feature_metadata(trainset)
            self._feature_names = list(features.columns)
            self._categorical_groups = infer_categorical_groups(
                self._feature_names,
                encoding_map,
            )
            self._binary_feature_indices = infer_binary_feature_indices(
                self._feature_names,
                feature_type,
                self._categorical_groups,
            )
            self._feature_lower_bounds = feature_array.min(axis=0).astype(
                np.float32,
                copy=False,
            )
            self._feature_upper_bounds = feature_array.max(axis=0).astype(
                np.float32,
                copy=False,
            )
            self._raw_model = raw_model
            self._output_activation_name = output_activation

            if self._feature_cost is not None and len(self._feature_cost) != len(
                self._feature_names
            ):
                raise ValueError(
                    "feature_cost must match the finalized feature dimension"
                )

            self._is_trained = True

    def _predict_probabilities_tensor(
        self,
        X: torch.Tensor,
        target_index: int,
    ) -> torch.Tensor:
        if X.ndim == 1:
            X = X.unsqueeze(0)

        logits = self._raw_model(X.to(self._device))
        probabilities = logits_to_prediction(
            logits,
            proba=True,
            output_activation=self._output_activation_name,
        )
        if target_index == 1:
            return probabilities
        if target_index == 0:
            return probabilities[:, [1, 0]]
        raise ValueError(f"Unsupported binary target index: {target_index}")

    def _get_target_index(self, original_prediction: int) -> int:
        if self._desired_class is not None:
            return int(self._class_to_index[self._desired_class])
        return 1 - int(original_prediction)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")
        if list(factuals.columns) != self._feature_names:
            factuals = factuals.loc[:, self._feature_names].copy(deep=True)

        with seed_context(self._seed):
            original_prediction_tensor = self._target_model.get_prediction(
                factuals,
                proba=True,
            )
            original_predictions = (
                original_prediction_tensor.argmax(dim=1).detach().cpu().numpy()
            )

            self._raw_model.eval()
            counterfactual_rows: list[np.ndarray] = []
            invalidation_rates: list[float] = []
            for row_index, (_, row) in enumerate(factuals.iterrows()):
                factual_array = row.to_numpy(dtype="float32")
                original_prediction = int(original_predictions[row_index])
                target_index = self._get_target_index(original_prediction)

                if (
                    self._desired_class is not None
                    and original_prediction == target_index
                ):
                    counterfactual_rows.append(factual_array.copy())
                    continue

                candidate, invalidation_rate = probe_optimize(
                    probability_model=lambda batch, target_index=target_index: self._predict_probabilities_tensor(
                        batch,
                        target_index=target_index,
                    ),
                    x=factual_array.reshape(1, -1),
                    categorical_groups=self._categorical_groups,
                    binary_feature_indices=self._binary_feature_indices,
                    binary_cat_features=self._binary_cat_features,
                    feature_costs=self._feature_cost,
                    feature_lower_bounds=self._feature_lower_bounds,
                    feature_upper_bounds=self._feature_upper_bounds,
                    lr=self._lr,
                    lambda_param=self._lambda,
                    y_target=self._y_target,
                    n_iter=self._n_iter,
                    max_minutes=self._max_minutes,
                    norm=self._norm,
                    clamp=self._clamp,
                    loss_type=self._loss_type,
                    invalidation_target=self._invalidation_target,
                    inval_target_eps=self._inval_target_eps,
                    noise_variance=self._noise_variance,
                    seed=int(self._seed or 0),
                    device=self._device,
                )
                invalidation_rates.append(float(invalidation_rate))

                if np.all(np.isfinite(candidate)):
                    counterfactual_rows.append(candidate.astype(np.float32, copy=False))
                else:
                    counterfactual_rows.append(
                        np.full(len(self._feature_names), np.nan, dtype=np.float32)
                    )

            if invalidation_rates:
                logging.getLogger(__name__).info(
                    "average invalidation rate of generated counterfactuals: %.4f",
                    float(np.mean(np.asarray(invalidation_rates, dtype=np.float32))),
                )

            candidates = pd.DataFrame(
                counterfactual_rows,
                index=factuals.index,
                columns=self._feature_names,
            )
            return validate_counterfactuals(
                self._target_model,
                factuals,
                candidates,
                desired_class=self._desired_class,
            )
