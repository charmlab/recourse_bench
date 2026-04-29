from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import torch
from lime.lime_tabular import LimeTabularExplainer

from dataset.dataset_object import DatasetObject
from method.larr.library.larr import LARRecourse
from method.larr.support import (
    RecourseModelAdapter,
    TorchModelTypes,
    ensure_supported_target_model,
    resolve_feature_groups,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context

LOGGER = logging.getLogger(__name__)


@register("larr")
class LarrMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        lime_seed: int = 0,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, TorchModelTypes, "LarrMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._alpha = float(alpha)
        self._beta = float(beta)
        self._lime_seed = int(lime_seed)
        self._adapter: RecourseModelAdapter | None = None
        self._feature_names: list[str] = []
        self._train_features: pd.DataFrame | None = None
        self._training_array: np.ndarray | None = None
        self._lime_explainer: LimeTabularExplainer | None = None
        self._class_to_index: dict[int | str, int] = {}
        self._lambda_ready = False
        self._method = LARRecourse(
            weights=None,
            bias=None,
            alpha=self._alpha,
            seed=self._seed,
        )

        if self._desired_class is not None and not isinstance(
            self._desired_class, (int, str)
        ):
            raise TypeError("desired_class must be int, str, or None")
        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._alpha < 0:
            raise ValueError("alpha must be >= 0")
        if not 0.0 <= self._beta <= 1.0:
            raise ValueError("beta must satisfy 0 <= beta <= 1")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for LarrMethod.fit()")

        with seed_context(self._seed):
            features = trainset.get(target=False)
            try:
                training_array = features.to_numpy(dtype="float32")
            except ValueError as error:
                raise ValueError(
                    "LarrMethod requires fully numeric input features"
                ) from error

            self._feature_names = list(features.columns)
            self._train_features = features.copy(deep=True)
            self._training_array = training_array
            self._adapter = RecourseModelAdapter(self._target_model, self._feature_names)

            self._class_to_index = self._target_model.get_class_to_index()
            if len(self._class_to_index) != 2:
                raise ValueError(
                    "LarrMethod currently supports binary classification only"
                )
            if (
                self._desired_class is not None
                and self._desired_class not in self._class_to_index
            ):
                raise ValueError("desired_class is invalid for the trained target model")

            feature_groups = resolve_feature_groups(trainset)
            self._method.imm_features = [
                self._feature_names.index(feature_name)
                for feature_name in feature_groups.immutable
            ]

            model = getattr(self._target_model, "_model", None)
            if isinstance(model, torch.nn.Linear):
                self._lime_explainer = None
            else:
                self._lime_explainer = LimeTabularExplainer(
                    training_data=self._training_array,
                    feature_names=self._feature_names,
                    mode="regression",
                    discretize_continuous=False,
                    feature_selection="none",
                    random_state=self._lime_seed,
                )

            self._lambda_ready = False
            self._is_trained = True
            self._ensure_generation_lambda(trigger="fit()")

    def _predict_proba_array(self, X: np.ndarray) -> np.ndarray:
        if self._adapter is None:
            raise RuntimeError("Method is not trained")
        prediction = self._adapter.predict_proba(X)
        if isinstance(prediction, torch.Tensor):
            return prediction.detach().cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(prediction, dtype=np.float32)

    def _get_target_index(self, original_prediction: int) -> int:
        if self._desired_class is not None:
            return int(self._class_to_index[self._desired_class])
        return 1 - int(original_prediction)

    def _get_linear_coefficients(self, target_index: int) -> tuple[np.ndarray, float] | None:
        model = getattr(self._target_model, "_model", None)
        if not isinstance(model, torch.nn.Linear):
            return None

        weight = model.weight.detach().cpu().numpy()
        bias = model.bias.detach().cpu().numpy()

        if weight.shape[0] == 1 and bias.shape[0] == 1:
            direction = 1.0 if target_index == 1 else -1.0
            coeff = direction * weight[0]
            intercept = float(direction * bias[0])
            return coeff.astype(np.float32, copy=False), intercept

        if weight.shape[0] != 2 or bias.shape[0] != 2:
            return None

        other_index = 1 - target_index
        coeff = weight[target_index] - weight[other_index]
        intercept = float(bias[target_index] - bias[other_index])
        return coeff.astype(np.float32, copy=False), intercept

    def _get_lime_coefficients(
        self, factual: np.ndarray, target_index: int
    ) -> tuple[np.ndarray, float]:
        if self._lime_explainer is None:
            raise RuntimeError("LIME explainer is unavailable")

        with seed_context(self._lime_seed):
            explanation = self._lime_explainer.explain_instance(
                data_row=factual.astype(np.float64, copy=False),
                predict_fn=self._predict_proba_array,
                num_features=len(self._feature_names),
            )

        coefficients = np.zeros(len(self._feature_names), dtype=np.float32)
        for feature_index, weight in explanation.local_exp[target_index]:
            coefficients[int(feature_index)] = float(weight)

        intercept = float(
            np.asarray(explanation.intercept[target_index]).reshape(-1)[0]
        )
        return coefficients, intercept

    def _get_surrogate(
        self, factual: np.ndarray, target_index: int
    ) -> tuple[np.ndarray, float]:
        linear_coefficients = self._get_linear_coefficients(target_index)
        if linear_coefficients is not None:
            return linear_coefficients
        return self._get_lime_coefficients(factual, target_index)

    def _select_lambda(self, factuals: pd.DataFrame, target_indices: np.ndarray) -> float:
        best_lambda = float(self._method.lamb)
        best_validity = -1.0
        factual_count = int(factuals.shape[0])

        for lamb in np.arange(0.1, 1.1, 0.1).round(1):
            lambda_start = time.monotonic()
            self._method.lamb = float(lamb)
            candidate_rows: list[np.ndarray] = []

            for row_index, (_, row) in enumerate(factuals.iterrows()):
                factual = row.to_numpy(dtype="float32")
                target_index = int(target_indices[row_index])
                coeff, intercept = self._get_surrogate(factual, target_index)
                self._method.set_weights(coeff)
                self._method.set_bias(intercept)
                candidate_rows.append(
                    np.asarray(
                        self._method.get_robust_recourse(factual), dtype=np.float32
                    )
                )

            candidate_df = pd.DataFrame(
                candidate_rows,
                index=factuals.index,
                columns=self._feature_names,
            )
            valid_rows = ~candidate_df.isna().any(axis=1)
            validity = 0.0
            if bool(valid_rows.any()):
                candidate_prediction = self._adapter.predict_label_indices(
                    candidate_df.loc[valid_rows]
                )
                success = np.zeros(len(candidate_df), dtype=bool)
                success[valid_rows.to_numpy()] = (
                    candidate_prediction.astype(np.int64, copy=False)
                    == target_indices[valid_rows.to_numpy()]
                )
                validity = float(success.mean())

            LOGGER.info(
                "LarrMethod lambda sweep: lambda=%.1f validity=%.4f rows=%d elapsed=%.3fs",
                float(lamb),
                validity,
                factual_count,
                time.monotonic() - lambda_start,
            )

            if validity >= best_validity:
                best_lambda = float(lamb)
                best_validity = validity
            else:
                LOGGER.info(
                    "LarrMethod lambda sweep stopping at lambda %.1f after validity dropped below %.4f; "
                    "keeping lambda %.1f",
                    float(lamb),
                    best_validity,
                    best_lambda,
                )
                break

        LOGGER.info(
            "LarrMethod selected lambda %.1f with best validity %.4f",
            best_lambda,
            best_validity,
        )
        return best_lambda

    def _ensure_generation_lambda(self, trigger: str = "predict()") -> None:
        if self._lambda_ready:
            return
        if self._train_features is None or self._adapter is None:
            raise RuntimeError("Method is not trained")

        original_prediction = self._adapter.predict_label_indices(self._train_features)
        target_indices = np.asarray(
            [
                self._get_target_index(int(prediction))
                for prediction in original_prediction
            ],
            dtype=np.int64,
        )
        needs_recourse = original_prediction != target_indices
        recourse_count = int(needs_recourse.sum())

        if recourse_count > 0:
            selection_start = time.monotonic()
            model = getattr(self._target_model, "_model", self._target_model)
            LOGGER.info(
                "LarrMethod selecting lambda during %s: train_rows=%d recourse_rows=%d "
                "features=%d target_model=%s lime_enabled=%s",
                trigger,
                int(self._train_features.shape[0]),
                recourse_count,
                len(self._feature_names),
                model.__class__.__name__,
                self._lime_explainer is not None,
            )
            self._method.lamb = self._select_lambda(
                self._train_features.loc[needs_recourse],
                target_indices[needs_recourse],
            )
            LOGGER.info(
                "LarrMethod finished lambda selection during %s in %.3fs with lambda %.1f",
                trigger,
                time.monotonic() - selection_start,
                self._method.lamb,
            )
        else:
            LOGGER.warning(
                "LarrMethod found no training instances that require recourse; "
                "keeping default lambda %.1f",
                self._method.lamb,
            )

        self._lambda_ready = True

    def predict(self, testset: DatasetObject, batch_size: int = 20) -> DatasetObject:
        if not self._lambda_ready:
            LOGGER.info(
                "LarrMethod predict() is triggering deferred lambda selection"
            )
        self._ensure_generation_lambda(trigger="predict()")
        return super().predict(testset, batch_size=batch_size)

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")
        if list(factuals.columns) != self._feature_names:
            factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        if not self._lambda_ready:
            LOGGER.info(
                "LarrMethod get_counterfactuals() is triggering deferred lambda selection"
            )
        self._ensure_generation_lambda(trigger="get_counterfactuals()")

        with seed_context(self._seed):
            original_prediction = self._adapter.predict_label_indices(factuals)
            counterfactual_rows: list[pd.Series] = []

            for row_index, (_, row) in enumerate(factuals.iterrows()):
                factual = row.to_numpy(dtype="float32")
                original_index = int(original_prediction[row_index])
                target_index = self._get_target_index(original_index)

                if (
                    self._desired_class is not None
                    and original_index == target_index
                ):
                    counterfactual_rows.append(
                        pd.Series(row.copy(deep=True), index=self._feature_names)
                    )
                    continue

                coeff, intercept = self._get_surrogate(factual, target_index)
                candidate = np.asarray(
                    self._method.run_method(
                        factual.reshape(1, -1),
                        coeff,
                        intercept,
                        beta=self._beta,
                    ),
                    dtype=np.float32,
                )
                if np.all(np.isfinite(candidate)):
                    counterfactual_rows.append(
                        pd.Series(candidate, index=self._feature_names)
                    )
                else:
                    counterfactual_rows.append(
                        pd.Series(np.nan, index=self._feature_names, dtype="float64")
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
