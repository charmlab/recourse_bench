from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression

from recourse_bench.dataset.dataset_object import DatasetObject
from recourse_bench.method.method_object import MethodObject
from recourse_bench.method.roar.utils import infer_categorical_groups, roar_optimize
from recourse_bench.model.model_object import ModelObject
from recourse_bench.utils.registry import register
from recourse_bench.utils.seed import seed_context


@register("roar")
class RoarMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = 0,
        device: str = "cpu",
        desired_class: int | str | None = None,
        lr: float = 0.001,
        lambda_: float = 0.1,
        delta_max: float = 0.1,
        norm: int | float = 1,
        max_minutes: float = 0.5,
        loss_type: str = "BCE",
        loss_threshold: float = 1e-4,
        lime_seed: int = 0,
        discretize_continuous: bool = False,
        enforce_encoding: bool = False,
        sample_around_instance: bool = True,
        **kwargs,
    ):
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._lr = float(lr)
        self._lambda = float(lambda_)
        self._delta_max = float(delta_max)
        self._norm = norm
        self._max_minutes = float(max_minutes)
        self._loss_type = str(loss_type).upper()
        self._loss_threshold = float(loss_threshold)
        self._lime_seed = int(lime_seed)
        self._discretize_continuous = bool(discretize_continuous)
        self._enforce_encoding = bool(enforce_encoding)
        self._sample_around_instance = bool(sample_around_instance)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._lr <= 0:
            raise ValueError("lr must be > 0")
        if self._lambda < 0:
            raise ValueError("lambda_ must be >= 0")
        if self._delta_max < 0:
            raise ValueError("delta_max must be >= 0")
        if self._max_minutes <= 0:
            raise ValueError("max_minutes must be > 0")
        if self._loss_type not in {"BCE", "MSE"}:
            raise ValueError("loss_type must be either 'BCE' or 'MSE'")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for RoarMethod.fit()")

        with seed_context(self._seed):
            features = trainset.get(target=False)
            try:
                training_array = features.to_numpy(dtype="float32")
            except ValueError as error:
                raise ValueError(
                    "RoarMethod requires fully numeric input features"
                ) from error

            self._feature_names = list(features.columns)
            if hasattr(trainset, "encoded_feature_type"):
                self._feature_type = deepcopy(trainset.attr("encoded_feature_type"))
                self._feature_mutability = deepcopy(
                    trainset.attr("encoded_feature_mutability")
                )
                self._feature_actionability = deepcopy(
                    trainset.attr("encoded_feature_actionability")
                )
            else:
                self._feature_type = deepcopy(trainset.attr("raw_feature_type"))
                self._feature_mutability = deepcopy(
                    trainset.attr("raw_feature_mutability")
                )
                self._feature_actionability = deepcopy(
                    trainset.attr("raw_feature_actionability")
                )

            self._class_to_index = self._target_model.get_class_to_index()
            if len(self._class_to_index) != 2:
                raise ValueError(
                    "RoarMethod currently supports binary classification only"
                )
            if (
                self._desired_class is not None
                and self._desired_class not in self._class_to_index
            ):
                raise ValueError(
                    "desired_class is invalid for the trained target model"
                )

            self._training_array = training_array
            self._categorical_groups = infer_categorical_groups(self._feature_names)
            self._lime_explainer = LimeTabularExplainer(
                training_data=self._training_array,
                feature_names=self._feature_names,
                discretize_continuous=self._discretize_continuous,
                feature_selection="none",
                sample_around_instance=self._sample_around_instance,
                random_state=self._lime_seed,
            )
            self._is_trained = True

    def _predict_proba_array(self, X: np.ndarray) -> np.ndarray:
        features = pd.DataFrame(X, columns=self._feature_names)
        prediction = self._target_model.get_prediction(features, proba=True)
        if isinstance(prediction, torch.Tensor):
            return prediction.detach().cpu().numpy()
        return np.asarray(prediction, dtype=np.float32)

    def _predict_label_array(self, X: np.ndarray) -> np.ndarray:
        features = pd.DataFrame(X, columns=self._feature_names)
        prediction = self._target_model.get_prediction(features, proba=False)
        if isinstance(prediction, torch.Tensor):
            return prediction.detach().cpu().numpy()
        return np.asarray(prediction, dtype=np.float32)

    def _get_target_index(self, original_prediction: int) -> int:
        if self._desired_class is not None:
            return int(self._class_to_index[self._desired_class])
        return 1 - int(original_prediction)

    def _get_linear_coefficients(
        self, target_index: int
    ) -> tuple[np.ndarray, float] | None:
        model = getattr(self._target_model, "_model", None)
        if not isinstance(model, torch.nn.Linear):
            return None

        weight = model.weight.detach().cpu().numpy()
        bias = model.bias.detach().cpu().numpy()
        if weight.shape[0] == 1 and bias.shape[0] == 1:
            if len(self._class_to_index) != 2:
                return None
            direction = 1.0 if target_index == 1 else -1.0
            coeff = direction * weight[0]
            intercept = float(direction * bias[0])
            return coeff.astype(np.float32), intercept
        if weight.shape[0] != 2 or bias.shape[0] != 2:
            return None

        other_index = 1 - target_index
        coeff = weight[target_index] - weight[other_index]
        intercept = float(bias[target_index] - bias[other_index])
        return coeff.astype(np.float32), intercept

    def _get_lime_coefficients(
        self, factual: np.ndarray, target_index: int
    ) -> tuple[np.ndarray, float]:
        with seed_context(self._lime_seed):
            explanation = self._lime_explainer.explain_instance(
                data_row=factual.astype(np.float64, copy=False),
                predict_fn=self._predict_label_array,
                labels=(target_index,),
                num_features=len(self._feature_names),
                model_regressor=LogisticRegression(random_state=self._lime_seed),
            )

        coefficients = np.zeros(len(self._feature_names), dtype=np.float32)
        local_exp = explanation.local_exp[target_index]
        if local_exp:
            first_entry = local_exp[0]
            feature_indices = np.asarray(first_entry[0])
            weights = np.asarray(first_entry[1], dtype=np.float32)
            if weights.ndim > 0 and weights.size == len(self._feature_names):
                coefficients = weights.reshape(-1).astype(np.float32, copy=False)
            elif feature_indices.ndim > 0 and weights.ndim > 0:
                flat_indices = feature_indices.reshape(-1).astype(int)
                flat_weights = weights.reshape(-1).astype(np.float32)
                coefficients[flat_indices] = flat_weights
            else:
                for feature_index, weight in local_exp:
                    weight_array = np.asarray(weight, dtype=np.float32)
                    feature_index_array = np.asarray(feature_index)
                    if weight_array.ndim > 0 and weight_array.size == len(
                        self._feature_names
                    ):
                        coefficients = weight_array.reshape(-1).astype(
                            np.float32, copy=False
                        )
                        break
                    if feature_index_array.ndim > 0 and weight_array.ndim > 0:
                        flat_indices = feature_index_array.reshape(-1).astype(int)
                        flat_weights = weight_array.reshape(-1).astype(np.float32)
                        coefficients[flat_indices] = flat_weights
                        continue
                    coefficients[int(feature_index)] = float(weight_array.item())

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

    def _is_successful_prediction(
        self, prediction: int, original_prediction: int, target_index: int
    ) -> bool:
        if self._desired_class is None:
            return prediction != original_prediction and prediction == target_index
        return prediction == target_index

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")
        if list(factuals.columns) != self._feature_names:
            factuals = factuals.loc[:, self._feature_names].copy(deep=True)

        with seed_context(self._seed):
            original_prediction_tensor = self._target_model.get_prediction(
                factuals, proba=True
            )
            original_predictions = (
                original_prediction_tensor.argmax(dim=1).detach().cpu().numpy()
            )

            counterfactual_rows: list[pd.Series] = []
            for row_index, (_, row) in enumerate(factuals.iterrows()):
                factual_array = row.to_numpy(dtype="float32")
                original_prediction = int(original_predictions[row_index])
                target_index = self._get_target_index(original_prediction)

                if (
                    self._desired_class is not None
                    and original_prediction == target_index
                ):
                    counterfactual_rows.append(
                        pd.Series(row.copy(deep=True), index=self._feature_names)
                    )
                    continue

                coefficients, intercept = self._get_surrogate(
                    factual=factual_array,
                    target_index=target_index,
                )
                candidate = roar_optimize(
                    x=factual_array,
                    coeff=coefficients,
                    intercept=intercept,
                    cat_feature_indices=self._categorical_groups,
                    lr=self._lr,
                    lambda_param=self._lambda,
                    delta_max=self._delta_max,
                    norm=self._norm,
                    loss_type=self._loss_type,
                    loss_threshold=self._loss_threshold,
                    max_minutes=self._max_minutes,
                    enforce_encoding=self._enforce_encoding,
                    seed=self._seed or 0,
                    device=self._device,
                )

                if not np.all(np.isfinite(candidate)):
                    counterfactual_rows.append(
                        pd.Series(np.nan, index=self._feature_names, dtype="float64")
                    )
                    continue

                candidate_df = pd.DataFrame(
                    [candidate], index=[row.name], columns=self._feature_names
                )
                candidate_prediction = (
                    self._target_model.get_prediction(candidate_df, proba=True)
                    .argmax(dim=1)
                    .item()
                )

                if self._is_successful_prediction(
                    prediction=int(candidate_prediction),
                    original_prediction=original_prediction,
                    target_index=target_index,
                ):
                    counterfactual_rows.append(
                        pd.Series(candidate, index=self._feature_names)
                    )
                else:
                    counterfactual_rows.append(
                        pd.Series(np.nan, index=self._feature_names, dtype="float64")
                    )

            return pd.DataFrame(counterfactual_rows, index=factuals.index)
