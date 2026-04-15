from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.diverse_dist.support import (
    DiverseDistModelAdapter,
    DiverseDistTrace,
    build_class_kdtrees,
    ensure_supported_target_model,
    generate_diverse_counterfactuals,
    normalize_candidate_selection,
    normalize_norm,
    to_numpy_features,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("diverse_dist")
class DiverseDistMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        total_cfs: int = 5,
        total_cfx: int | None = None,
        norm: int | float = 1,
        alpha: int = 50,
        beta: float = 0.5,
        gamma: float = 0.1,
        opt: bool = True,
        candidate_selection: str = "angle",
        **kwargs,
    ):
        ensure_supported_target_model(
            target_model,
            (MlpModel,),
            "DiverseDistMethod",
        )
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        if total_cfx is not None:
            total_cfs = total_cfx

        self._total_cfs = int(total_cfs)
        self._norm = normalize_norm(norm)
        self._alpha = int(alpha)
        self._beta = float(beta)
        self._gamma = float(gamma)
        self._opt = bool(opt)
        self._candidate_selection = normalize_candidate_selection(candidate_selection)
        self._last_explanation_sets: list[DiverseDistTrace] = []

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._total_cfs < 1:
            raise ValueError("total_cfs must be >= 1")
        if self._alpha < 1:
            raise ValueError("alpha must be >= 1")
        if self._beta < 0:
            raise ValueError("beta must be >= 0")
        if self._gamma <= 0:
            raise ValueError("gamma must be > 0")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for DiverseDistMethod.fit()")

        with seed_context(self._seed):
            class_to_index = self._target_model.get_class_to_index()
            if len(class_to_index) != 2:
                raise ValueError(
                    "DiverseDistMethod currently supports binary classification only"
                )
            if self._desired_class is not None and self._desired_class not in class_to_index:
                raise ValueError("desired_class is invalid for the trained target model")

            features = trainset.get(target=False)
            if features.isna().any(axis=1).any():
                raise ValueError("DiverseDistMethod requires non-NaN training features")

            self._feature_names = list(features.columns)
            self._train_array = to_numpy_features(features, self._feature_names)
            self._adapter = DiverseDistModelAdapter(
                self._target_model,
                self._feature_names,
            )
            predicted_labels = self._adapter.predict_label_indices(features)
            self._class_to_index = class_to_index
            self._kdtrees, self._class_points, self._class_indices = build_class_kdtrees(
                train_array=self._train_array,
                predicted_labels=predicted_labels,
                num_classes=len(class_to_index),
            )
            self._last_explanation_sets = []
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=1).any():
            raise ValueError("factuals must not contain NaN values")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        if factuals.empty:
            self._last_explanation_sets = []
            return factuals.copy(deep=True)

        original_prediction = self._adapter.predict_label_indices(factuals)
        rows: list[pd.Series] = []
        traces: list[DiverseDistTrace] = []

        for row_index, (_, row) in enumerate(factuals.iterrows()):
            factual_array = row.to_numpy(dtype=np.float32, copy=True)
            if self._desired_class is None:
                target_class_index = 1 - int(original_prediction[row_index])
            else:
                target_class_index = int(self._class_to_index[self._desired_class])

            if int(original_prediction[row_index]) == target_class_index:
                rows.append(
                    pd.Series(np.nan, index=self._feature_names, dtype="float64")
                )
                traces.append(
                    DiverseDistTrace(
                        status="already_desired_class",
                        target_class_index=int(target_class_index),
                        original_class_index=int(original_prediction[row_index]),
                        candidate_indices=[],
                        candidate_distances=[],
                        selected_candidate_indices=[],
                        selected_candidate_distances=[],
                        counterfactuals=[],
                        chosen_counterfactual=None,
                    )
                )
                continue

            _, trace = generate_diverse_counterfactuals(
                model_adapter=self._adapter,
                factual=factual_array,
                original_class_index=int(original_prediction[row_index]),
                target_class_index=int(target_class_index),
                alpha=self._alpha,
                total_cfs=self._total_cfs,
                beta=self._beta,
                gamma=self._gamma,
                norm=self._norm,
                candidate_selection=self._candidate_selection,
                opt=self._opt,
                kdtrees=self._kdtrees,
                class_points=self._class_points,
                class_indices=self._class_indices,
            )
            traces.append(trace)
            if trace.chosen_counterfactual is None:
                rows.append(
                    pd.Series(np.nan, index=self._feature_names, dtype="float64")
                )
            else:
                rows.append(
                    pd.Series(
                        trace.chosen_counterfactual.astype(np.float64, copy=False),
                        index=self._feature_names,
                    )
                )
        candidates = pd.DataFrame(
            rows,
            index=factuals.index,
            columns=self._feature_names,
        )
        validated = validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )

        success_mask = ~validated.isna().any(axis=1)
        for trace, success in zip(traces, success_mask.tolist()):
            if trace.status == "success" and not success:
                trace.status = "validated_failure"

        self._last_explanation_sets = traces
        return validated
