from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.feature_tweak.model import FeatureTweak
from method.feature_tweak.support import (
    FeatureTweakTargetModelAdapter,
    build_feature_tweak_context,
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


@register("feature_tweak")
class FeatureTweakMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = 1,
        eps: float = 0.1,
        **kwargs,
    ):
        del kwargs
        ensure_supported_target_model(target_model, "FeatureTweakMethod")

        self._target_model = target_model
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class
        self._eps = float(eps)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._eps <= 0:
            raise ValueError("eps must be > 0")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for FeatureTweakMethod.fit()")
        if not getattr(self._target_model, "_is_trained", False):
            raise RuntimeError("Target model must be trained before method.fit()")

        with seed_context(self._seed):
            ensure_binary_classifier(self._target_model, "FeatureTweakMethod")
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
                train_features.loc[:, :].to_numpy(dtype="float64")
            except ValueError as error:
                raise ValueError(
                    "FeatureTweakMethod requires finalized numeric input features"
                ) from error

            self._feature_context = build_feature_tweak_context(trainset)
            self._feature_names = list(self._feature_context.feature_names)
            self._adapter = FeatureTweakTargetModelAdapter(
                target_model=self._target_model,
                feature_context=self._feature_context,
            )
            if self._desired_class is None:
                self._feature_tweak_by_target = {
                    target_class: FeatureTweak(
                        mlmodel=self._adapter,
                        context=self._feature_context,
                        desired_class=target_class,
                        eps=self._eps,
                    )
                    for target_class in (0, 1)
                }
            else:
                self._desired_class_index = int(class_to_index[self._desired_class])
                self._feature_tweak = FeatureTweak(
                    mlmodel=self._adapter,
                    context=self._feature_context,
                    desired_class=self._desired_class_index,
                    eps=self._eps,
                )
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        with seed_context(self._seed):
            if self._desired_class is not None:
                counterfactuals = self._feature_tweak.get_counterfactuals(
                    factuals=factuals
                )
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

            for target_class, feature_tweak in self._feature_tweak_by_target.items():
                subset = ordered_factuals.loc[target_labels == target_class]
                if subset.empty:
                    continue
                subset_counterfactuals = feature_tweak.get_counterfactuals(
                    factuals=subset
                )
                counterfactuals.loc[subset.index, self._feature_names] = (
                    subset_counterfactuals.reindex(
                        index=subset.index,
                        columns=self._feature_names,
                    )
                )

        return counterfactuals.loc[:, self._feature_names].copy(deep=True)
