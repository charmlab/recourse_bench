from __future__ import annotations

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
        if self._desired_class is None:
            raise ValueError("FeatureTweakMethod requires desired_class to be set")
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
            if self._desired_class not in class_to_index:
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
            self._desired_class_index = int(class_to_index[self._desired_class])
            self._adapter = FeatureTweakTargetModelAdapter(
                target_model=self._target_model,
                feature_context=self._feature_context,
            )
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
            counterfactuals = self._feature_tweak.get_counterfactuals(factuals=factuals)
        return counterfactuals.loc[:, self._feature_names].copy(deep=True)
