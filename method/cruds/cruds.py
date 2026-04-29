from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.cruds.model import CRUDS
from method.cruds.support import (
    CrudsTargetModelAdapter,
    build_cruds_feature_context,
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


@register("cruds")
class CrudsMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = 1,
        lambda_param: float = 0.001,
        optimizer: str = "rmsprop",
        learning_rate: float = 0.008,
        max_iter: int = 2000,
        vae_hidden_dim: int = 64,
        vae_latent_dim: int = 6,
        vae_epochs: int = 5,
        vae_learning_rate: float = 1e-3,
        vae_batch_size: int = 32,
        **kwargs,
    ):
        del kwargs
        ensure_supported_target_model(target_model, "CrudsMethod")

        self._target_model = target_model
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = True
        self._is_trained = False
        self._desired_class = desired_class

        self._lambda_param = float(lambda_param)
        self._optimizer = str(optimizer).lower()
        self._learning_rate = float(learning_rate)
        self._max_iter = int(max_iter)
        self._vae_hidden_dim = int(vae_hidden_dim)
        self._vae_latent_dim = int(vae_latent_dim)
        self._vae_epochs = int(vae_epochs)
        self._vae_learning_rate = float(vae_learning_rate)
        self._vae_batch_size = int(vae_batch_size)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._lambda_param < 0:
            raise ValueError("lambda_param must be >= 0")
        if self._optimizer not in {"rmsprop", "adam"}:
            raise ValueError("optimizer must be 'rmsprop' or 'adam'")
        if self._learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self._max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self._vae_hidden_dim < 1:
            raise ValueError("vae_hidden_dim must be >= 1")
        if self._vae_latent_dim < 1:
            raise ValueError("vae_latent_dim must be >= 1")
        if self._vae_epochs < 1:
            raise ValueError("vae_epochs must be >= 1")
        if self._vae_learning_rate <= 0:
            raise ValueError("vae_learning_rate must be > 0")
        if self._vae_batch_size < 1:
            raise ValueError("vae_batch_size must be >= 1")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for CrudsMethod.fit()")

        with seed_context(self._seed):
            ensure_binary_classifier(self._target_model, "CrudsMethod")
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
                    "CrudsMethod requires finalized numeric input features"
                ) from error

            self._feature_context = build_cruds_feature_context(trainset)
            self._feature_names = list(self._feature_context.feature_names)
            self._adapter = CrudsTargetModelAdapter(
                target_model=self._target_model,
                feature_context=self._feature_context,
                trainset=trainset,
            )
            target_class = (
                1
                if self._desired_class is None
                else int(class_to_index[self._desired_class])
            )
            self._cruds = CRUDS(
                mlmodel=self._adapter,
                data_name=self._feature_context.dataset_name,
                target_class=target_class,
                lambda_param=self._lambda_param,
                optimizer=self._optimizer,
                learning_rate=self._learning_rate,
                max_iter=self._max_iter,
                vae_hidden_dim=self._vae_hidden_dim,
                vae_latent_dim=self._vae_latent_dim,
                vae_epochs=self._vae_epochs,
                vae_learning_rate=self._vae_learning_rate,
                vae_batch_size=self._vae_batch_size,
                device=self._device,
            )
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        with seed_context(self._seed):
            if self._desired_class is not None:
                counterfactuals = self._cruds.get_counterfactuals(factuals=factuals)
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
            original_target_class = int(self._cruds._target_class)

            try:
                for target_class in (0, 1):
                    subset = ordered_factuals.loc[target_labels == target_class]
                    if subset.empty:
                        continue
                    self._cruds._target_class = int(target_class)
                    subset_counterfactuals = self._cruds.get_counterfactuals(
                        factuals=subset
                    )
                    counterfactuals.loc[subset.index, self._feature_names] = (
                        subset_counterfactuals.reindex(
                            index=subset.index,
                            columns=self._feature_names,
                        )
                    )
            finally:
                self._cruds._target_class = original_target_class

        return counterfactuals.loc[:, self._feature_names].copy(deep=True)
