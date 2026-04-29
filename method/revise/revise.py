from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.method_object import MethodObject
from method.revise.model import Revise
from method.revise.support import (
    ReviseTargetModelAdapter,
    build_revise_feature_context,
    ensure_binary_classifier,
    ensure_supported_target_model,
)
from model.model_object import ModelObject
from model.model_utils import resolve_device
from utils.registry import register
from utils.seed import seed_context


def _as_label_array(predictions: object) -> np.ndarray:
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    return np.asarray(predictions, dtype=np.int64).reshape(-1)


@register("revise")
class ReviseMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = 1,
        lambda_: float = 0.5,
        optimizer: str = "adam",
        learning_rate: float = 0.1,
        max_iter: int = 1500,
        binary_cat_features: bool = True,
        vae_layers: list[int] | None = None,
        vae_train: bool = True,
        vae_lambda_reg: float = 1e-6,
        vae_epochs: int = 5,
        vae_learning_rate: float = 1e-3,
        vae_batch_size: int = 32,
        **kwargs,
    ):
        del kwargs
        ensure_supported_target_model(target_model, "ReviseMethod")

        self._target_model = target_model
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = True
        self._is_trained = False
        self._desired_class = desired_class

        self._lambda_param = float(lambda_)
        self._optimizer = str(optimizer).lower()
        self._learning_rate = float(learning_rate)
        self._max_iter = int(max_iter)
        self._binary_cat_features = bool(binary_cat_features)
        self._vae_layers = None if vae_layers is None else [int(x) for x in vae_layers]
        self._vae_train = bool(vae_train)
        self._vae_lambda_reg = float(vae_lambda_reg)
        self._vae_epochs = int(vae_epochs)
        self._vae_learning_rate = float(vae_learning_rate)
        self._vae_batch_size = int(vae_batch_size)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._lambda_param < 0:
            raise ValueError("lambda_ must be >= 0")
        if self._optimizer not in {"adam", "rmsprop"}:
            raise ValueError("optimizer must be 'adam' or 'rmsprop'")
        if self._learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self._max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self._vae_layers is not None and any(
            layer < 1 for layer in self._vae_layers
        ):
            raise ValueError("vae_layers entries must be >= 1")
        if self._vae_lambda_reg < 0:
            raise ValueError("vae_lambda_reg must be >= 0")
        if self._vae_epochs < 1:
            raise ValueError("vae_epochs must be >= 1")
        if self._vae_learning_rate <= 0:
            raise ValueError("vae_learning_rate must be > 0")
        if self._vae_batch_size < 1:
            raise ValueError("vae_batch_size must be >= 1")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ReviseMethod.fit()")
        if not getattr(self._target_model, "_is_trained", False):
            raise RuntimeError("Target model must be trained before method.fit()")

        with seed_context(self._seed):
            ensure_binary_classifier(self._target_model, "ReviseMethod")
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
                    "ReviseMethod requires finalized numeric input features"
                ) from error

            self._feature_context = build_revise_feature_context(trainset)
            self._feature_names = list(self._feature_context.feature_names)
            self._adapter = ReviseTargetModelAdapter(
                target_model=self._target_model,
                feature_context=self._feature_context,
                trainset=trainset,
            )

            mutable_dim = int(self._feature_context.mutable_mask.sum())
            hidden_layers = list(self._vae_layers or [512, 256, 8])
            self._target_class_by_index = {
                0: [1.0, 0.0],
                1: [0.0, 1.0],
            }
            desired_class_index = (
                1
                if self._desired_class is None
                else int(class_to_index[self._desired_class])
            )
            hyperparams = {
                "data_name": self._feature_context.dataset_name,
                "lambda": self._lambda_param,
                "optimizer": self._optimizer,
                "lr": self._learning_rate,
                "max_iter": self._max_iter,
                "target_class": list(self._target_class_by_index[desired_class_index]),
                "binary_cat_features": self._binary_cat_features,
                "vae_params": {
                    "layers": [mutable_dim] + hidden_layers,
                    "train": self._vae_train,
                    "lambda_reg": self._vae_lambda_reg,
                    "epochs": self._vae_epochs,
                    "lr": self._vae_learning_rate,
                    "batch_size": self._vae_batch_size,
                },
            }

            self._revise = Revise(
                mlmodel=self._adapter,
                hyperparams=hyperparams,
                device=self._device,
            )
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        with seed_context(self._seed):
            if self._desired_class is not None:
                counterfactuals = self._revise.get_counterfactuals(factuals=factuals)
                return counterfactuals.reindex(
                    index=factuals.index,
                    columns=factuals.columns,
                ).copy(deep=True)

            ordered_factuals = self._adapter.get_ordered_features(factuals).reindex(
                index=factuals.index,
                columns=self._feature_names,
            )
            if ordered_factuals.empty:
                return ordered_factuals.reindex(
                    index=factuals.index,
                    columns=factuals.columns,
                ).copy(deep=True)

            predicted_labels = _as_label_array(self._adapter.predict(ordered_factuals))
            target_labels = 1 - predicted_labels
            counterfactuals = pd.DataFrame(
                np.nan,
                index=factuals.index.copy(),
                columns=factuals.columns,
            )
            original_target_class = list(self._revise._target_class)
            original_desired_class = int(self._revise._desired_class)

            try:
                for target_class, target_vector in self._target_class_by_index.items():
                    subset = ordered_factuals.loc[target_labels == target_class]
                    if subset.empty:
                        continue
                    self._revise._target_class = list(target_vector)
                    self._revise._desired_class = int(target_class)
                    subset_counterfactuals = self._revise.get_counterfactuals(
                        factuals=subset
                    )
                    counterfactuals.loc[subset.index, factuals.columns] = (
                        subset_counterfactuals.reindex(
                            index=subset.index,
                            columns=factuals.columns,
                        )
                    )
            finally:
                self._revise._target_class = original_target_class
                self._revise._desired_class = original_desired_class

        return counterfactuals.reindex(
            index=factuals.index,
            columns=factuals.columns,
        ).copy(deep=True)
