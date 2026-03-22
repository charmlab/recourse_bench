from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.method_object import MethodObject
from method.trex.support import (
    TorchModelTypes,
    TreXCounterfactualTorch,
    TreXLogitModel,
    ensure_supported_target_model,
    resolve_target_indices,
    validate_counterfactuals,
)
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("trex")
class TrexMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        target_class: int | str | None = None,
        norm: int = 2,
        cf_confidence: float = 0.5,
        cf_steps: int = 60,
        cf_step_size: float = 0.02,
        tau: float = 0.75,
        k: int = 1000,
        sigma: float = 0.1,
        trex_step_size: float = 0.01,
        trex_max_steps: int = 100,
        trex_epsilon: float = 1.0,
        trex_p: float = 2,
        batch_size: int = 1,
        clamp: bool | tuple[float | list[float], float | list[float]] | None = None,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, TorchModelTypes, "TrexMethod")

        self._target_model = target_model
        self._seed = seed
        self._device = str(device).lower()
        self._need_grad = True
        self._is_trained = False
        self._desired_class = (
            desired_class if desired_class is not None else target_class
        )

        self._norm = int(norm)
        self._cf_confidence = float(cf_confidence)
        self._cf_steps = int(cf_steps)
        self._cf_step_size = float(cf_step_size)
        self._tau = float(tau)
        self._k = int(k)
        self._sigma = float(sigma)
        self._trex_step_size = float(trex_step_size)
        self._trex_max_steps = int(trex_max_steps)
        self._trex_epsilon = float(trex_epsilon)
        self._trex_p = trex_p
        self._batch_size = int(batch_size)
        self._clamp_config = clamp

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if not self._target_model._need_grad:
            raise ValueError("TrexMethod requires a gradient-enabled target model")
        if self._norm not in {1, 2}:
            raise ValueError("norm must be 1 or 2")
        if self._cf_steps < 1:
            raise ValueError("cf_steps must be >= 1")
        if self._cf_step_size <= 0:
            raise ValueError("cf_step_size must be > 0")
        if self._k < 1:
            raise ValueError("k must be >= 1")
        if self._sigma < 0:
            raise ValueError("sigma must be >= 0")
        if self._trex_step_size <= 0:
            raise ValueError("trex_step_size must be > 0")
        if self._trex_max_steps < 1:
            raise ValueError("trex_max_steps must be >= 1")
        if self._trex_epsilon < 0:
            raise ValueError("trex_epsilon must be >= 0")
        if self._batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    @staticmethod
    def _normalize_bound(
        bound: float | list[float] | np.ndarray,
        *,
        bound_name: str,
        expected_dim: int,
    ) -> float | np.ndarray:
        if np.isscalar(bound):
            return float(bound)

        array = np.asarray(bound, dtype=np.float32).reshape(-1)
        if array.size != expected_dim:
            raise ValueError(
                f"{bound_name} must be scalar or length {expected_dim}, "
                f"received length {array.size}"
            )
        return array

    def _resolve_clamp(
        self,
        features: pd.DataFrame,
    ) -> tuple[float | np.ndarray, float | np.ndarray] | None:
        if self._clamp_config is False:
            return None

        if self._clamp_config is None or self._clamp_config is True:
            return (
                features.min(axis=0).to_numpy(dtype=np.float32),
                features.max(axis=0).to_numpy(dtype=np.float32),
            )

        if (
            not isinstance(self._clamp_config, (tuple, list))
            or len(self._clamp_config) != 2
        ):
            raise TypeError("clamp must be None, bool, or a 2-item tuple/list")

        low, high = self._clamp_config
        expected_dim = features.shape[1]
        return (
            self._normalize_bound(low, bound_name="clamp low", expected_dim=expected_dim),
            self._normalize_bound(
                high, bound_name="clamp high", expected_dim=expected_dim
            ),
        )

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for TrexMethod.fit()")

        with seed_context(self._seed):
            features = trainset.get(target=False)
            self._feature_names = list(features.columns)
            self._class_to_index = self._target_model.get_class_to_index()
            if len(self._class_to_index) != 2:
                raise ValueError(
                    "TrexMethod currently supports binary classification only"
                )
            if (
                self._desired_class is not None
                and self._desired_class not in self._class_to_index
            ):
                raise ValueError("desired_class is invalid for the trained target model")

            self._clamp = self._resolve_clamp(features)
            self._generator = TreXCounterfactualTorch(
                model=TreXLogitModel(self._target_model),
                input_dim=features.shape[1],
                num_classes=len(self._class_to_index),
                clamp=self._clamp,
                norm=self._norm,
                cf_steps=self._cf_steps,
                cf_step_size=self._cf_step_size,
                cf_confidence=self._cf_confidence,
                tau=self._tau,
                k=self._k,
                sigma=self._sigma,
                trex_max_steps=self._trex_max_steps,
                trex_epsilon=self._trex_epsilon,
                trex_step_size=self._trex_step_size,
                trex_p=self._trex_p,
                batch_size=self._batch_size,
                device=self._device,
            )
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)

        with seed_context(self._seed):
            original_prediction_tensor = self._target_model.get_prediction(
                factuals, proba=True
            )
            original_prediction = (
                original_prediction_tensor.argmax(dim=1).detach().cpu().numpy()
            )
            target_prediction = resolve_target_indices(
                target_model=self._target_model,
                original_prediction=original_prediction,
                desired_class=self._desired_class,
            )

            rows: list[pd.Series] = []
            for row_index, (_, row) in enumerate(factuals.iterrows()):
                original_index = int(original_prediction[row_index])
                target_index = int(target_prediction[row_index])

                if (
                    self._desired_class is not None
                    and original_index == target_index
                ):
                    rows.append(pd.Series(row.copy(deep=True), index=self._feature_names))
                    continue

                x_np = row.to_numpy(dtype=np.float32).reshape(1, -1)
                x_cf, _, is_valid = self._generator.generate(
                    x_np=x_np,
                    target_labels=np.array([target_index], dtype=np.int64),
                    apply_trex=True,
                )

                if bool(is_valid[0]):
                    rows.append(pd.Series(x_cf[0], index=self._feature_names))
                else:
                    rows.append(pd.Series(np.nan, index=self._feature_names))

        candidates = pd.DataFrame(rows, index=factuals.index, columns=self._feature_names)
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )
