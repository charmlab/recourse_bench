from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.apas.support import (
    ApasContext,
    compute_delta_max,
    ensure_binary_mlp_target_model,
    generate_apas_counterfactual,
    prepare_apas_context,
    resolve_target_index,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("apas")
class ApasMethod(MethodObject):
    _context: ApasContext

    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        delta: float | None = None,
        alpha: float = 0.999,
        r: float = 0.995,
        delta_init: float = 1e-4,
        num_concretizations: int | None = None,
        eps_init: float = 0.01,
        eps_step: float = 0.2,
        max_iter: int = 100,
        big_m: float = 10000.0,
        use_biases: bool = True,
        rounding: int | None = 5,
        **kwargs,
    ):
        ensure_binary_mlp_target_model(target_model, "ApasMethod")

        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._delta = None if delta is None else float(delta)
        self._alpha = float(alpha)
        self._r = float(r)
        self._delta_init = float(delta_init)
        self._num_concretizations = (
            None if num_concretizations is None else int(num_concretizations)
        )
        self._eps_init = float(eps_init)
        self._eps_step = float(eps_step)
        self._max_iter = int(max_iter)
        self._big_m = float(big_m)
        self._use_biases = bool(use_biases)
        self._rounding = rounding if rounding is None else int(rounding)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._delta is not None and self._delta < 0:
            raise ValueError("delta must be >= 0")
        if not (0.0 < self._alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if not (0.0 < self._r < 1.0):
            raise ValueError("r must be in (0, 1)")
        if self._delta_init <= 0:
            raise ValueError("delta_init must be > 0")
        if self._num_concretizations is not None and self._num_concretizations < 1:
            raise ValueError("num_concretizations must be >= 1 when provided")
        if self._eps_init <= 0:
            raise ValueError("eps_init must be > 0")
        if self._eps_step < 0:
            raise ValueError("eps_step must be >= 0")
        if self._max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self._big_m <= 0:
            raise ValueError("big_m must be > 0")
        if self._rounding is not None and self._rounding < 0:
            raise ValueError("rounding must be >= 0 or None")

        class_to_index = self._target_model.get_class_to_index()
        if self._desired_class is not None and self._desired_class not in class_to_index:
            raise ValueError("desired_class is invalid for the trained target model")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ApasMethod.fit()")

        with seed_context(self._seed):
            self._context = prepare_apas_context(self._target_model, trainset)
            self._feature_names = list(self._context.feature_schema.feature_names)
            self._class_to_index = dict(self._context.class_to_index)
            self._is_trained = True

    def _postprocess_candidate(self, candidate: np.ndarray) -> np.ndarray:
        if self._rounding is None:
            return candidate.astype(np.float64, copy=False)
        return np.round(candidate.astype(np.float64, copy=False), self._rounding)

    def _row_seed(self, base_seed: int, row_index: int) -> int:
        return int(base_seed + 1009 * (row_index + 1))

    def estimate_delta_max(self, counterfactuals: pd.DataFrame) -> pd.Series:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if counterfactuals.isna().any(axis=None):
            raise ValueError("Counterfactuals cannot contain NaN when estimating delta_max")

        counterfactuals = counterfactuals.loc[:, self._feature_names].copy(deep=True)
        predictions = self._target_model.get_prediction(counterfactuals, proba=True)
        prediction_indices = predictions.detach().cpu().numpy().argmax(axis=1)

        values = []
        with seed_context(self._seed) as active_seed:
            for row_position, (_, row) in enumerate(counterfactuals.iterrows()):
                target_index = int(prediction_indices[row_position])
                values.append(
                    compute_delta_max(
                        network=self._context.target_networks[target_index],
                        candidate=row.to_numpy(dtype="float64"),
                        alpha=self._alpha,
                        r=self._r,
                        delta_init=self._delta_init,
                        num_concretizations=self._num_concretizations,
                        use_biases=self._use_biases,
                        seed=self._row_seed(active_seed, row_position),
                    )
                )
        return pd.Series(values, index=counterfactuals.index, name="delta_max")

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)

        original_prediction = self._target_model.get_prediction(factuals, proba=True)
        original_indices = original_prediction.detach().cpu().numpy().argmax(axis=1)

        rows: list[pd.Series] = []
        with seed_context(self._seed) as active_seed:
            for row_position, (_, row) in enumerate(factuals.iterrows()):
                original_index = int(original_indices[row_position])
                target_index = resolve_target_index(
                    class_to_index=self._class_to_index,
                    original_prediction=original_index,
                    desired_class=self._desired_class,
                )

                if self._desired_class is not None and original_index == target_index:
                    rows.append(pd.Series(row.copy(deep=True), index=self._feature_names))
                    continue

                network = self._context.target_networks[target_index]
                candidate = generate_apas_counterfactual(
                    schema=self._context.feature_schema,
                    network=network,
                    factual=row.to_numpy(dtype="float64"),
                    delta=self._delta,
                    alpha=self._alpha,
                    r=self._r,
                    delta_init=self._delta_init,
                    num_concretizations=self._num_concretizations,
                    eps_init=self._eps_init,
                    eps_step=self._eps_step,
                    max_iter=self._max_iter,
                    big_m=self._big_m,
                    use_biases=self._use_biases,
                    seed=self._row_seed(active_seed, row_position),
                )

                if candidate is None:
                    rows.append(
                        pd.Series(np.nan, index=self._feature_names, dtype="float64")
                    )
                    continue

                rows.append(
                    pd.Series(
                        self._postprocess_candidate(candidate),
                        index=self._feature_names,
                    )
                )

        candidates = pd.DataFrame(rows, index=factuals.index, columns=self._feature_names)
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
