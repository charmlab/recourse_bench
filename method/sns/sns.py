from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.method_object import MethodObject
from method.sns.support import (
    TorchModelTypes,
    ensure_supported_target_model,
    min_l2_search,
    resolve_feature_groups,
    resolve_target_indices,
    sns_search,
    validate_counterfactuals,
)
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("sns")
class SnsMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        base_search: str = "min_l2",
        base_steps: int = 1000,
        base_step_size: float = 1e-2,
        base_lambda_start: float = 1e-2,
        base_lambda_growth: float = 2.0,
        base_lambda_max: float = 1e4,
        sns_eps: float = 0.1,
        sns_nb_iters: int = 200,
        sns_eps_iter: float = 1e-3,
        n_interpolations: int = 10,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, TorchModelTypes, "SnsMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._base_search = str(base_search).lower()
        self._base_steps = int(base_steps)
        self._base_step_size = float(base_step_size)
        self._base_lambda_start = float(base_lambda_start)
        self._base_lambda_growth = float(base_lambda_growth)
        self._base_lambda_max = float(base_lambda_max)
        self._sns_eps = float(sns_eps)
        self._sns_nb_iters = int(sns_nb_iters)
        self._sns_eps_iter = float(sns_eps_iter)
        self._n_interpolations = int(n_interpolations)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._base_search != "min_l2":
            raise ValueError("Only base_search='min_l2' is currently supported")
        if self._base_steps < 1:
            raise ValueError("base_steps must be >= 1")
        if self._base_step_size <= 0:
            raise ValueError("base_step_size must be > 0")
        if self._base_lambda_start <= 0:
            raise ValueError("base_lambda_start must be > 0")
        if self._base_lambda_growth <= 1:
            raise ValueError("base_lambda_growth must be > 1")
        if self._base_lambda_max < self._base_lambda_start:
            raise ValueError("base_lambda_max must be >= base_lambda_start")
        if self._sns_eps <= 0:
            raise ValueError("sns_eps must be > 0")
        if self._sns_nb_iters < 1:
            raise ValueError("sns_nb_iters must be >= 1")
        if self._sns_eps_iter <= 0:
            raise ValueError("sns_eps_iter must be > 0")
        if self._n_interpolations < 1:
            raise ValueError("n_interpolations must be >= 1")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for SnsMethod.fit()")
        with seed_context(self._seed):
            feature_groups = resolve_feature_groups(trainset)
            self._feature_names = list(feature_groups.feature_names)
            features = trainset.get(target=False).loc[:, self._feature_names].copy(deep=True)
            try:
                train_array = features.to_numpy(dtype=np.float64)
            except ValueError as error:
                raise ValueError(
                    "SnsMethod requires fully numeric input features"
                ) from error
            self._clamp = (float(np.min(train_array)), float(np.max(train_array)))
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        with seed_context(self._seed):
            original_prediction = (
                self._target_model.get_prediction(factuals, proba=True)
                .detach()
                .cpu()
                .numpy()
                .argmax(axis=1)
            )
            target_indices = resolve_target_indices(
                self._target_model,
                original_prediction,
                self._desired_class,
            )

            rows: list[np.ndarray] = []
            for row_position, (_, row) in enumerate(factuals.iterrows()):
                factual = row.to_numpy(dtype=np.float64)
                target_index = int(target_indices[row_position])
                base_candidate = min_l2_search(
                    self._target_model,
                    factual=factual,
                    target_index=target_index,
                    clamp=self._clamp,
                    steps=self._base_steps,
                    step_size=self._base_step_size,
                    lambda_start=self._base_lambda_start,
                    lambda_growth=self._base_lambda_growth,
                    lambda_max=self._base_lambda_max,
                )
                if base_candidate is None:
                    rows.append(np.full(len(self._feature_names), np.nan, dtype=np.float64))
                    continue
                refined = sns_search(
                    self._target_model,
                    counterfactual=base_candidate,
                    target_index=target_index,
                    clamp=self._clamp,
                    sns_eps=self._sns_eps,
                    sns_nb_iters=self._sns_nb_iters,
                    sns_eps_iter=self._sns_eps_iter,
                    n_interpolations=self._n_interpolations,
                )
                rows.append(refined)

        candidates = pd.DataFrame(rows, index=factuals.index, columns=self._feature_names)
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )
