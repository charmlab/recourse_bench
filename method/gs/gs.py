from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from dataset.dataset_object import DatasetObject
from method.gs.search import growing_spheres_search
from method.gs.support import (
    BlackBoxModelTypes,
    RecourseModelAdapter,
    ensure_supported_target_model,
    resolve_feature_groups,
    resolve_target_indices,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("gs")
class GsMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        n_search_samples: int = 1000,
        p_norm: int = 2,
        eta: float | None = None,
        step: float = 0.2,
        max_iter: int = 1000,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, BlackBoxModelTypes, "GsMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class
        self._n_search_samples = int(n_search_samples)
        self._p_norm = int(p_norm)
        resolved_eta = step if eta is None else eta
        self._eta = float(resolved_eta)
        self._max_iter = int(max_iter)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._n_search_samples < 1:
            raise ValueError("n_search_samples must be >= 1")
        if self._p_norm not in {1, 2}:
            raise ValueError("p_norm must be either 1 or 2")
        if self._eta <= 0:
            raise ValueError("eta/step must be > 0")
        if self._max_iter < 1:
            raise ValueError("max_iter must be >= 1")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for GsMethod.fit()")

        with seed_context(self._seed):
            feature_groups = resolve_feature_groups(trainset)
            self._feature_names = list(feature_groups.feature_names)
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )
            self._keys_mutable = list(feature_groups.mutable)
            self._keys_immutable = list(feature_groups.immutable)
            self._continuous = list(feature_groups.continuous)
            self._categorical = list(feature_groups.categorical)
            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        rows = []
        with seed_context(self._seed):
            original_prediction = self._adapter.predict_label_indices(factuals)
            target_prediction = resolve_target_indices(
                target_model=self._target_model,
                original_prediction=original_prediction,
                desired_class=self._desired_class,
            )
            iterator = zip(factuals.iterrows(), target_prediction, strict=True)
            for (index, row), target_label in tqdm(
                iterator,
                total=factuals.shape[0],
                desc="gs-search",
                leave=False,
            ):
                rows.append(
                    growing_spheres_search(
                        row,
                        self._keys_mutable,
                        self._keys_immutable,
                        self._continuous,
                        self._categorical,
                        self._feature_names,
                        self._adapter,
                        target_label=int(target_label),
                        n_search_samples=self._n_search_samples,
                        p_norm=self._p_norm,
                        eta=self._eta,
                        max_iter=self._max_iter,
                    )
                )

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )
