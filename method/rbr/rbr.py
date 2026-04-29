from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from dataset.dataset_object import DatasetObject
from method.method_object import MethodObject
from method.rbr.library.rbr_loss import robust_bayesian_recourse
from method.rbr.support import (
    BlackBoxModelTypes,
    RecourseModelAdapter,
    apply_onehot_constraints,
    ensure_supported_target_model,
    resolve_onehot_feature_indices,
    resolve_target_indices,
    validate_counterfactuals,
)
from model.model_object import ModelObject
from model.randomforest.randomforest import RandomForestModel
from utils.registry import register
from utils.seed import seed_context


def _compute_max_l2_distance(
    values: np.ndarray,
    device: str,
    chunk_size: int = 256,
) -> float:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("RBR max-distance computation requires a 2D array")
    if array.shape[0] < 2:
        return 0.0

    tensor = torch.tensor(array, dtype=torch.float32, device=device)
    max_distance = 0.0
    for start in range(0, tensor.shape[0], chunk_size):
        batch = tensor[start : start + chunk_size]
        distance = torch.cdist(batch, tensor, p=2)
        batch_max = float(distance.max().item())
        if batch_max > max_distance:
            max_distance = batch_max
    return max_distance


@register("rbr")
class RbrMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        num_samples: int = 200,
        perturb_radius: float = 0.2,
        delta_plus: float = 1.0,
        sigma: float = 1.0,
        epsilon_op: float = 0.5,
        epsilon_pe: float = 1.0,
        max_iter: int = 1000,
        clamp: bool = False,
        enforce_encoding: bool = False,
        random_state: int = 42,
        show_progress: bool = False,
        progress_desc: str | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, BlackBoxModelTypes, "RbrMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._num_samples = int(num_samples)
        self._perturb_radius = float(perturb_radius)
        self._delta_plus = float(delta_plus)
        self._sigma = float(sigma)
        self._epsilon_op = float(epsilon_op)
        self._epsilon_pe = float(epsilon_pe)
        self._max_iter = int(max_iter)
        self._clamp = bool(clamp)
        self._enforce_encoding = bool(enforce_encoding)
        self._random_state = int(random_state)
        self._show_progress = bool(show_progress)
        self._progress_desc = (
            str(progress_desc) if progress_desc is not None else "rbr-cf"
        )
        self._verbose = bool(verbose)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if isinstance(target_model, RandomForestModel) and self._device != "cpu":
            raise ValueError("RbrMethod with RandomForestModel requires device='cpu'")
        if self._num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if self._perturb_radius <= 0:
            raise ValueError("perturb_radius must be > 0")
        if self._delta_plus < 0:
            raise ValueError("delta_plus must be >= 0")
        if self._sigma <= 0:
            raise ValueError("sigma must be > 0")
        if self._epsilon_op < 0:
            raise ValueError("epsilon_op must be >= 0")
        if self._epsilon_pe < 0:
            raise ValueError("epsilon_pe must be >= 0")
        if self._max_iter < 1:
            raise ValueError("max_iter must be >= 1")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for RbrMethod.fit()")

        with seed_context(self._seed):
            features = trainset.get(target=False)
            try:
                train_data = features.to_numpy(dtype="float32")
            except ValueError as error:
                raise ValueError(
                    "RbrMethod requires fully numeric input features"
                ) from error

            if np.isnan(train_data).any():
                raise ValueError("RbrMethod trainset features cannot contain NaN")

            self._feature_names = list(features.columns)
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )
            self._train_data = train_data
            self._train_t = torch.tensor(
                train_data,
                dtype=torch.float32,
                device=self._device,
            )
            self._train_label = torch.tensor(
                self._adapter.predict_label_indices(self._train_t),
                dtype=torch.long,
                device=self._device,
            )
            self._onehot_feature_indices = resolve_onehot_feature_indices(
                trainset,
                self._feature_names,
            )
            self._train_max_distance = _compute_max_l2_distance(
                train_data,
                device=self._device,
            )

            self._class_to_index = self._target_model.get_class_to_index()
            if len(self._class_to_index) != 2:
                raise ValueError(
                    "RbrMethod currently supports binary classification only"
                )
            if (
                self._desired_class is not None
                and self._desired_class not in self._class_to_index
            ):
                raise ValueError(
                    "desired_class is invalid for the trained target model"
                )

            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")
        if list(factuals.columns) != self._feature_names:
            factuals = factuals.loc[:, self._feature_names].copy(deep=True)

        original_prediction = self._adapter.predict_label_indices(factuals)
        target_indices = resolve_target_indices(
            self._target_model,
            original_prediction,
            self._desired_class,
        )

        rows: list[pd.Series] = []
        with seed_context(self._seed):
            factual_iterator = tqdm(
                factuals.iterrows(),
                total=factuals.shape[0],
                desc=self._progress_desc,
                unit="cf",
                leave=False,
                disable=not self._show_progress,
            )
            for row_index, (_, row) in enumerate(factual_iterator):
                factual = row.to_numpy(dtype="float32", copy=True)
                original_index = int(original_prediction[row_index])
                target_index = int(target_indices[row_index])

                if self._desired_class is not None and original_index == target_index:
                    rows.append(pd.Series(factual.copy(), index=self._feature_names))
                    continue

                candidate = robust_bayesian_recourse(
                    raw_model=self._adapter,
                    x0=factual,
                    y_target=target_index,
                    cat_features_indices=(
                        self._onehot_feature_indices if self._enforce_encoding else None
                    ),
                    train_data=self._train_data,
                    train_t=self._train_t,
                    train_label=self._train_label,
                    num_samples=self._num_samples,
                    perturb_radius=self._perturb_radius * self._train_max_distance,
                    delta_plus=self._delta_plus,
                    sigma=self._sigma,
                    epsilon_op=self._epsilon_op,
                    epsilon_pe=self._epsilon_pe,
                    max_iter=self._max_iter,
                    dev=self._device,
                    random_state=self._random_state,
                    verbose=self._verbose,
                )

                if not np.all(np.isfinite(candidate)):
                    rows.append(
                        pd.Series(np.nan, index=self._feature_names, dtype="float64")
                    )
                    continue

                if self._clamp:
                    candidate = np.clip(candidate, 0.0, 1.0)
                if self._enforce_encoding:
                    candidate = apply_onehot_constraints(
                        candidate,
                        self._onehot_feature_indices,
                    )
                rows.append(pd.Series(candidate, index=self._feature_names))

        candidates = pd.DataFrame(
            rows, index=factuals.index, columns=self._feature_names
        )
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )
