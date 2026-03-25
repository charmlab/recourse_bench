from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.cvas_proj.support import (
    BlackBoxModelTypes,
    RecourseModelAdapter,
    compute_max_distance,
    derive_row_seed,
    ensure_supported_target_model,
    find_boundary_point,
    fit_local_surrogate,
    resolve_projection_features,
    resolve_target_indices,
    sample_uniform_ball,
    select_nearest_training_points,
    solve_l1_projection,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("cvas_proj")
class CvasProjMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = 1,
        surrogate_method: str = "fr_rmpm",
        rho_neg: float = 10.0,
        rho_pos: float = 0.0,
        perturb_radius: float = 0.05,
        num_samples: int = 1000,
        num_boundary_neighbors: int = 10,
        line_search_steps: int = 100,
        threshold_shift: float = 0.0,
        projection_epsilon: float = 0.1,
        surrogate_solver: str = "CLARABEL",
        projection_solver: str = "HIGHS",
        **kwargs,
    ):
        ensure_supported_target_model(
            target_model,
            BlackBoxModelTypes,
            "CvasProjMethod",
        )
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._surrogate_method = str(surrogate_method).lower()
        self._rho_neg = float(rho_neg)
        self._rho_pos = float(rho_pos)
        self._perturb_radius = float(perturb_radius)
        self._num_samples = int(num_samples)
        self._num_boundary_neighbors = int(num_boundary_neighbors)
        self._line_search_steps = int(line_search_steps)
        self._threshold_shift = float(threshold_shift)
        self._projection_epsilon = float(projection_epsilon)
        self._surrogate_solver = str(surrogate_solver).upper()
        self._projection_solver = str(projection_solver).upper()

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._surrogate_method not in {"mpm", "quad_rmpm", "bw_rmpm", "fr_rmpm"}:
            raise ValueError(
                "surrogate_method must be one of ['mpm', 'quad_rmpm', "
                "'bw_rmpm', 'fr_rmpm']"
            )
        if self._rho_neg < 0 or self._rho_pos < 0:
            raise ValueError("rho_neg and rho_pos must be >= 0")
        if self._perturb_radius <= 0:
            raise ValueError("perturb_radius must be > 0")
        if self._num_samples < 2:
            raise ValueError("num_samples must be >= 2")
        if self._num_boundary_neighbors < 1:
            raise ValueError("num_boundary_neighbors must be >= 1")
        if self._line_search_steps < 2:
            raise ValueError("line_search_steps must be >= 2")
        if self._projection_epsilon < 0:
            raise ValueError("projection_epsilon must be >= 0")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for CvasProjMethod.fit()")

        with seed_context(self._seed):
            feature_df = trainset.get(target=False)
            if feature_df.isna().any(axis=None):
                raise ValueError("CvasProjMethod does not support NaN training features")

            try:
                self._train_data = feature_df.to_numpy(dtype=np.float64)
            except ValueError as error:
                raise ValueError(
                    "CvasProjMethod requires fully numeric input features"
                ) from error

            projection_features = resolve_projection_features(trainset)
            self._feature_names = list(projection_features.feature_names)
            self._boolean_feature_indices = list(
                projection_features.boolean_feature_indices
            )
            self._adapter = RecourseModelAdapter(self._target_model, self._feature_names)

            self._class_to_index = self._target_model.get_class_to_index()
            if len(self._class_to_index) != 2:
                raise ValueError(
                    "CvasProjMethod currently supports binary classification only"
                )
            if (
                self._desired_class is not None
                and self._desired_class not in self._class_to_index
            ):
                raise ValueError(
                    "desired_class is invalid for the trained target model"
                )

            self._train_labels = self._adapter.predict_label_indices(feature_df)
            self._max_distance = compute_max_distance(self._train_data)
            self._is_trained = True

    def _generate_single_counterfactual(
        self,
        x0: np.ndarray,
        target_index: int,
        row_seed: int | None,
    ) -> np.ndarray | None:
        prototypes = select_nearest_training_points(
            train_data=self._train_data,
            train_labels=self._train_labels,
            x0=x0,
            target_label=target_index,
            num_neighbors=self._num_boundary_neighbors,
        )
        if prototypes.shape[0] == 0:
            return None

        boundary_point = find_boundary_point(
            x0=x0,
            target_label=target_index,
            prototypes=prototypes,
            adapter=self._adapter,
            line_search_steps=self._line_search_steps,
        )
        if boundary_point is None:
            return None

        local_samples = sample_uniform_ball(
            center=boundary_point,
            radius=self._perturb_radius * self._max_distance,
            num_samples=self._num_samples,
            random_state=row_seed,
        )
        local_labels = self._adapter.predict_label_indices(local_samples)

        surrogate = fit_local_surrogate(
            X=local_samples,
            y=local_labels,
            method=self._surrogate_method,
            rho_neg=self._rho_neg,
            rho_pos=self._rho_pos,
            solver_name=self._surrogate_solver,
        )
        if surrogate is None:
            return None

        coef = np.asarray(surrogate.coef, dtype=np.float64).reshape(-1)
        intercept = float(surrogate.intercept)
        coef_norm = float(np.linalg.norm(coef, ord=2))
        if (not np.isfinite(coef_norm)) or coef_norm <= 0:
            return None

        coef = coef / coef_norm
        intercept = intercept / coef_norm
        sign = -1.0 if float(np.dot(coef, x0) + intercept) <= 0 else 1.0
        intercept = intercept + sign * self._threshold_shift

        candidate = solve_l1_projection(
            x0=x0,
            coef=coef,
            intercept=intercept,
            boolean_feature_indices=self._boolean_feature_indices,
            epsilon=self._projection_epsilon,
            solver_name=self._projection_solver,
        )
        if candidate is None or not np.all(np.isfinite(candidate)):
            return None
        return candidate

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        original_prediction = self._adapter.predict_label_indices(factuals)
        target_indices = resolve_target_indices(
            self._target_model,
            original_prediction=original_prediction,
            desired_class=self._desired_class,
        )

        rows: list[pd.Series] = []
        with seed_context(self._seed):
            for row_position, (row_index, row) in enumerate(factuals.iterrows()):
                x0 = row.to_numpy(dtype=np.float64, copy=True)
                if (
                    self._desired_class is not None
                    and int(original_prediction[row_position]) == int(target_indices[row_position])
                ):
                    rows.append(pd.Series(x0, index=self._feature_names))
                    continue

                candidate = self._generate_single_counterfactual(
                    x0=x0,
                    target_index=int(target_indices[row_position]),
                    row_seed=derive_row_seed(self._seed, row_index),
                )
                if candidate is None:
                    rows.append(
                        pd.Series(np.nan, index=self._feature_names, dtype=np.float64)
                    )
                else:
                    rows.append(pd.Series(candidate, index=self._feature_names))

        candidates = pd.DataFrame(rows, index=factuals.index, columns=self._feature_names)
        return validate_counterfactuals(
            self._target_model,
            factuals,
            candidates,
            desired_class=self._desired_class,
        )
