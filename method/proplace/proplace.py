from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from tqdm import tqdm

from dataset.dataset_object import DatasetObject
from method.method_object import MethodObject
from method.proplace.support import (
    OptSolver,
    OptSolverRC4,
    SolverConfig,
    TorchModelTypes,
    build_inn,
    build_proplace_dataset,
    build_vertex_hull,
    ensure_supported_target_model,
    extract_scalar_network,
    predict_label_indices,
    resolve_target_index,
    validate_counterfactual_array,
)
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("proplace")
class ProplaceMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        delta: float = 0.03,
        k: int = 20,
        epsilon: float = 0.01,
        big_m: float = 1000.0,
        solver_output: bool = False,
        solver_time_limit: float | None = None,
        solver_threads: int | None = None,
        nn_leaf_size: int = 40,
        convex_hull_prune: bool = True,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, TorchModelTypes, "ProplaceMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._delta = float(delta)
        self._k = int(k)
        self._epsilon = float(epsilon)
        self._big_m = float(big_m)
        self._nn_leaf_size = int(nn_leaf_size)
        self._convex_hull_prune = bool(convex_hull_prune)
        self._solver_config = SolverConfig(
            output=bool(solver_output),
            time_limit=solver_time_limit,
            threads=solver_threads,
            seed=seed,
        )

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if not getattr(self._target_model, "_is_trained", False):
            raise RuntimeError("Target model must be trained before ProplaceMethod")
        if self._delta < 0.0:
            raise ValueError("delta must be >= 0")
        if self._k < 1:
            raise ValueError("k must be >= 1")
        if self._epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if self._big_m <= 0.0:
            raise ValueError("big_m must be > 0")
        if self._nn_leaf_size < 1:
            raise ValueError("nn_leaf_size must be >= 1")
        if solver_time_limit is not None and float(solver_time_limit) <= 0.0:
            raise ValueError("solver_time_limit must be > 0 when provided")
        if solver_threads is not None and int(solver_threads) < 1:
            raise ValueError("solver_threads must be >= 1 when provided")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ProplaceMethod.fit()")

        with seed_context(self._seed):
            self._dataset_spec, train_array = build_proplace_dataset(trainset)
            self._feature_names = list(self._dataset_spec.feature_names)
            self._train_features = trainset.get(target=False).loc[:, self._feature_names].copy(
                deep=True
            )
            self._train_array = train_array.astype(np.float64, copy=False)

            class_to_index = self._target_model.get_class_to_index()
            if len(class_to_index) != 2:
                raise ValueError("ProplaceMethod supports binary classification only")
            if self._desired_class is not None and self._desired_class not in class_to_index:
                raise ValueError("desired_class is invalid for the trained target model")

            self._prediction_by_index = predict_label_indices(
                self._target_model,
                self._train_features,
                self._feature_names,
            )
            self._train_points_by_target: dict[int, np.ndarray] = {}
            for target_index in range(2):
                self._train_points_by_target[target_index] = self._train_array[
                    self._prediction_by_index == target_index
                ]

            self._scalar_networks = {
                target_index: extract_scalar_network(self._target_model, target_index)
                for target_index in range(2)
            }
            self._interval_networks = {
                target_index: build_inn(self._scalar_networks[target_index], self._delta)
                for target_index in range(2)
            }
            self._robust_training_points: dict[int, np.ndarray] = {}
            self._robust_kdtrees: dict[int, KDTree | None] = {}
            self._is_trained = True

    def _build_nan_row(self) -> np.ndarray:
        return np.full(len(self._feature_names), np.nan, dtype=np.float64)

    def _get_robust_training_points(self, target_index: int) -> np.ndarray:
        if target_index in self._robust_training_points:
            return self._robust_training_points[target_index]

        candidates = self._train_points_by_target.get(target_index)
        if candidates is None or candidates.size == 0:
            robust_points = np.empty((0, len(self._feature_names)), dtype=np.float64)
            self._robust_training_points[target_index] = robust_points
            self._robust_kdtrees[target_index] = None
            return robust_points

        robust_rows: list[np.ndarray] = []
        interval_network = self._interval_networks[target_index]
        iterator = tqdm(
            candidates,
            total=int(candidates.shape[0]),
            desc=f"proplace-certify-target-{target_index}",
            leave=False,
        )
        for point in iterator:
            solver = OptSolver(
                dataset=self._dataset_spec,
                inn=interval_network,
                y_prime=1,
                x=point,
                mode=1,
                eps=self._epsilon,
                big_m=self._big_m,
                x_prime=point,
                solver_config=self._solver_config,
            )
            is_robust, _ = solver.compute_inn_bounds()
            if is_robust == 1:
                robust_rows.append(point.copy())

        if robust_rows:
            robust_points = np.vstack(robust_rows).astype(np.float64, copy=False)
            self._robust_kdtrees[target_index] = KDTree(
                robust_points, leaf_size=self._nn_leaf_size
            )
        else:
            robust_points = np.empty((0, len(self._feature_names)), dtype=np.float64)
            self._robust_kdtrees[target_index] = None

        self._robust_training_points[target_index] = robust_points
        return robust_points

    def _build_plausible_vertices(
        self,
        factual: np.ndarray,
        target_index: int,
    ) -> np.ndarray | None:
        robust_points = self._get_robust_training_points(target_index)
        if robust_points.shape[0] == 0:
            return None

        tree = self._robust_kdtrees[target_index]
        if tree is None:
            return None

        k = min(self._k, robust_points.shape[0])
        _, indices = tree.query(factual.reshape(1, -1), k=k)
        indices = np.asarray(indices).reshape(-1)
        neighbours = robust_points[indices]
        vertices = np.concatenate([factual.reshape(1, -1), neighbours], axis=0)
        return build_vertex_hull(vertices, prune=self._convex_hull_prune)

    def _solve_counterfactual_for_row(
        self,
        factual: np.ndarray,
        target_index: int,
    ) -> np.ndarray | None:
        vertices = self._build_plausible_vertices(factual, target_index)
        if vertices is None or vertices.shape[0] == 0:
            return None

        solver = OptSolverRC4(
            dataset=self._dataset_spec,
            inn=self._interval_networks[target_index],
            y_prime=1,
            x=factual,
            eps=self._epsilon,
            big_m=self._big_m,
            delta=self._delta,
            nns=vertices,
            solver_config=self._solver_config,
        )
        candidate = solver.run()
        if candidate is None:
            return None

        if not validate_counterfactual_array(
            self._target_model,
            self._feature_names,
            candidate,
            target_index,
        ):
            return None

        certificate = OptSolver(
            dataset=self._dataset_spec,
            inn=self._interval_networks[target_index],
            y_prime=1,
            x=factual,
            mode=1,
            eps=self._epsilon,
            big_m=self._big_m,
            x_prime=candidate,
            solver_config=self._solver_config,
        )
        is_robust, _ = certificate.compute_inn_bounds()
        if is_robust != 1:
            return None

        return candidate

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")
        if set(factuals.columns) != set(self._feature_names):
            raise ValueError("Input factuals must contain the fitted feature columns")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        with seed_context(self._seed):
            original_prediction = predict_label_indices(
                self._target_model,
                factuals,
                self._feature_names,
            )

            output_rows: list[np.ndarray] = []
            factual_iterator = tqdm(
                enumerate(factuals.iterrows()),
                total=int(factuals.shape[0]),
                desc="proplace-generate",
                leave=False,
            )
            for row_position, (_, row) in factual_iterator:
                factual = row.to_numpy(dtype=np.float64, copy=True)
                target_index = resolve_target_index(
                    self._target_model,
                    int(original_prediction[row_position]),
                    self._desired_class,
                )

                if (
                    self._desired_class is not None
                    and int(original_prediction[row_position]) == target_index
                ):
                    output_rows.append(factual)
                    continue

                candidate = self._solve_counterfactual_for_row(factual, target_index)
                if candidate is None:
                    output_rows.append(self._build_nan_row())
                else:
                    output_rows.append(candidate)

        return pd.DataFrame(output_rows, index=factuals.index, columns=self._feature_names)
