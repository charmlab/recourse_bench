from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from dataset.dataset_object import DatasetObject
from method.face.graph import (
    FaceSearchResult,
    build_base_graph,
    build_source_edge_weights,
    search_paths,
)
from method.face.support import (
    BlackBoxModelTypes,
    RecourseModelAdapter,
    ensure_supported_target_model,
    resolve_actionability_rules,
    resolve_feature_groups,
    resolve_target_indices,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("face")
class FaceMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        mode: str = "kde",
        epsilon: float = 0.5,
        k_neighbors: int = 5,
        prediction_threshold: float = 0.5,
        density_threshold: float | None = None,
        weight_function: str = "negative_log",
        kde_bandwidth: float = 0.5,
        kde_kernel: str = "tophat",
        subsample: float | int | None = None,
        store_top_k_paths: int = 5,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, BlackBoxModelTypes, "FaceMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class
        self._mode = self._resolve_mode(mode)
        self._epsilon = float(epsilon)
        self._k_neighbors = int(k_neighbors)
        self._prediction_threshold = float(prediction_threshold)
        self._density_threshold = (
            None if density_threshold is None else float(density_threshold)
        )
        self._weight_function = str(weight_function).lower()
        self._kde_bandwidth = float(kde_bandwidth)
        self._kde_kernel = str(kde_kernel).lower()
        self._subsample = self._resolve_subsample(subsample)
        self._store_top_k_paths = int(store_top_k_paths)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if self._k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if not (0.0 <= self._prediction_threshold <= 1.0):
            raise ValueError("prediction_threshold must satisfy 0 <= value <= 1")
        if self._density_threshold is not None and self._density_threshold < 0:
            raise ValueError("density_threshold must be >= 0 when provided")
        if self._kde_bandwidth <= 0:
            raise ValueError("kde_bandwidth must be > 0")
        if self._kde_kernel not in {
            "gaussian",
            "tophat",
            "epanechnikov",
            "exponential",
            "linear",
            "cosine",
        }:
            raise ValueError(f"Unsupported kde_kernel: {self._kde_kernel}")
        if self._store_top_k_paths < 1:
            raise ValueError("store_top_k_paths must be >= 1")

    @staticmethod
    def _resolve_mode(mode: str) -> str:
        mode_name = str(mode).lower()
        aliases = {
            "egraph": "epsilon",
            "epsilon-graph": "epsilon",
            "eps": "epsilon",
        }
        mode_name = aliases.get(mode_name, mode_name)
        if mode_name not in {"kde", "knn", "epsilon"}:
            raise ValueError("mode must be one of {'kde', 'knn', 'epsilon'}")
        return mode_name

    @staticmethod
    def _resolve_subsample(
        subsample: float | int | None,
    ) -> float | int | None:
        if subsample is None:
            return None
        if isinstance(subsample, int):
            if subsample < 1:
                raise ValueError("integer subsample must be >= 1")
            return int(subsample)
        if isinstance(subsample, float):
            if subsample <= 0 or subsample > 1:
                raise ValueError("float subsample must satisfy 0 < value <= 1")
            return float(subsample)
        raise TypeError("subsample must be int, float, or None")

    def _maybe_subsample_features(self, train_features: pd.DataFrame) -> pd.DataFrame:
        if self._subsample is None:
            return train_features.copy(deep=True)

        if isinstance(self._subsample, int):
            sample_size = min(int(self._subsample), train_features.shape[0])
        else:
            sample_size = int(np.ceil(train_features.shape[0] * float(self._subsample)))
            sample_size = max(1, min(sample_size, train_features.shape[0]))

        sampled_positions = np.random.permutation(train_features.shape[0])[:sample_size]
        return train_features.iloc[sampled_positions].copy(deep=True)

    def _score_density(self, values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return np.exp(self._density_estimator.score_samples(values))

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for FaceMethod.fit()")

        with seed_context(self._seed):
            self._feature_groups = resolve_feature_groups(trainset)
            self._feature_names = list(self._feature_groups.feature_names)
            self._constraints = resolve_actionability_rules(trainset)
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )

            train_features = trainset.get(target=False).copy(deep=True)
            self._train_features = self._maybe_subsample_features(train_features)
            self._train_values = self._train_features.to_numpy(dtype=np.float64)

            self._density_estimator = KernelDensity(
                bandwidth=self._kde_bandwidth,
                kernel=self._kde_kernel,
            )
            self._density_estimator.fit(self._train_values)

            probabilities = np.asarray(
                self._adapter.predict_proba(self._train_features),
                dtype=np.float64,
            )
            if probabilities.ndim != 2:
                raise ValueError("FACE requires predict_proba to return a 2D array")
            self._train_probabilities = probabilities
            self._train_prediction_indices = probabilities.argmax(axis=1)
            self._train_density = self._score_density(self._train_values)

            pairwise_delta = (
                self._train_values[:, None, :] - self._train_values[None, :, :]
            )
            self._pairwise_distances = np.linalg.norm(pairwise_delta, ord=2, axis=2)
            self._pairwise_feasibility = self._constraints.pairwise_feasibility(
                self._train_values
            )
            self._base_graph = build_base_graph(
                values=self._train_values,
                feasible_mask=self._pairwise_feasibility,
                distances=self._pairwise_distances,
                mode=self._mode,
                epsilon=self._epsilon,
                k_neighbors=self._k_neighbors,
                density_fn=self._score_density,
                weight_function=self._weight_function,
            )
            self._is_trained = True

    def _get_target_index(self, original_index: int) -> int:
        return int(
            resolve_target_indices(
                target_model=self._target_model,
                original_prediction=np.asarray([original_index], dtype=np.int64),
                desired_class=self._desired_class,
            )[0]
        )

    def _build_candidate_mask(self, target_index: int) -> np.ndarray:
        mask = self._train_prediction_indices == int(target_index)
        mask &= (
            self._train_probabilities[:, int(target_index)]
            >= self._prediction_threshold
        )
        if self._density_threshold is not None:
            mask &= self._train_density >= self._density_threshold
        return mask

    def _empty_result(self) -> FaceSearchResult:
        return FaceSearchResult(
            counterfactual=np.full(len(self._feature_names), np.nan, dtype=np.float64),
            path_found=False,
            path_cost=float("nan"),
            path_hops=float("nan"),
            endpoint_density=float("nan"),
            endpoint_confidence=float("nan"),
            top_paths=[],
        )

    def _search_single(self, factual: pd.Series) -> FaceSearchResult:
        source_value = factual.to_numpy(dtype=np.float64)
        source_df = factual.to_frame().T
        source_probabilities = np.asarray(
            self._adapter.predict_proba(source_df),
            dtype=np.float64,
        )
        original_index = int(source_probabilities.argmax(axis=1)[0])
        target_index = self._get_target_index(original_index)

        if self._desired_class is not None and original_index == target_index:
            endpoint_density = float(self._score_density(source_value)[0])
            endpoint_confidence = float(source_probabilities[0, target_index])
            return FaceSearchResult(
                counterfactual=source_value.copy(),
                path_found=True,
                path_cost=0.0,
                path_hops=0.0,
                endpoint_density=endpoint_density,
                endpoint_confidence=endpoint_confidence,
                top_paths=[
                    {
                        "cost": 0.0,
                        "hops": 0,
                        "endpoint_density": endpoint_density,
                        "endpoint_confidence": endpoint_confidence,
                        "trace": [
                            {
                                feature_name: float(feature_value)
                                for feature_name, feature_value in zip(
                                    self._feature_names,
                                    source_value,
                                )
                            }
                        ],
                    }
                ],
            )

        candidate_indices = np.flatnonzero(self._build_candidate_mask(target_index))
        if candidate_indices.size == 0:
            return self._empty_result()

        source_distances = np.linalg.norm(self._train_values - source_value, axis=1)
        source_feasibility = self._constraints.source_feasibility(
            source=source_value,
            destinations=self._train_values,
        )
        source_weights = build_source_edge_weights(
            source_value=source_value,
            reference_values=self._train_values,
            feasible_mask=source_feasibility,
            distances=source_distances,
            mode=self._mode,
            epsilon=self._epsilon,
            k_neighbors=self._k_neighbors,
            density_fn=self._score_density,
            weight_function=self._weight_function,
        )
        if not np.any(source_weights > 0.0):
            return self._empty_result()

        return search_paths(
            base_graph=self._base_graph,
            source_weights=source_weights,
            candidate_indices=candidate_indices,
            source_value=source_value,
            reference_values=self._train_values,
            feature_names=self._feature_names,
            endpoint_densities=self._train_density,
            endpoint_confidences=self._train_probabilities[:, target_index],
            store_top_k_paths=self._store_top_k_paths,
        )

    def _build_counterfactuals_and_metadata(
        self,
        factuals: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame | pd.Series]]:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        rows: list[pd.Series] = []
        path_found: list[bool] = []
        path_costs: list[float] = []
        path_hops: list[float] = []
        endpoint_densities: list[float] = []
        endpoint_confidences: list[float] = []
        top_paths: list[list[dict[str, float | int | list[dict[str, float]]]]] = []

        with seed_context(self._seed):
            for _, row in factuals.iterrows():
                result = self._search_single(row)
                rows.append(pd.Series(result.counterfactual, index=self._feature_names))
                path_found.append(bool(result.path_found))
                path_costs.append(float(result.path_cost))
                path_hops.append(float(result.path_hops))
                endpoint_densities.append(float(result.endpoint_density))
                endpoint_confidences.append(float(result.endpoint_confidence))
                top_paths.append(deepcopy(result.top_paths))

        candidates = pd.DataFrame(
            rows,
            index=factuals.index,
            columns=self._feature_names,
        )
        candidates = validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )

        failure_mask = candidates.isna().any(axis=1)
        for position, failed in enumerate(failure_mask.to_numpy()):
            if not failed:
                continue
            path_found[position] = False
            path_costs[position] = float("nan")
            path_hops[position] = float("nan")
            endpoint_densities[position] = float("nan")
            endpoint_confidences[position] = float("nan")
            top_paths[position] = []

        metadata: dict[str, pd.DataFrame | pd.Series] = {
            "face_path_found": pd.DataFrame(
                path_found,
                index=factuals.index,
                columns=["face_path_found"],
                dtype=bool,
            ),
            "face_path_cost": pd.DataFrame(
                path_costs,
                index=factuals.index,
                columns=["face_path_cost"],
                dtype=float,
            ),
            "face_path_hops": pd.DataFrame(
                path_hops,
                index=factuals.index,
                columns=["face_path_hops"],
                dtype=float,
            ),
            "face_endpoint_density": pd.DataFrame(
                endpoint_densities,
                index=factuals.index,
                columns=["face_endpoint_density"],
                dtype=float,
            ),
            "face_endpoint_confidence": pd.DataFrame(
                endpoint_confidences,
                index=factuals.index,
                columns=["face_endpoint_confidence"],
                dtype=float,
            ),
            "face_top_paths": pd.Series(
                top_paths,
                index=factuals.index,
                name="face_top_paths",
                dtype=object,
            ),
        }
        return candidates, metadata

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        candidates, _ = self._build_counterfactuals_and_metadata(factuals)
        return candidates

    def predict(self, testset: DatasetObject, batch_size: int = 20) -> DatasetObject:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if getattr(testset, "counterfactual", False):
            raise ValueError("testset must not already be marked as counterfactual")

        factuals = testset.get(target=False)
        counterfactual_batches: list[pd.DataFrame] = []
        metadata_batches: dict[str, list[pd.DataFrame | pd.Series]] = {
            "face_path_found": [],
            "face_path_cost": [],
            "face_path_hops": [],
            "face_endpoint_density": [],
            "face_endpoint_confidence": [],
            "face_top_paths": [],
        }

        for start in range(0, factuals.shape[0], batch_size):
            batch = factuals.iloc[start : start + batch_size]
            counterfactual_batch, metadata = self._build_counterfactuals_and_metadata(
                batch
            )
            counterfactual_batch = counterfactual_batch.reindex(
                index=batch.index,
                columns=batch.columns,
            )
            counterfactual_batches.append(counterfactual_batch)
            for key, value in metadata.items():
                metadata_batches[key].append(value)

        if counterfactual_batches:
            counterfactual_features = pd.concat(counterfactual_batches, axis=0)
            counterfactual_features = counterfactual_features.reindex(
                index=factuals.index
            )
        else:
            counterfactual_features = factuals.iloc[0:0].copy(deep=True)

        combined_metadata: dict[str, pd.DataFrame | pd.Series] = {}
        for key, values in metadata_batches.items():
            if not values:
                continue
            combined = pd.concat(values, axis=0)
            combined = combined.reindex(index=factuals.index)
            combined_metadata[key] = combined

        target_column = testset.target_column
        counterfactual_target = pd.DataFrame(
            -1.0,
            index=counterfactual_features.index,
            columns=[target_column],
        )
        counterfactual_df = pd.concat(
            [counterfactual_features, counterfactual_target],
            axis=1,
        )
        counterfactual_df = counterfactual_df.reindex(
            columns=testset.ordered_features()
        )

        output = testset.clone()
        output.update("counterfactual", True, df=counterfactual_df)
        for key, value in combined_metadata.items():
            output.update(key, value)

        if self._desired_class is not None:
            class_to_index = self._target_model.get_class_to_index()
            prediction = self._target_model.predict(testset, batch_size=batch_size)
            predicted_label = prediction.argmax(dim=1).cpu().numpy()
            evaluation_filter = pd.DataFrame(
                predicted_label != class_to_index[self._desired_class],
                index=counterfactual_df.index,
                columns=["evaluation_filter"],
                dtype=bool,
            )
            output.update("evaluation_filter", evaluation_filter)

        output.freeze()
        return output
