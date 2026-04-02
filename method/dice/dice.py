from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from method.dice.search import (
    DiceSearchConfig,
    DiceSearchMetadata,
    optimize_diverse_counterfactuals,
)
from method.dice.support import (
    RecourseModelAdapter,
    compute_feature_bounds,
    compute_feature_weights,
    compute_sparsity_thresholds,
    ensure_binary_classifier,
    ensure_model_supports_gradients,
    resolve_binary_feature_value_map,
    resolve_categorical_groups,
    resolve_feature_groups,
    resolve_target_indices,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from preprocess.preprocess_utils import resolve_feature_metadata
from utils.registry import register
from utils.seed import seed_context


@register("dice")
class DiceMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        num: int = 1,
        total_cfs: int | None = None,
        algorithm: str = "DiverseCF",
        yloss_type: str = "hinge_loss",
        diversity_loss_type: str = "dpp_style:inverse_dist",
        feature_weights: str | Mapping[str, float] = "inverse_mad",
        proximity_weight: float = 0.5,
        diversity_weight: float = 1.0,
        categorical_penalty: float = 0.1,
        optimizer: str = "adam",
        learning_rate: float = 0.05,
        min_iter: int = 500,
        max_iter: int = 5000,
        project_iter: int = 0,
        loss_diff_thres: float = 1e-5,
        loss_converge_maxiter: int = 2,
        verbose: bool = False,
        init_near_query_instance: bool = False,
        tie_random: bool = False,
        stopping_threshold: float = 0.5,
        posthoc_sparsity_param: float = 0.1,
        posthoc_sparsity_algorithm: str = "linear",
        respect_mutability: bool = False,
        **kwargs,
    ):
        ensure_model_supports_gradients(target_model, "DiceMethod")

        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = True
        self._is_trained = False
        self._desired_class = desired_class

        resolved_total_cfs = total_cfs if total_cfs is not None else num
        self._total_cfs = int(resolved_total_cfs)
        self._algorithm = str(algorithm).lower()
        self._yloss_type = str(yloss_type)
        self._diversity_loss_type = str(diversity_loss_type)
        self._feature_weights = feature_weights
        self._proximity_weight = float(proximity_weight)
        self._diversity_weight = float(diversity_weight)
        self._categorical_penalty = float(categorical_penalty)
        self._optimizer = str(optimizer)
        self._learning_rate = float(learning_rate)
        self._min_iter = int(min_iter)
        self._max_iter = int(max_iter)
        self._project_iter = int(project_iter)
        self._loss_diff_thres = float(loss_diff_thres)
        self._loss_converge_maxiter = int(loss_converge_maxiter)
        self._verbose = bool(verbose)
        self._init_near_query_instance = bool(init_near_query_instance)
        self._tie_random = bool(tie_random)
        self._stopping_threshold = float(stopping_threshold)
        self._posthoc_sparsity_param = float(posthoc_sparsity_param)
        self._posthoc_sparsity_algorithm = str(posthoc_sparsity_algorithm).lower()
        self._respect_mutability = bool(respect_mutability)

        if self._total_cfs < 1:
            raise ValueError("total_cfs/num must be >= 1")
        if self._algorithm not in {"diversecf", "randominitcf"}:
            raise ValueError(
                "DiceMethod supports algorithm='DiverseCF' or 'RandomInitCF' only"
            )
        if self._learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self._min_iter < 1:
            raise ValueError("min_iter must be >= 1")
        if self._max_iter < self._min_iter:
            raise ValueError("max_iter must be >= min_iter")
        if self._project_iter < 0:
            raise ValueError("project_iter must be >= 0")
        if self._loss_converge_maxiter < 1:
            raise ValueError("loss_converge_maxiter must be >= 1")
        if not 0.0 <= self._stopping_threshold <= 1.0:
            raise ValueError("stopping_threshold must lie in [0, 1]")
        if not 0.0 <= self._posthoc_sparsity_param <= 1.0:
            raise ValueError("posthoc_sparsity_param must lie in [0, 1]")
        if self._posthoc_sparsity_algorithm not in {"linear", "binary"}:
            raise ValueError("posthoc_sparsity_algorithm must be 'linear' or 'binary'")
        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for DiceMethod.fit()")

        with seed_context(self._seed):
            ensure_binary_classifier(self._target_model, "DiceMethod")

            feature_groups = resolve_feature_groups(trainset)
            self._feature_names = list(feature_groups.feature_names)
            self._feature_names_tuple = tuple(self._feature_names)
            self._adapter = RecourseModelAdapter(
                self._target_model, self._feature_names
            )

            train_features = (
                trainset.get(target=False).loc[:, self._feature_names].copy(deep=True)
            )
            feature_type, _, _ = resolve_feature_metadata(trainset)
            categorical_groups = resolve_categorical_groups(
                trainset, self._feature_names
            )
            binary_feature_value_map = resolve_binary_feature_value_map(
                train_features=train_features,
                feature_names=self._feature_names,
                feature_type=feature_type,
                categorical_groups=categorical_groups,
            )
            lower_bounds, upper_bounds = compute_feature_bounds(
                train_features=train_features,
                feature_names=self._feature_names,
            )
            weights = compute_feature_weights(
                train_features=train_features,
                feature_names=self._feature_names,
                continuous_feature_names=feature_groups.continuous,
                feature_weights=self._feature_weights,
            )
            sparsity_thresholds = compute_sparsity_thresholds(
                train_features=train_features,
                continuous_feature_names=feature_groups.continuous,
                quantile=self._posthoc_sparsity_param,
            )
            sparsity_order = tuple(
                self._feature_names.index(feature_name)
                for feature_name, _ in sorted(
                    sparsity_thresholds.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if feature_name in self._feature_names
            )
            continuous_indices = tuple(
                self._feature_names.index(feature_name)
                for feature_name in feature_groups.continuous
            )

            self._continuous_feature_names = tuple(feature_groups.continuous)
            self._continuous_indices = continuous_indices
            self._categorical_groups = tuple(categorical_groups)
            self._binary_feature_value_map = dict(binary_feature_value_map)
            self._mutable_mask_array = feature_groups.mutable_mask.copy()
            self._feature_weights_array = weights.copy()
            self._search_metadata = DiceSearchMetadata(
                feature_names=self._feature_names_tuple,
                continuous_indices=continuous_indices,
                continuous_features=self._continuous_feature_names,
                categorical_groups=self._categorical_groups,
                binary_feature_value_map=self._binary_feature_value_map,
                lower_bounds=torch.as_tensor(
                    lower_bounds,
                    dtype=torch.float32,
                    device=self._device,
                ),
                upper_bounds=torch.as_tensor(
                    upper_bounds,
                    dtype=torch.float32,
                    device=self._device,
                ),
                feature_weights=torch.as_tensor(
                    weights,
                    dtype=torch.float32,
                    device=self._device,
                ),
                mutable_mask=torch.as_tensor(
                    feature_groups.mutable_mask,
                    dtype=torch.bool,
                    device=self._device,
                ),
                sparsity_thresholds=sparsity_thresholds,
                sparsity_order=sparsity_order,
                device=torch.device(self._device),
            )
            self._is_trained = True

    def get_diverse_counterfactuals(
        self,
        factual: pd.DataFrame,
        total_cfs: int | None = None,
    ) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")

        query = factual.loc[:, self._feature_names].copy(deep=True)
        if query.shape[0] != 1:
            raise ValueError(
                "get_diverse_counterfactuals() expects exactly one factual row"
            )
        if query.isna().any(axis=1).any():
            raise ValueError("factuals cannot contain NaN")

        original_prediction = self._adapter.predict_label_indices(query)
        target_index = int(
            resolve_target_indices(
                target_model=self._target_model,
                original_prediction=original_prediction,
                desired_class=self._desired_class,
            )[0]
        )

        resolved_total_cfs = self._total_cfs if total_cfs is None else int(total_cfs)
        if resolved_total_cfs < 1:
            raise ValueError("total_cfs must be >= 1")

        search_config = DiceSearchConfig(
            total_cfs=resolved_total_cfs,
            algorithm=self._algorithm,
            yloss_type=self._yloss_type,
            diversity_loss_type=self._diversity_loss_type,
            proximity_weight=self._proximity_weight,
            diversity_weight=self._diversity_weight,
            categorical_penalty=self._categorical_penalty,
            optimizer=self._optimizer,
            learning_rate=self._learning_rate,
            min_iter=self._min_iter,
            max_iter=self._max_iter,
            project_iter=self._project_iter,
            loss_diff_thres=self._loss_diff_thres,
            loss_converge_maxiter=self._loss_converge_maxiter,
            init_near_query_instance=self._init_near_query_instance,
            tie_random=self._tie_random,
            stopping_threshold=self._stopping_threshold,
            posthoc_sparsity_param=self._posthoc_sparsity_param,
            posthoc_sparsity_algorithm=self._posthoc_sparsity_algorithm,
            respect_mutability=self._respect_mutability,
            verbose=self._verbose,
        )

        with seed_context(self._seed):
            candidates = optimize_diverse_counterfactuals(
                query_instance=query.iloc[0].to_numpy(dtype=np.float32),
                target_index=target_index,
                adapter=self._adapter,
                metadata=self._search_metadata,
                config=search_config,
            )

        if candidates.empty:
            return candidates.reindex(columns=self._feature_names)

        candidate_prediction = self._adapter.predict_label_indices(candidates)
        valid_mask = candidate_prediction.astype(np.int64, copy=False) == target_index
        filtered = candidates.loc[valid_mask].copy(deep=True)
        if filtered.empty:
            return filtered.reindex(columns=self._feature_names)

        filtered = filtered.reindex(columns=self._feature_names).drop_duplicates(
            ignore_index=True
        )
        return filtered

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=1).any():
            raise ValueError("factuals cannot contain NaN")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        selected_rows: list[pd.Series] = []

        with seed_context(self._seed):
            for _, row in factuals.iterrows():
                query = row.to_frame().T
                diverse_counterfactuals = self.get_diverse_counterfactuals(query)
                if diverse_counterfactuals.empty:
                    selected_rows.append(pd.Series(np.nan, index=self._feature_names))
                    continue

                weighted_diff = np.abs(
                    diverse_counterfactuals.to_numpy(dtype=np.float32)
                    - query.to_numpy(dtype=np.float32)
                ) * self._feature_weights_array.reshape(1, -1)
                best_index = int(weighted_diff.sum(axis=1).argmin())
                selected_rows.append(
                    diverse_counterfactuals.iloc[best_index].copy(deep=True)
                )

        candidates = pd.DataFrame(
            selected_rows,
            index=factuals.index,
            columns=self._feature_names,
        )
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )
