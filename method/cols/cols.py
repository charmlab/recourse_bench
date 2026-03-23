from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.cols.support import (
    DEFAULT_INVALID_COST,
    RecourseModelAdapter,
    RuntimeSearchContext,
    SupportedTargetModelTypes,
    build_runtime_context,
    build_search_schema,
    choose_l1_closest,
    compute_benefit_matrix,
    compute_candidate_cost_matrix,
    compute_emc,
    decode_feature_dataframe,
    encode_state_dataframe,
    ensure_supported_target_model,
    infer_base_state_spaces,
    is_ordered_feature,
    resolve_target_index,
    select_replacement_pairs,
    split_budget,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@dataclass
class _SearchArtifacts:
    state_set: pd.DataFrame
    encoded_set: pd.DataFrame
    cost_matrix: np.ndarray
    valid_mask: np.ndarray
    emc: float


@register("cols")
class ColsMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        num_cfs: int = 10,
        num_mcmc: int = 1000,
        budget: int = 5000,
        num_parallel_runs: int = 1,
        hamming_dist: int = 2,
        perturb_type: str = "all",
        init_type: str = "ham_2",
        iter_type: str = "best",
        alpha: float | None = None,
        variance: float = 0.001,
        invalid_cost: float = DEFAULT_INVALID_COST,
        **kwargs,
    ):
        ensure_supported_target_model(
            target_model,
            SupportedTargetModelTypes,
            "ColsMethod",
        )

        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._num_cfs = int(num_cfs)
        self._num_mcmc = int(num_mcmc)
        self._budget = int(budget)
        self._num_parallel_runs = int(num_parallel_runs)
        self._hamming_dist = int(hamming_dist)
        self._perturb_type = str(perturb_type).lower()
        self._init_type = str(init_type).lower()
        self._iter_type = str(iter_type).lower()
        self._alpha = None if alpha is None else float(alpha)
        self._variance = float(variance)
        self._invalid_cost = float(invalid_cost)

        self._last_counterfactual_sets: list[pd.DataFrame] = []
        self._last_counterfactual_validity: list[pd.Series] = []
        self._last_search_stats: list[dict[str, object]] = []
        self._query_count = 0

        if self._num_cfs < 1:
            raise ValueError("num_cfs must be >= 1")
        if self._num_mcmc < 1:
            raise ValueError("num_mcmc must be >= 1")
        if self._budget < 1:
            raise ValueError("budget must be >= 1")
        if self._num_parallel_runs < 1:
            raise ValueError("num_parallel_runs must be >= 1")
        if self._hamming_dist < 1:
            raise ValueError("hamming_dist must be >= 1")
        if self._variance <= 0:
            raise ValueError("variance must be > 0")
        if self._alpha is not None and not 0.0 <= self._alpha <= 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha <= 1")
        if self._iter_type not in {"best", "linear"}:
            raise ValueError("iter_type must be 'best' or 'linear'")
        if self._perturb_type not in {"one", "all"} and not self._perturb_type.startswith(
            "frac_"
        ):
            raise ValueError("perturb_type must be 'one', 'all', or 'frac_x'")
        if self._init_type not in {"random", "org"} and not self._init_type.startswith(
            "ham_"
        ):
            raise ValueError("init_type must be 'random', 'org', or 'ham_x'")
        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ColsMethod.fit()")

        with seed_context(self._seed):
            features = trainset.get(target=False)
            self._feature_names = list(features.columns)
            self._schema = build_search_schema(trainset)
            self._decoded_training = decode_feature_dataframe(features, self._schema)
            self._base_state_spaces = infer_base_state_spaces(
                self._decoded_training,
                self._schema,
            )
            self._adapter = RecourseModelAdapter(self._target_model, self._feature_names)

            class_to_index = self._target_model.get_class_to_index()
            if (
                self._desired_class is not None
                and self._desired_class not in class_to_index
            ):
                raise ValueError("desired_class is invalid for the trained target model")

            self._is_trained = True

    def _predict_label_indices(self, encoded_set: pd.DataFrame) -> np.ndarray:
        self._query_count += int(encoded_set.shape[0])
        return self._adapter.predict_label_indices(encoded_set)

    def _active_features(self, runtime_context: RuntimeSearchContext) -> list[str]:
        active = [
            source_name
            for source_name in runtime_context.schema.source_features
            if len(runtime_context.feature_contexts[source_name].search_states) > 1
        ]
        if active:
            return active
        return list(runtime_context.schema.source_features)

    def _choose_alternative_state(
        self,
        source_name: str,
        current_value: object,
        runtime_context: RuntimeSearchContext,
        rng: np.random.Generator,
    ) -> object:
        choices = runtime_context.feature_contexts[source_name].search_states
        if len(choices) == 1:
            return choices[0]

        alternatives = [
            choice
            for choice in choices
            if str(choice) != str(current_value)
            or choice != current_value
        ]
        if not alternatives:
            alternatives = list(choices)
        return alternatives[int(rng.integers(0, len(alternatives)))]

    def _random_state_set(
        self,
        factual_state: dict[str, object],
        runtime_context: RuntimeSearchContext,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for _ in range(self._num_cfs):
            row: dict[str, object] = {}
            for source_name in runtime_context.schema.source_features:
                choices = runtime_context.feature_contexts[source_name].search_states
                row[source_name] = choices[int(rng.integers(0, len(choices)))]
            rows.append(row)
        return pd.DataFrame(rows, columns=runtime_context.schema.source_features)

    def _hamming_initial_state_set(
        self,
        factual_state: dict[str, object],
        runtime_context: RuntimeSearchContext,
        rng: np.random.Generator,
        order: int,
    ) -> pd.DataFrame:
        active_features = self._active_features(runtime_context)
        rows: list[dict[str, object]] = []
        for _ in range(self._num_cfs):
            row = dict(factual_state)
            if active_features:
                features_to_edit = rng.choice(
                    np.array(active_features, dtype=object),
                    size=min(order, len(active_features)),
                    replace=False,
                )
                for source_name in features_to_edit.tolist():
                    row[str(source_name)] = self._choose_alternative_state(
                        str(source_name),
                        row[str(source_name)],
                        runtime_context,
                        rng,
                    )
            rows.append(row)
        return pd.DataFrame(rows, columns=runtime_context.schema.source_features)

    def _initialize_state_set(
        self,
        factual_state: dict[str, object],
        runtime_context: RuntimeSearchContext,
        rng: np.random.Generator,
        init_type: str | None = None,
    ) -> pd.DataFrame:
        resolved_init_type = self._init_type if init_type is None else str(init_type).lower()
        if resolved_init_type == "random":
            return self._random_state_set(factual_state, runtime_context, rng)
        if resolved_init_type == "org":
            rows = [dict(factual_state) for _ in range(self._num_cfs)]
            return pd.DataFrame(rows, columns=runtime_context.schema.source_features)
        order = int(resolved_init_type.split("_")[-1])
        return self._hamming_initial_state_set(
            factual_state,
            runtime_context,
            rng,
            max(1, order),
        )

    def _resolve_num_row_edits(self, num_rows: int) -> int:
        if self._perturb_type == "one":
            return 1
        if self._perturb_type == "all":
            return num_rows
        fraction = int(self._perturb_type.split("_")[-1])
        return max(1, (num_rows * fraction) // 10)

    def _resolve_num_feature_edits(
        self,
        max_features: int,
        rng: np.random.Generator,
    ) -> int:
        capped_hamming = max(1, min(self._hamming_dist, max_features))
        base_weights = np.array(
            [1.0, 1.0, 0.4, 0.2, 0.1] + [0.05] * 30,
            dtype="float64",
        )
        weights = base_weights[:capped_hamming]
        weights = weights / weights.sum()
        return int(
            rng.choice(np.arange(1, capped_hamming + 1, dtype=int), p=weights)
        )

    def _perturb_state_set(
        self,
        base_state_set: pd.DataFrame,
        runtime_context: RuntimeSearchContext,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        candidates = base_state_set.copy(deep=True).reset_index(drop=True)
        active_features = self._active_features(runtime_context)
        num_row_edits = min(self._resolve_num_row_edits(len(candidates)), len(candidates))
        row_indices = rng.choice(
            np.arange(len(candidates), dtype=int),
            size=num_row_edits,
            replace=False,
        )

        for row_index in row_indices.tolist():
            if not active_features:
                continue
            num_feature_edits = self._resolve_num_feature_edits(len(active_features), rng)
            features_to_edit = rng.choice(
                np.array(active_features, dtype=object),
                size=min(num_feature_edits, len(active_features)),
                replace=False,
            )
            for source_name in features_to_edit.tolist():
                source_name = str(source_name)
                candidates.at[row_index, source_name] = self._choose_alternative_state(
                    source_name,
                    candidates.at[row_index, source_name],
                    runtime_context,
                    rng,
                )

        return candidates

    def _evaluate_state_set(
        self,
        candidate_states: pd.DataFrame,
        runtime_context: RuntimeSearchContext,
        target_index: int,
    ) -> _SearchArtifacts:
        encoded_set = encode_state_dataframe(candidate_states, runtime_context.schema)
        predicted_labels = self._predict_label_indices(encoded_set)
        valid_mask = predicted_labels.astype(int) == int(target_index)
        cost_matrix = compute_candidate_cost_matrix(
            candidate_states,
            runtime_context,
            valid_mask,
        )
        emc = compute_emc(cost_matrix, invalid_cost=self._invalid_cost)
        return _SearchArtifacts(
            state_set=candidate_states.reset_index(drop=True),
            encoded_set=encoded_set.reset_index(drop=True),
            cost_matrix=cost_matrix,
            valid_mask=valid_mask.astype(bool, copy=False),
            emc=emc,
        )

    def _run_cols_once(
        self,
        factual_state: dict[str, object],
        runtime_context: RuntimeSearchContext,
        target_index: int,
        run_budget: int,
        rng: np.random.Generator,
    ) -> _SearchArtifacts:
        run_start_queries = self._query_count
        init_type = self._init_type
        best: _SearchArtifacts | None = None
        best_buffer: pd.DataFrame | None = None
        for attempt in range(16):
            best = self._evaluate_state_set(
                self._initialize_state_set(
                    factual_state,
                    runtime_context,
                    rng,
                    init_type=init_type,
                ),
                runtime_context,
                target_index,
            )
            best_buffer = best.state_set.copy(deep=True)
            if bool(best.valid_mask.any()):
                break
            if attempt >= 9:
                init_type = "random"

        if best is None or best_buffer is None:
            raise RuntimeError("COLS initialization failed")
        linear_base = best.state_set.copy(deep=True)

        while self._query_count - run_start_queries < run_budget:
            base_state_set = (
                best_buffer if self._iter_type == "best" else linear_base
            )
            candidate_states = self._perturb_state_set(
                base_state_set,
                runtime_context,
                rng,
            )
            candidate = self._evaluate_state_set(
                candidate_states,
                runtime_context,
                target_index,
            )
            linear_base = candidate.state_set.copy(deep=True)

            if self._num_cfs == 1:
                if candidate.emc < best.emc:
                    best = candidate
                if best.emc <= 0.0:
                    break
                continue

            replacements = select_replacement_pairs(
                compute_benefit_matrix(best.cost_matrix, candidate.cost_matrix)
            )
            if not replacements:
                continue

            for best_index, candidate_index in replacements:
                best.state_set.iloc[best_index] = candidate.state_set.iloc[candidate_index]
                best.encoded_set.iloc[best_index] = candidate.encoded_set.iloc[
                    candidate_index
                ]
                best.cost_matrix[best_index] = candidate.cost_matrix[candidate_index]
                best.valid_mask[best_index] = candidate.valid_mask[candidate_index]
                best.emc = compute_emc(best.cost_matrix, invalid_cost=self._invalid_cost)
                valid_rows = best.state_set.loc[best.valid_mask].copy(deep=True)
                if not valid_rows.empty:
                    best_buffer = (
                        pd.concat(
                            [best_buffer, valid_rows],
                            ignore_index=True,
                        )
                        .tail(self._num_cfs)
                        .reset_index(drop=True)
                    )
            if best.emc <= 0.0:
                break

        return best

    def _run_search(
        self,
        factual_state: dict[str, object],
        runtime_context: RuntimeSearchContext,
        target_index: int,
        rng: np.random.Generator,
    ) -> _SearchArtifacts:
        run_budgets = split_budget(self._budget, self._num_parallel_runs)
        best_result: _SearchArtifacts | None = None

        for run_budget in run_budgets:
            if run_budget <= 0:
                continue
            result = self._run_cols_once(
                factual_state=factual_state,
                runtime_context=runtime_context,
                target_index=target_index,
                run_budget=run_budget,
                rng=rng,
            )
            if best_result is None or result.emc < best_result.emc:
                best_result = result

        if best_result is not None:
            return best_result

        return self._evaluate_state_set(
            self._initialize_state_set(factual_state, runtime_context, rng),
            runtime_context,
            target_index,
        )

    def get_counterfactual_sets(self, factuals: pd.DataFrame) -> list[pd.DataFrame]:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        decoded_factuals = decode_feature_dataframe(factuals, self._schema)

        self._last_counterfactual_sets = []
        self._last_counterfactual_validity = []
        self._last_search_stats = []

        with seed_context(self._seed) as active_seed:
            rng = np.random.default_rng(active_seed)

            for row_index in factuals.index:
                self._query_count = 0
                factual_encoded = factuals.loc[[row_index]].copy(deep=True)
                factual_state = decoded_factuals.loc[row_index].to_dict()

                original_prediction = int(self._predict_label_indices(factual_encoded)[0])
                target_index = resolve_target_index(
                    self._target_model,
                    original_prediction=original_prediction,
                    desired_class=self._desired_class,
                )
                runtime_context = build_runtime_context(
                    schema=self._schema,
                    decoded_training=self._decoded_training,
                    base_state_spaces=self._base_state_spaces,
                    factual_state=factual_state,
                    num_mcmc=self._num_mcmc,
                    alpha=self._alpha,
                    variance=self._variance,
                    invalid_cost=self._invalid_cost,
                    rng=rng,
                )
                result = self._run_search(
                    factual_state=factual_state,
                    runtime_context=runtime_context,
                    target_index=target_index,
                    rng=rng,
                )

                counterfactual_set = result.encoded_set.copy(deep=True)
                validity_mask = pd.Series(
                    result.valid_mask,
                    index=counterfactual_set.index,
                    dtype=bool,
                )
                self._last_counterfactual_sets.append(counterfactual_set)
                self._last_counterfactual_validity.append(validity_mask)
                self._last_search_stats.append(
                    {
                        "row_index": row_index,
                        "target_index": target_index,
                        "emc": result.emc,
                        "num_valid": int(result.valid_mask.sum()),
                        "num_queries": self._query_count,
                    }
                )

        return [counterfactual_set.copy(deep=True) for counterfactual_set in self._last_counterfactual_sets]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        counterfactual_sets = self.get_counterfactual_sets(factuals)

        selected_rows: list[pd.Series] = []
        for factual_index, counterfactual_set, validity_mask in zip(
            factuals.index,
            counterfactual_sets,
            self._last_counterfactual_validity,
            strict=True,
        ):
            selected_rows.append(
                choose_l1_closest(
                    factual_row=factuals.loc[factual_index],
                    candidate_set=counterfactual_set,
                    valid_mask=validity_mask.to_numpy(),
                )
            )

        return pd.DataFrame(
            selected_rows,
            index=factuals.index,
            columns=self._feature_names,
        )
