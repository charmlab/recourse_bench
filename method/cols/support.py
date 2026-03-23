from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from model.randomforest.randomforest import RandomForestModel

SupportedTargetModelTypes = (LinearModel, MlpModel, RandomForestModel)
EPSILON = 1e-7
DEFAULT_INVALID_COST = 99999.0


def ensure_supported_target_model(
    target_model: ModelObject,
    supported_types: Sequence[type[ModelObject]],
    method_name: str,
) -> None:
    if isinstance(target_model, tuple(supported_types)):
        return

    supported_names = ", ".join(cls.__name__ for cls in supported_types)
    raise TypeError(
        f"{method_name} supports target models [{supported_names}] only, "
        f"received {target_model.__class__.__name__}"
    )


def normalize_state_value(value: object) -> object:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return int(bool(value))
    return str(value)


def _safe_unique(values: Sequence[object]) -> list[object]:
    seen: set[object] = set()
    unique_values: list[object] = []
    for value in values:
        key = normalize_state_value(value)
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(value)
    return unique_values


def sort_states(values: Sequence[object], ordered: bool) -> list[object]:
    unique_values = _safe_unique(values)
    if not unique_values:
        return []

    if ordered:
        try:
            return sorted(unique_values, key=float)
        except (TypeError, ValueError):
            pass

    try:
        return sorted(unique_values)
    except TypeError:
        return unique_values


def is_ordered_feature(source_type: str, actionability: str) -> bool:
    return source_type == "numerical" or actionability in {
        "same-or-increase",
        "same-or-decrease",
    }


@dataclass(frozen=True)
class FeatureSpec:
    source_name: str
    source_type: str
    mutable: bool
    actionability: str
    representation: str
    output_columns: list[str]
    categories: list[object]


@dataclass(frozen=True)
class SearchSchema:
    output_columns: list[str]
    source_features: list[str]
    feature_specs: dict[str, FeatureSpec]


@dataclass(frozen=True)
class RuntimeFeatureContext:
    spec: FeatureSpec
    states: list[object]
    search_states: list[object]
    current_value: object
    current_index: int
    state_index: dict[object, int]
    percentiles: dict[object, float]


@dataclass(frozen=True)
class RuntimeSearchContext:
    schema: SearchSchema
    feature_contexts: dict[str, RuntimeFeatureContext]
    cost_matrices: dict[str, np.ndarray]
    num_mcmc: int
    invalid_cost: float


class RecourseModelAdapter:
    def __init__(self, target_model: ModelObject, feature_names: Sequence[str]):
        self._target_model = target_model
        self._feature_names = list(feature_names)

    def get_ordered_features(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self._feature_names].copy(deep=True)

    def predict_label_indices(self, X: pd.DataFrame) -> np.ndarray:
        prediction = self._target_model.get_prediction(
            self.get_ordered_features(X), proba=True
        )
        return prediction.detach().cpu().numpy().argmax(axis=1)


def resolve_target_index(
    target_model: ModelObject,
    original_prediction: int,
    desired_class: int | str | None,
) -> int:
    class_to_index = target_model.get_class_to_index()
    if desired_class is not None:
        if desired_class not in class_to_index:
            raise ValueError("desired_class is invalid for the trained target model")
        return int(class_to_index[desired_class])

    if len(class_to_index) != 2:
        raise ValueError(
            "desired_class=None is supported for binary classification only"
        )
    return int(1 - original_prediction)


def _infer_representation(source_name: str, columns: Sequence[str]) -> tuple[str, list[str]]:
    output_columns = list(columns)
    if len(output_columns) == 1:
        return "scalar", []

    onehot_prefix = f"{source_name}_cat_"
    thermometer_prefix = f"{source_name}_therm_"
    if all(column.startswith(onehot_prefix) for column in output_columns):
        categories = [column[len(onehot_prefix) :] for column in output_columns]
        return "onehot", categories
    if all(column.startswith(thermometer_prefix) for column in output_columns):
        categories = [column[len(thermometer_prefix) :] for column in output_columns]
        return "thermometer", categories
    raise ValueError(
        "Could not infer encoded representation for feature "
        f"{source_name}: {output_columns}"
    )


def build_search_schema(dataset: DatasetObject) -> SearchSchema:
    feature_df = dataset.get(target=False)
    output_columns = list(feature_df.columns)

    raw_feature_type = dataset.attr("raw_feature_type")
    raw_feature_mutability = dataset.attr("raw_feature_mutability")
    raw_feature_actionability = dataset.attr("raw_feature_actionability")

    source_features = [
        feature_name
        for feature_name in raw_feature_type
        if feature_name != dataset.target_column
    ]
    if hasattr(dataset, "encoding"):
        encoding_map = dataset.attr("encoding")
    else:
        encoding_map = {}

    source_to_columns: dict[str, list[str]] = {}
    for source_name in source_features:
        if source_name in encoding_map:
            source_to_columns[source_name] = list(encoding_map[source_name])
            continue
        if source_name in output_columns:
            source_to_columns[source_name] = [source_name]
            continue
        raise KeyError(f"Could not resolve output columns for source feature: {source_name}")

    feature_specs: dict[str, FeatureSpec] = {}
    for source_name in source_features:
        representation, categories = _infer_representation(
            source_name, source_to_columns[source_name]
        )
        feature_specs[source_name] = FeatureSpec(
            source_name=source_name,
            source_type=str(raw_feature_type[source_name]).lower(),
            mutable=bool(raw_feature_mutability[source_name]),
            actionability=str(raw_feature_actionability[source_name]).lower(),
            representation=representation,
            output_columns=list(source_to_columns[source_name]),
            categories=list(categories),
        )

    return SearchSchema(
        output_columns=output_columns,
        source_features=source_features,
        feature_specs=feature_specs,
    )


def decode_feature_dataframe(
    df: pd.DataFrame,
    schema: SearchSchema,
) -> pd.DataFrame:
    decoded_columns: dict[str, object] = {}
    for source_name in schema.source_features:
        spec = schema.feature_specs[source_name]
        if spec.representation == "scalar":
            decoded_columns[source_name] = df[spec.output_columns[0]].copy(deep=True)
            continue

        feature_frame = df.loc[:, spec.output_columns].copy(deep=True)
        if spec.representation == "onehot":
            values = feature_frame.to_numpy(dtype="float32")
            category_index = values.argmax(axis=1)
            decoded_columns[source_name] = [
                spec.categories[int(index)] for index in category_index
            ]
            continue

        thermometer_values = feature_frame.to_numpy(dtype="float32")
        category_counts = np.rint(thermometer_values.sum(axis=1)).astype(int)
        category_counts = category_counts.clip(1, len(spec.categories))
        decoded_columns[source_name] = [
            spec.categories[int(count - 1)] for count in category_counts
        ]

    return pd.DataFrame(decoded_columns, index=df.index)


def encode_state_dataframe(
    state_df: pd.DataFrame,
    schema: SearchSchema,
) -> pd.DataFrame:
    encoded_columns: dict[str, object] = {}
    for source_name in schema.source_features:
        spec = schema.feature_specs[source_name]
        source_values = state_df[source_name]

        if spec.representation == "scalar":
            encoded_columns[spec.output_columns[0]] = source_values.to_numpy(copy=True)
            continue

        category_to_index = {
            normalize_state_value(category): idx
            for idx, category in enumerate(spec.categories)
        }
        encoded_index = np.array(
            [
                category_to_index.get(normalize_state_value(value), 0)
                for value in source_values.tolist()
            ],
            dtype=int,
        )

        if spec.representation == "onehot":
            for index, column in enumerate(spec.output_columns):
                encoded_columns[column] = (encoded_index == index).astype("float32")
            continue

        for index, column in enumerate(spec.output_columns):
            encoded_columns[column] = (encoded_index >= index).astype("float32")

    encoded_df = pd.DataFrame(encoded_columns, index=state_df.index)
    return encoded_df.loc[:, schema.output_columns]


def infer_base_state_spaces(
    decoded_training: pd.DataFrame,
    schema: SearchSchema,
) -> dict[str, list[object]]:
    state_spaces: dict[str, list[object]] = {}
    for source_name in schema.source_features:
        spec = schema.feature_specs[source_name]
        ordered = is_ordered_feature(spec.source_type, spec.actionability)
        if spec.representation in {"onehot", "thermometer"}:
            state_spaces[source_name] = list(spec.categories)
            continue
        state_spaces[source_name] = sort_states(
            decoded_training[source_name].dropna().tolist(),
            ordered=ordered,
        )
    return state_spaces


def _ensure_state_in_space(
    states: list[object],
    current_value: object,
    ordered: bool,
) -> list[object]:
    augmented = list(states)
    current_key = normalize_state_value(current_value)
    if current_key not in {normalize_state_value(value) for value in augmented}:
        augmented.append(current_value)
        augmented = sort_states(augmented, ordered=ordered)
    return augmented


def get_state_index(
    state_index: dict[object, int],
    states: Sequence[object],
    value: object,
) -> int:
    normalized_value = normalize_state_value(value)
    if normalized_value in state_index:
        return int(state_index[normalized_value])

    try:
        state_array = np.asarray([float(state) for state in states], dtype="float64")
        return int(np.argmin(np.abs(state_array - float(normalized_value))))
    except (TypeError, ValueError):
        for index, state in enumerate(states):
            if str(state) == str(value):
                return index
    raise KeyError(f"Unknown state value: {value}")


def _resolve_search_states(
    spec: FeatureSpec,
    states: list[object],
    current_index: int,
) -> list[object]:
    return list(states)


def _build_percentile_lookup(
    series: pd.Series,
    states: Sequence[object],
    spec: FeatureSpec,
) -> dict[object, float]:
    if not is_ordered_feature(spec.source_type, spec.actionability):
        return {}

    if spec.source_type == "numerical":
        values = np.sort(series.astype("float64").to_numpy())
        if values.size == 0:
            return {normalize_state_value(state): 0.0 for state in states}
        return {
            normalize_state_value(state): float(
                np.searchsorted(values, float(state), side="right") / values.size
            )
            for state in states
        }

    value_counts = series.value_counts()
    total = float(max(1, int(series.shape[0])))
    cumulative = 0.0
    percentile_lookup: dict[object, float] = {}
    for state in states:
        cumulative += float(value_counts.get(state, 0))
        percentile_lookup[normalize_state_value(state)] = cumulative / total
    return percentile_lookup


def _ordered_linear_cost_means(
    states: Sequence[object],
    current_index: int,
    actionability: str,
) -> np.ndarray:
    means = np.full(len(states), np.inf, dtype="float64")
    means[current_index] = 0.0

    distance_above = len(states) - current_index - 1
    distance_below = current_index

    for state_index in range(current_index + 1, len(states)):
        if actionability == "same-or-decrease":
            continue
        if distance_above <= 0:
            continue
        means[state_index] = (state_index - current_index) / distance_above

    for state_index in range(0, current_index):
        if actionability == "same-or-increase":
            continue
        if distance_below <= 0:
            continue
        means[state_index] = (current_index - state_index) / distance_below

    return means


def _ordered_percentile_cost_means(
    states: Sequence[object],
    current_index: int,
    percentile_lookup: dict[object, float],
    actionability: str,
) -> np.ndarray:
    means = np.full(len(states), np.inf, dtype="float64")
    current_key = normalize_state_value(states[current_index])
    current_percentile = percentile_lookup.get(current_key, 0.0)
    means[current_index] = 0.0

    for state_index, state in enumerate(states):
        if state_index == current_index:
            continue
        if actionability == "same-or-increase" and state_index < current_index:
            continue
        if actionability == "same-or-decrease" and state_index > current_index:
            continue
        state_percentile = percentile_lookup.get(normalize_state_value(state), 0.0)
        means[state_index] = abs(state_percentile - current_percentile)
    return means


def _unordered_cost_means(
    num_states: int,
    current_index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    means = rng.uniform(0.0, 1.0, size=num_states).astype("float64")
    means[current_index] = 0.0
    return means


def _compute_feature_cost_means(
    feature_context: RuntimeFeatureContext,
    preference_score: float,
    editable: bool,
    alpha: float,
    variance: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    num_states = len(feature_context.states)
    means = np.full(num_states, np.inf, dtype="float64")
    vars_ = np.zeros(num_states, dtype="float64")
    means[feature_context.current_index] = 0.0

    if (not editable) or num_states <= 1:
        return means, vars_

    spec = feature_context.spec
    if is_ordered_feature(spec.source_type, spec.actionability):
        linear_means = _ordered_linear_cost_means(
            feature_context.states,
            feature_context.current_index,
            spec.actionability,
        )
        percentile_means = _ordered_percentile_cost_means(
            feature_context.states,
            feature_context.current_index,
            feature_context.percentiles,
            spec.actionability,
        )
    else:
        linear_means = _unordered_cost_means(
            num_states,
            feature_context.current_index,
            rng,
        )
        percentile_means = _unordered_cost_means(
            num_states,
            feature_context.current_index,
            rng,
        )

    finite_mask = np.isfinite(linear_means) & np.isfinite(percentile_means)
    if finite_mask.any():
        means[finite_mask] = (
            alpha * linear_means[finite_mask]
            + (1.0 - alpha) * percentile_means[finite_mask]
        )
        means[finite_mask] *= 1.0 - preference_score
    means[feature_context.current_index] = 0.0
    vars_[(means > 0.0) & np.isfinite(means)] = variance
    return means, vars_


def sample_cost_vector(
    means: np.ndarray,
    vars_: np.ndarray,
    invalid_cost: float,
    rng: np.random.Generator,
) -> np.ndarray:
    samples = np.full(means.shape, invalid_cost, dtype="float64")
    zero_mask = means == 0.0
    finite_mask = np.isfinite(means)
    positive_mask = finite_mask & (means > 0.0)

    if positive_mask.any():
        mean_values = means[positive_mask] + EPSILON
        mean_values = np.clip(mean_values, EPSILON, 1.0 - EPSILON)
        var_values = vars_[positive_mask] + EPSILON

        alpha_values = (
            ((1.0 - mean_values) / var_values) - (1.0 / mean_values)
        ) * np.square(mean_values)
        alpha_values = np.maximum(alpha_values, EPSILON)
        beta_values = alpha_values * ((1.0 / mean_values) - 1.0)
        beta_values = np.maximum(beta_values, EPSILON)
        samples[positive_mask] = rng.beta(alpha_values, beta_values)

    samples[zero_mask] = 0.0
    return samples


def _sample_preference_scores(
    source_features: Sequence[str],
    editable_features: set[str],
    rng: np.random.Generator,
) -> dict[str, float]:
    if not editable_features:
        return {feature: 0.0 for feature in source_features}

    concentration = np.array(
        [1.0 if feature in editable_features else EPSILON for feature in source_features],
        dtype="float64",
    )
    preference_vector = rng.dirichlet(concentration)
    return {
        feature: (
            float(preference_vector[index]) if feature in editable_features else 0.0
        )
        for index, feature in enumerate(source_features)
    }


def build_runtime_context(
    schema: SearchSchema,
    decoded_training: pd.DataFrame,
    base_state_spaces: dict[str, list[object]],
    factual_state: dict[str, object],
    num_mcmc: int,
    alpha: float | None,
    variance: float,
    invalid_cost: float,
    rng: np.random.Generator,
) -> RuntimeSearchContext:
    feature_contexts: dict[str, RuntimeFeatureContext] = {}
    mutable_features: list[str] = []

    for source_name in schema.source_features:
        spec = schema.feature_specs[source_name]
        ordered = is_ordered_feature(spec.source_type, spec.actionability)
        current_value = factual_state[source_name]
        states = _ensure_state_in_space(
            base_state_spaces[source_name],
            current_value,
            ordered=ordered,
        )
        state_index = {
            normalize_state_value(state): index for index, state in enumerate(states)
        }
        current_index = get_state_index(state_index, states, current_value)
        search_states = _resolve_search_states(spec, states, current_index)
        percentiles = _build_percentile_lookup(
            decoded_training[source_name],
            states,
            spec,
        )
        feature_contexts[source_name] = RuntimeFeatureContext(
            spec=spec,
            states=states,
            search_states=search_states,
            current_value=current_value,
            current_index=current_index,
            state_index=state_index,
            percentiles=percentiles,
        )
        if len(search_states) > 1 and spec.mutable and spec.actionability not in {
            "none",
            "same",
        }:
            mutable_features.append(source_name)

    cost_matrices = {
        source_name: np.empty(
            (num_mcmc, len(feature_contexts[source_name].states)),
            dtype="float64",
        )
        for source_name in schema.source_features
    }

    for sample_index in range(num_mcmc):
        if mutable_features:
            subset_size = int(rng.integers(1, len(mutable_features) + 1))
            editable_array = rng.choice(
                np.array(mutable_features, dtype=object),
                size=subset_size,
                replace=False,
            )
            editable_features = {str(feature) for feature in editable_array.tolist()}
        else:
            editable_features = set()

        preference_scores = _sample_preference_scores(
            schema.source_features,
            editable_features,
            rng,
        )
        sample_alpha = float(alpha) if alpha is not None else float(
            np.round(rng.uniform(0.0, 1.0), 2)
        )

        for source_name in schema.source_features:
            feature_context = feature_contexts[source_name]
            means, vars_ = _compute_feature_cost_means(
                feature_context=feature_context,
                preference_score=preference_scores[source_name],
                editable=source_name in editable_features,
                alpha=sample_alpha,
                variance=variance,
                rng=rng,
            )
            cost_matrices[source_name][sample_index] = sample_cost_vector(
                means,
                vars_,
                invalid_cost=invalid_cost,
                rng=rng,
            )

    return RuntimeSearchContext(
        schema=schema,
        feature_contexts=feature_contexts,
        cost_matrices=cost_matrices,
        num_mcmc=num_mcmc,
        invalid_cost=invalid_cost,
    )


def compute_candidate_cost_matrix(
    candidate_states: pd.DataFrame,
    runtime_context: RuntimeSearchContext,
    valid_mask: np.ndarray,
) -> np.ndarray:
    num_candidates = candidate_states.shape[0]
    cost_matrix = np.full(
        (num_candidates, runtime_context.num_mcmc),
        runtime_context.invalid_cost,
        dtype="float64",
    )

    valid_indices = np.flatnonzero(valid_mask.astype(bool))
    for row_index in valid_indices:
        total_cost = np.zeros(runtime_context.num_mcmc, dtype="float64")
        candidate_row = candidate_states.iloc[row_index]
        for source_name in runtime_context.schema.source_features:
            feature_context = runtime_context.feature_contexts[source_name]
            state_index = get_state_index(
                feature_context.state_index,
                feature_context.states,
                candidate_row[source_name],
            )
            total_cost += runtime_context.cost_matrices[source_name][:, state_index]
        cost_matrix[row_index] = total_cost
    return cost_matrix


def compute_emc(cost_matrix: np.ndarray, invalid_cost: float = DEFAULT_INVALID_COST) -> float:
    if cost_matrix.size == 0:
        return float(invalid_cost)
    return float(cost_matrix.min(axis=0).mean())


def compute_benefit_matrix(
    best_cost_matrix: np.ndarray,
    candidate_cost_matrix: np.ndarray,
) -> np.ndarray:
    num_best, num_samples = best_cost_matrix.shape
    num_candidates = candidate_cost_matrix.shape[0]
    benefits = np.zeros((num_best, num_candidates), dtype="float64")

    if num_best == 0 or num_candidates == 0:
        return benefits

    best_index_per_sample = best_cost_matrix.argmin(axis=0)
    if num_best > 1:
        second_best_index_per_sample = best_cost_matrix.argsort(axis=0)[1]
    else:
        second_best_index_per_sample = best_index_per_sample.copy()

    for best_index in range(num_best):
        relevant_samples = np.where(best_index_per_sample == best_index)[0]
        if relevant_samples.size == 0:
            continue
        for candidate_index in range(num_candidates):
            benefit = 0.0
            for sample_index in relevant_samples:
                best_cost = best_cost_matrix[best_index, sample_index]
                candidate_cost = candidate_cost_matrix[candidate_index, sample_index]
                if candidate_cost < best_cost:
                    benefit += best_cost - candidate_cost
                    continue
                backup_index = second_best_index_per_sample[sample_index]
                backup_cost = best_cost_matrix[backup_index, sample_index]
                benefit += best_cost - min(candidate_cost, backup_cost)
            benefits[best_index, candidate_index] = benefit
    return benefits


def select_replacement_pairs(benefits: np.ndarray) -> list[tuple[int, int]]:
    if benefits.size == 0 or not bool((benefits > 0.0).any()):
        return []

    replacements: list[tuple[int, int]] = []
    used_best: set[int] = set()
    for candidate_index in range(benefits.shape[1]):
        ranked_best = benefits[:, candidate_index].argsort()[::-1]
        for best_index in ranked_best:
            if best_index in used_best:
                continue
            if benefits[best_index, candidate_index] <= 0.0:
                continue
            replacements.append((int(best_index), int(candidate_index)))
            used_best.add(int(best_index))
            break
    return replacements


def split_budget(total_budget: int, num_runs: int) -> list[int]:
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")
    base_budget = total_budget // num_runs
    remainder = total_budget % num_runs
    return [
        base_budget + (1 if run_index < remainder else 0)
        for run_index in range(num_runs)
    ]


def choose_l1_closest(
    factual_row: pd.Series,
    candidate_set: pd.DataFrame,
    valid_mask: Sequence[bool],
) -> pd.Series:
    valid_series = pd.Series(valid_mask, index=candidate_set.index, dtype=bool)
    valid_candidates = candidate_set.loc[valid_series.to_numpy()].copy(deep=True)
    if valid_candidates.empty:
        return pd.Series(np.nan, index=candidate_set.columns, dtype="float64")

    try:
        factual_values = factual_row.loc[candidate_set.columns].to_numpy(dtype="float64")
        candidate_values = valid_candidates.to_numpy(dtype="float64")
        distances = np.abs(candidate_values - factual_values).sum(axis=1)
        best_index = int(np.argmin(distances))
        return valid_candidates.iloc[best_index].copy(deep=True)
    except ValueError:
        distances = (valid_candidates != factual_row.loc[candidate_set.columns]).sum(
            axis=1
        )
        best_label = distances.astype("float64").idxmin()
        return valid_candidates.loc[best_label].copy(deep=True)
