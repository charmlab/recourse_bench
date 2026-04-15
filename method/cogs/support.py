from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from model.randomforest.randomforest import RandomForestModel
from preprocess.preprocess_utils import resolve_feature_metadata

TorchModelTypes = (LinearModel, MlpModel)
BlackBoxModelTypes = (LinearModel, MlpModel, RandomForestModel)


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


def to_feature_dataframe(
    values: pd.DataFrame | np.ndarray | torch.Tensor,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.loc[:, list(feature_names)].copy(deep=True)

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    if array.ndim == 1:
        array = array.reshape(1, -1)
    return pd.DataFrame(array, columns=list(feature_names))


@dataclass
class FeatureGroups:
    feature_names: list[str]
    continuous: list[str]
    categorical: list[str]
    binary: list[str]
    immutable: list[str]
    mutable: list[str]
    plausibility_constraints: list[str | None]


def resolve_feature_groups(dataset: DatasetObject) -> FeatureGroups:
    feature_df = dataset.get(target=False)
    feature_names = list(feature_df.columns)
    feature_type, feature_mutability, feature_actionability = resolve_feature_metadata(
        dataset
    )

    continuous: list[str] = []
    categorical: list[str] = []
    binary: list[str] = []
    immutable: list[str] = []
    mutable: list[str] = []
    plausibility_constraints: list[str | None] = []

    for feature_name in feature_names:
        feature_kind = str(feature_type[feature_name]).lower()
        is_mutable = bool(feature_mutability[feature_name])
        actionability = str(feature_actionability[feature_name]).lower()

        if feature_kind == "numerical":
            continuous.append(feature_name)
        else:
            categorical.append(feature_name)
            if feature_kind == "binary":
                binary.append(feature_name)

        constraint: str | None = None
        if (not is_mutable) or actionability in {"none", "same"}:
            immutable.append(feature_name)
            constraint = "="
        else:
            mutable.append(feature_name)
            if actionability in {"same-or-increase", "increase"}:
                constraint = ">="
            elif actionability in {"same-or-decrease", "decrease"}:
                constraint = "<="
            elif actionability == "any":
                constraint = None
            else:
                constraint = None
        plausibility_constraints.append(constraint)

    return FeatureGroups(
        feature_names=feature_names,
        continuous=continuous,
        categorical=categorical,
        binary=binary,
        immutable=immutable,
        mutable=mutable,
        plausibility_constraints=plausibility_constraints,
    )


class RecourseModelAdapter:
    def __init__(self, target_model: ModelObject, feature_names: Sequence[str]):
        self._target_model = target_model
        self._feature_names = list(feature_names)
        class_to_index = target_model.get_class_to_index()
        self._index_to_class = {
            index: class_value for class_value, index in class_to_index.items()
        }
        self.classes_ = np.array(
            [self._index_to_class[index] for index in sorted(self._index_to_class)]
        )

    def get_ordered_features(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> pd.DataFrame:
        return to_feature_dataframe(X, self._feature_names)

    def predict_proba(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        features = self.get_ordered_features(X)
        prediction = self._target_model.get_prediction(features, proba=True)
        if isinstance(prediction, torch.Tensor):
            return prediction.detach().cpu().numpy()
        return np.asarray(prediction)

    def predict(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        features = self.get_ordered_features(X)
        prediction = self._target_model.get_prediction(features, proba=False)
        if isinstance(prediction, torch.Tensor):
            label_indices = prediction.detach().cpu().numpy().argmax(axis=1)
        else:
            label_indices = np.asarray(prediction).argmax(axis=1)
        return np.asarray(
            [self._index_to_class[int(index)] for index in label_indices], dtype=object
        )

    def predict_label_indices(
        self, X: pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        features = self.get_ordered_features(X)
        probabilities = self._target_model.get_prediction(features, proba=True)
        if isinstance(probabilities, torch.Tensor):
            return probabilities.detach().cpu().numpy().argmax(axis=1)
        return np.asarray(probabilities).argmax(axis=1)


def resolve_target_classes(
    target_model: ModelObject,
    original_predictions: np.ndarray,
    desired_class: int | str | None,
) -> np.ndarray:
    class_to_index = target_model.get_class_to_index()
    index_to_class = {index: label for label, index in class_to_index.items()}

    if desired_class is not None:
        if desired_class not in class_to_index:
            raise ValueError("desired_class is invalid for the trained target model")
        return np.full(
            shape=original_predictions.shape,
            fill_value=desired_class,
            dtype=object,
        )

    if len(class_to_index) != 2:
        raise ValueError(
            "desired_class=None is supported for binary classification only"
        )

    opposite_indices = 1 - original_predictions.astype(np.int64, copy=False)
    return np.asarray([index_to_class[int(index)] for index in opposite_indices], dtype=object)


def validate_counterfactuals(
    target_model: ModelObject,
    factuals: pd.DataFrame,
    candidates: pd.DataFrame,
    desired_class: int | str | None = None,
) -> pd.DataFrame:
    if list(candidates.columns) != list(factuals.columns):
        candidates = candidates.reindex(columns=factuals.columns)
    candidates = candidates.copy(deep=True)

    if candidates.shape[0] != factuals.shape[0]:
        raise ValueError("Candidates must preserve the number of factual rows")

    valid_rows = ~candidates.isna().any(axis=1)
    if not bool(valid_rows.any()):
        return candidates

    adapter = RecourseModelAdapter(target_model, factuals.columns)
    original_prediction = adapter.predict_label_indices(factuals)
    target_classes = resolve_target_classes(
        target_model=target_model,
        original_predictions=original_prediction,
        desired_class=desired_class,
    )

    candidate_prediction = adapter.predict(candidates.loc[valid_rows])
    success_mask = pd.Series(False, index=candidates.index, dtype=bool)
    success_mask.loc[valid_rows] = candidate_prediction == target_classes[
        valid_rows.to_numpy()
    ]
    candidates.loc[~success_mask, :] = np.nan
    return candidates


def compute_ranges_numerical_features(
    feature_intervals: np.ndarray,
    indices_categorical_features: Sequence[int] | None,
) -> np.ndarray:
    mask_num = np.ones(len(feature_intervals), dtype=bool)
    if indices_categorical_features:
        mask_num[list(indices_categorical_features)] = False
    num_feature_intervals = feature_intervals[mask_num]
    if len(num_feature_intervals) == 0:
        return np.array([], dtype=np.float64)
    ranges = np.array(
        [
            float(interval[1] - interval[0])
            if not np.isclose(float(interval[1]), float(interval[0]))
            else 1.0
            for interval in num_feature_intervals
        ],
        dtype=np.float64,
    )
    return ranges


def get_mask_categorical_features(
    num_features: int, indices_categorical_features: Sequence[int] | None
) -> np.ndarray:
    mask_categorical = np.zeros(num_features, dtype=bool)
    if indices_categorical_features:
        mask_categorical[list(indices_categorical_features)] = True
    return mask_categorical


def gower_distance(
    candidates: np.ndarray,
    x: np.ndarray,
    num_feature_ranges: np.ndarray,
    indices_categorical_features: Sequence[int] | None = None,
) -> np.ndarray:
    is_single_candidate = candidates.ndim == 1
    if is_single_candidate:
        candidates = candidates.reshape((1, -1))

    num_features = candidates.shape[1]
    d_c = np.zeros(shape=candidates.shape[0], dtype=np.float64)
    if indices_categorical_features:
        categorical_indices = list(indices_categorical_features)
        d_c = np.sum(
            candidates[:, categorical_indices] != x[categorical_indices], axis=1
        ).astype(np.float64)
        is_numerical = np.ones(num_features, dtype=bool)
        is_numerical[categorical_indices] = False
        candidates_num = candidates[:, is_numerical]
        x_num = x[is_numerical]
    else:
        candidates_num = candidates
        x_num = x

    if candidates_num.size == 0:
        d_n = np.zeros(shape=candidates.shape[0], dtype=np.float64)
    else:
        d_n = np.sum(
            np.divide(
                np.abs(candidates_num - x_num),
                num_feature_ranges,
                where=num_feature_ranges != 0,
                out=np.zeros_like(candidates_num, dtype=np.float64),
            ),
            axis=1,
        )
    distances = (d_c + d_n) / num_features
    if is_single_candidate:
        return distances[0]
    return distances


def fix_categorical_features(
    genes: np.ndarray,
    feature_intervals: np.ndarray,
    indices_categorical_features: Sequence[int] | None,
) -> np.ndarray:
    is_single_candidate = genes.ndim == 1
    if is_single_candidate:
        genes = genes.reshape((1, -1))

    if indices_categorical_features:
        for feature_index in indices_categorical_features:
            categories = np.asarray(feature_intervals[feature_index], dtype=np.float64)
            distances = np.abs(
                np.repeat(genes[:, feature_index], len(categories)).reshape(
                    (-1, len(categories))
                )
                - categories
            )
            closest_category_indices = np.argmin(distances, axis=1)
            genes[:, feature_index] = categories[closest_category_indices]

    if is_single_candidate:
        return genes[0]
    return genes


def fix_features(
    genes: np.ndarray,
    x: np.ndarray,
    feature_intervals: np.ndarray,
    indices_categorical_features: Sequence[int] | None = None,
    plausibility_constraints: Sequence[str | None] | None = None,
) -> np.ndarray:
    is_single_candidate = genes.ndim == 1
    if is_single_candidate:
        genes = genes.reshape((1, -1))

    if plausibility_constraints is not None:
        for feature_index, constraint in enumerate(plausibility_constraints):
            if constraint == "=":
                genes[:, feature_index] = x[feature_index]
            elif constraint == ">=":
                genes[:, feature_index] = np.where(
                    genes[:, feature_index] > x[feature_index],
                    genes[:, feature_index],
                    x[feature_index],
                )
            elif constraint == "<=":
                genes[:, feature_index] = np.where(
                    genes[:, feature_index] < x[feature_index],
                    genes[:, feature_index],
                    x[feature_index],
                )

    genes = fix_categorical_features(
        genes, feature_intervals, indices_categorical_features
    )
    if is_single_candidate:
        genes = genes.reshape((1, -1))

    mask_categorical = get_mask_categorical_features(
        genes.shape[1], indices_categorical_features
    )
    for feature_index, is_categorical in enumerate(mask_categorical):
        if is_categorical:
            continue
        low = float(feature_intervals[feature_index][0])
        high = float(feature_intervals[feature_index][1])
        genes[:, feature_index] = np.clip(genes[:, feature_index], low, high)

    if is_single_candidate:
        return genes[0]
    return genes


def gower_fitness_function(
    genes: np.ndarray,
    x: np.ndarray,
    blackbox: RecourseModelAdapter,
    desired_class: int | str,
    feature_intervals: np.ndarray,
    indices_categorical_features: Sequence[int] | None = None,
    plausibility_constraints: Sequence[str | None] | None = None,
    apply_fixes: bool = False,
) -> np.ndarray:
    is_single_candidate = genes.ndim == 1
    if is_single_candidate:
        genes = genes.reshape((1, -1))

    if apply_fixes:
        genes = fix_features(
            genes,
            x,
            feature_intervals,
            indices_categorical_features=indices_categorical_features,
            plausibility_constraints=plausibility_constraints,
        )

    num_feature_ranges = compute_ranges_numerical_features(
        feature_intervals, indices_categorical_features
    )
    gower_dist = gower_distance(
        genes, x, num_feature_ranges, indices_categorical_features
    )
    l_0 = np.sum(genes != x, axis=1) / genes.shape[1]
    dist = 0.5 * gower_dist + 0.5 * l_0

    preds = blackbox.predict(genes)
    failed_preds = preds != desired_class
    dist = dist + failed_preds.astype(np.float64)
    fitness_values = -dist
    if is_single_candidate:
        return fitness_values[0]
    return fitness_values


class Population:
    def __init__(self, population_size: int, genotype_length: int):
        self.genes = np.empty(shape=(population_size, genotype_length), dtype=np.float64)
        self.fitnesses = np.zeros(shape=(population_size,), dtype=np.float64)

    def initialize(
        self,
        x: np.ndarray,
        feature_intervals: np.ndarray,
        indices_categorical_features: Sequence[int] | None,
        plausibility_constraints: Sequence[str | None] | None = None,
        temperature: float = 0.8,
    ) -> None:
        n = self.genes.shape[0]
        mask_categorical = get_mask_categorical_features(
            self.genes.shape[1], indices_categorical_features
        )

        for feature_index, is_categorical in enumerate(mask_categorical):
            init_feature = None
            constraint = (
                plausibility_constraints[feature_index]
                if plausibility_constraints is not None
                else None
            )
            if is_categorical:
                if constraint == "=":
                    init_feature = np.full(n, x[feature_index], dtype=np.float64)
                else:
                    init_feature = np.random.choice(
                        np.asarray(feature_intervals[feature_index], dtype=np.float64),
                        size=n,
                    )
            else:
                low = float(feature_intervals[feature_index][0])
                high = float(feature_intervals[feature_index][1])
                if constraint == "=":
                    init_feature = np.full(n, x[feature_index], dtype=np.float64)
                elif constraint == ">=":
                    init_feature = np.random.uniform(low=x[feature_index], high=high, size=n)
                elif constraint == "<=":
                    init_feature = np.random.uniform(low=low, high=x[feature_index], size=n)
                else:
                    init_feature = np.random.uniform(low=low, high=high, size=n)
            self.genes[:, feature_index] = init_feature

        which_from_x = np.random.choice(
            (True, False),
            size=self.genes.shape,
            p=[1.0 - temperature, temperature],
        )
        self.genes = np.where(which_from_x, x, self.genes)

    def stack(self, other: "Population") -> None:
        self.genes = np.vstack((self.genes, other.genes))
        self.fitnesses = np.concatenate((self.fitnesses, other.fitnesses))

    def shuffle(self) -> None:
        random_order = np.random.permutation(self.genes.shape[0])
        self.genes = self.genes[random_order, :]
        self.fitnesses = self.fitnesses[random_order]

    def is_converged(self) -> bool:
        return len(np.unique(self.genes, axis=0)) < 2

    def delete(self, indices: Sequence[int]) -> None:
        self.genes = np.delete(self.genes, indices, axis=0)
        self.fitnesses = np.delete(self.fitnesses, indices)


def crossover(genes: np.ndarray) -> np.ndarray:
    offspring = genes.copy()
    parents_1 = np.vstack((genes[: len(genes) // 2], genes[: len(genes) // 2]))
    parents_2 = np.vstack((genes[len(genes) // 2 :], genes[len(genes) // 2 :]))
    mask_cross = np.random.choice([True, False], size=genes.shape)
    offspring = np.where(mask_cross, parents_1, parents_2)
    return offspring


def generate_plausible_mutations(
    genes: np.ndarray,
    feature_intervals: np.ndarray,
    indices_categorical_features: Sequence[int] | None,
    x: np.ndarray,
    plausibility_constraints: Sequence[str | None] | None = None,
    num_features_mutation_strength: float = 0.25,
) -> np.ndarray:
    mask_categorical = get_mask_categorical_features(
        genes.shape[1], indices_categorical_features
    )
    mutations = np.zeros(shape=genes.shape, dtype=np.float64)

    for feature_index, is_categorical in enumerate(mask_categorical):
        constraint = (
            plausibility_constraints[feature_index]
            if plausibility_constraints is not None
            else None
        )
        if is_categorical:
            candidates = np.asarray(feature_intervals[feature_index], dtype=np.float64)
            if constraint == "=":
                candidates = np.array([x[feature_index]], dtype=np.float64)
            mutations[:, feature_index] = np.random.choice(
                candidates, size=mutations.shape[0]
            )
            continue

        if constraint == "=":
            low = high = range_num = 0.0
        elif constraint == ">=":
            range_num = float(feature_intervals[feature_index][1] - x[feature_index])
            low = 0.0
            high = num_features_mutation_strength
        elif constraint == "<=":
            range_num = float(x[feature_index] - feature_intervals[feature_index][0])
            low = -num_features_mutation_strength
            high = 0.0
        else:
            range_num = float(
                feature_intervals[feature_index][1] - feature_intervals[feature_index][0]
            )
            low = -num_features_mutation_strength / 2.0
            high = num_features_mutation_strength / 2.0

        mutations[:, feature_index] = range_num * np.random.uniform(
            low=low, high=high, size=mutations.shape[0]
        )
        mutations[:, feature_index] += genes[:, feature_index]
        mutations[:, feature_index] = np.clip(
            mutations[:, feature_index],
            float(feature_intervals[feature_index][0]),
            float(feature_intervals[feature_index][1]),
        )

    return mutations


def mutate(
    genes: np.ndarray,
    feature_intervals: np.ndarray,
    indices_categorical_features: Sequence[int] | None,
    x: np.ndarray,
    plausibility_constraints: Sequence[str | None] | None = None,
    mutation_probability: float = 0.1,
    num_features_mutation_strength: float = 0.05,
) -> np.ndarray:
    mask_mut = np.random.choice(
        [True, False],
        size=genes.shape,
        p=[mutation_probability, 1.0 - mutation_probability],
    )
    mutations = generate_plausible_mutations(
        genes,
        feature_intervals,
        indices_categorical_features,
        x,
        plausibility_constraints,
        num_features_mutation_strength,
    )
    offspring = np.where(mask_mut, mutations, genes).astype(np.float64)
    return offspring


def truncation_select(population: Population, selection_size: int) -> Population:
    genotype_length = population.genes.shape[1]
    selected = Population(selection_size, genotype_length)
    population.shuffle()
    sort_order = np.argsort(population.fitnesses * -1.0)[:selection_size]
    selected.genes = population.genes[sort_order, :]
    selected.fitnesses = population.fitnesses[sort_order]
    return selected


def one_tournament_round(
    population: Population,
    tournament_size: int,
    return_winner_index: bool = False,
):
    rand_perm = np.random.permutation(len(population.fitnesses))
    competing_fitnesses = population.fitnesses[rand_perm[:tournament_size]]
    winning_index = rand_perm[np.argmax(competing_fitnesses)]
    if return_winner_index:
        return winning_index
    return {
        "genotype": population.genes[winning_index, :],
        "fitness": population.fitnesses[winning_index],
    }


def tournament_select(
    population: Population,
    selection_size: int,
    tournament_size: int = 4,
) -> Population:
    genotype_length = population.genes.shape[1]
    selected = Population(selection_size, genotype_length)

    n = len(population.fitnesses)
    num_selected_per_iteration = n // tournament_size
    num_parses = selection_size // num_selected_per_iteration

    for parse_index in range(num_parses):
        population.shuffle()
        winning_indices = np.argmax(
            population.fitnesses.reshape((-1, tournament_size)), axis=1
        )
        winning_indices += np.arange(0, n, tournament_size)
        start = parse_index * num_selected_per_iteration
        end = (parse_index + 1) * num_selected_per_iteration
        selected.genes[start:end, :] = population.genes[winning_indices, :]
        selected.fitnesses[start:end] = population.fitnesses[winning_indices]
    return selected


def select(
    population: Population,
    selection_size: int,
    selection_name: str = "tournament_4",
) -> Population:
    if "tournament" in selection_name:
        tournament_size = int(selection_name.split("_")[-1])
        return tournament_select(population, selection_size, tournament_size)
    if selection_name == "truncation":
        return truncation_select(population, selection_size)
    raise ValueError(f"Invalid selection name: {selection_name}")


class Evolution:
    def __init__(
        self,
        x: np.ndarray,
        fitness_function,
        fitness_function_kwargs: dict,
        feature_intervals: np.ndarray,
        indices_categorical_features: Sequence[int] | None = None,
        plausibility_constraints: Sequence[str | None] | None = None,
        evolution_type: str = "classic",
        population_size: int = 1000,
        n_generations: int = 100,
        mutation_probability: str | float = "inv_mutable_genotype_length",
        num_features_mutation_strength: float = 0.25,
        num_features_mutation_strength_decay: float | None = None,
        num_features_mutation_strength_decay_generations: Sequence[int] | None = None,
        init_temperature: float = 0.8,
        selection_name: str = "tournament_2",
        noisy_evaluations: bool = False,
        verbose: bool = False,
    ):
        self.x = np.asarray(x, dtype=np.float64)
        self.fitness_function = fitness_function
        self.fitness_function_kwargs = dict(fitness_function_kwargs)
        self.feature_intervals = np.asarray(feature_intervals, dtype=object)
        self.indices_categorical_features = (
            None
            if indices_categorical_features is None
            else list(indices_categorical_features)
        )
        self.plausibility_constraints = (
            None if plausibility_constraints is None else list(plausibility_constraints)
        )
        self.evolution_type = str(evolution_type)
        self.population_size = int(population_size)
        self.n_generations = int(n_generations)
        self.mutation_probability = mutation_probability
        self.num_features_mutation_strength = float(num_features_mutation_strength)
        self.num_features_mutation_strength_decay = (
            None
            if num_features_mutation_strength_decay is None
            else float(num_features_mutation_strength_decay)
        )
        self.num_features_mutation_strength_decay_generations = (
            None
            if num_features_mutation_strength_decay_generations is None
            else list(num_features_mutation_strength_decay_generations)
        )
        self.init_temperature = float(init_temperature)
        self.selection_name = str(selection_name)
        self.noisy_evaluations = bool(noisy_evaluations)
        self.verbose = bool(verbose)

        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if self.population_size % 2 != 0:
            raise ValueError("population_size must be even")
        if self.n_generations < 1:
            raise ValueError("n_generations must be >= 1")
        if not (0.0 <= self.init_temperature <= 1.0):
            raise ValueError("init_temperature must be between 0 and 1")
        if self.num_features_mutation_strength <= 0.0:
            raise ValueError("num_features_mutation_strength must be > 0")
        if (
            self.num_features_mutation_strength_decay is not None
            and self.num_features_mutation_strength_decay <= 0.0
        ):
            raise ValueError("num_features_mutation_strength_decay must be > 0")

        if "tournament" in self.selection_name:
            self.tournament_size = int(self.selection_name.split("_")[-1])
            if self.population_size % self.tournament_size != 0:
                raise ValueError(
                    "population_size must be a multiple of the tournament size"
                )
        else:
            self.tournament_size = None

        self.genotype_length = len(self.feature_intervals)
        self.population = Population(self.population_size, self.genotype_length)
        self.elite: np.ndarray | None = None
        self.elite_fitness = -np.inf

        if mutation_probability == "inv_genotype_length":
            self.mutation_probability = 1.0 / self.genotype_length
        elif mutation_probability == "inv_mutable_genotype_length":
            num_unmutable = (
                len([c for c in self.plausibility_constraints if c == "="])
                if self.plausibility_constraints is not None
                else 0
            )
            num_mutable = self.genotype_length - num_unmutable
            if num_mutable <= 0:
                raise ValueError("No mutable features available for mutation")
            self.mutation_probability = 1.0 / num_mutable
        else:
            self.mutation_probability = float(mutation_probability)
            if not (0.0 <= self.mutation_probability <= 1.0):
                raise ValueError("mutation_probability must be between 0 and 1")

        if self.evolution_type == "p+o" and self.noisy_evaluations:
            raise ValueError(
                "evolution_type='p+o' is not compatible with noisy_evaluations=True"
            )

    def _update_elite(self, population: Population) -> None:
        best_fitness_idx = int(np.argmax(population.fitnesses))
        best_fitness = float(population.fitnesses[best_fitness_idx])
        if self.noisy_evaluations or best_fitness > self.elite_fitness:
            self.elite = population.genes[best_fitness_idx, :].copy()
            self.elite_fitness = best_fitness

    def _evaluate(self, genes: np.ndarray) -> np.ndarray:
        return self.fitness_function(
            genes=genes,
            x=self.x,
            feature_intervals=self.feature_intervals,
            indices_categorical_features=self.indices_categorical_features,
            plausibility_constraints=self.plausibility_constraints,
            **self.fitness_function_kwargs,
        )

    def _classic_generation(self, merge_parent_offspring: bool = False) -> None:
        offspring = Population(self.population_size, self.genotype_length)
        offspring.genes[:] = self.population.genes[:]
        offspring.shuffle()
        offspring.genes = crossover(offspring.genes)
        offspring.genes = mutate(
            offspring.genes,
            self.feature_intervals,
            self.indices_categorical_features,
            self.x,
            self.plausibility_constraints,
            mutation_probability=self.mutation_probability,
            num_features_mutation_strength=self.num_features_mutation_strength,
        )
        offspring.fitnesses = self._evaluate(offspring.genes)
        self._update_elite(offspring)
        if merge_parent_offspring:
            self.population.stack(offspring)
        else:
            self.population = offspring
        self.population = select(
            self.population,
            self.population_size,
            selection_name=self.selection_name,
        )

    def _regularized_aging_generation(self, ablate_age_reg: bool = False) -> None:
        num_sub_generations = self.population_size // 2
        for _ in range(num_sub_generations):
            idx_first_parent = one_tournament_round(
                self.population,
                self.tournament_size,
                return_winner_index=True,
            )
            idx_second_parent = one_tournament_round(
                self.population,
                self.tournament_size,
                return_winner_index=True,
            )
            offspring = Population(2, self.genotype_length)
            offspring.genes[:] = self.population.genes[
                [idx_first_parent, idx_second_parent], :
            ]
            offspring.genes = crossover(offspring.genes)
            offspring.genes = mutate(
                offspring.genes,
                self.feature_intervals,
                self.indices_categorical_features,
                self.x,
                self.plausibility_constraints,
                mutation_probability=self.mutation_probability,
                num_features_mutation_strength=self.num_features_mutation_strength,
            )
            offspring.fitnesses = self._evaluate(offspring.genes)
            self._update_elite(offspring)
            self.population.stack(offspring)

            if not ablate_age_reg:
                self.population.delete([0, 1])
            else:
                indices_weakest = np.argsort(self.population.fitnesses)[:2]
                self.population.delete(indices_weakest)

    def run(self) -> None:
        self.population.initialize(
            self.x,
            self.feature_intervals,
            self.indices_categorical_features,
            self.plausibility_constraints,
            self.init_temperature,
        )
        self.population.fitnesses = self._evaluate(self.population.genes)
        self._update_elite(self.population)

        for generation_index in range(self.n_generations):
            if self.num_features_mutation_strength_decay_generations is not None:
                if generation_index in self.num_features_mutation_strength_decay_generations:
                    self.num_features_mutation_strength *= (
                        self.num_features_mutation_strength_decay
                    )

            if self.evolution_type == "classic":
                self._classic_generation(merge_parent_offspring=False)
            elif self.evolution_type == "p+o":
                self._classic_generation(merge_parent_offspring=True)
            elif self.evolution_type == "age_reg":
                self._regularized_aging_generation()
            elif self.evolution_type == "abl_age_reg":
                self._regularized_aging_generation(ablate_age_reg=True)
            else:
                raise ValueError(f"unknown evolution type: {self.evolution_type}")

            if self.verbose:
                print(
                    "generation:",
                    generation_index + 1,
                    "best fitness:",
                    self.elite_fitness,
                    "avg. fitness:",
                    float(np.mean(self.population.fitnesses)),
                )
            if self.population.is_converged():
                break
