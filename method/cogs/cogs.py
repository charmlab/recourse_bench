from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.cogs.support import (
    BlackBoxModelTypes,
    Evolution,
    RecourseModelAdapter,
    ensure_supported_target_model,
    gower_fitness_function,
    resolve_feature_groups,
    resolve_target_classes,
    validate_counterfactuals,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("cogs")
class CogsMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        evolution_type: str = "classic",
        population_size: int = 1000,
        n_generations: int = 100,
        mutation_probability: str | float = "inv_mutable_genotype_length",
        num_features_mutation_strength: float = 0.25,
        num_features_mutation_strength_decay: float | None = None,
        num_features_mutation_strength_decay_generations: list[int] | None = None,
        init_temperature: float = 0.8,
        selection_name: str = "tournament_2",
        noisy_evaluations: bool = False,
        fitness_name: str = "gower",
        apply_fixes: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        ensure_supported_target_model(target_model, BlackBoxModelTypes, "CogsMethod")
        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class

        self._evolution_type = str(evolution_type)
        self._population_size = int(population_size)
        self._n_generations = int(n_generations)
        self._mutation_probability = mutation_probability
        self._num_features_mutation_strength = float(num_features_mutation_strength)
        self._num_features_mutation_strength_decay = (
            None
            if num_features_mutation_strength_decay is None
            else float(num_features_mutation_strength_decay)
        )
        self._num_features_mutation_strength_decay_generations = (
            None
            if num_features_mutation_strength_decay_generations is None
            else [int(value) for value in num_features_mutation_strength_decay_generations]
        )
        self._init_temperature = float(init_temperature)
        self._selection_name = str(selection_name)
        self._noisy_evaluations = bool(noisy_evaluations)
        self._fitness_name = str(fitness_name).lower()
        self._apply_fixes = bool(apply_fixes)
        self._verbose = bool(verbose)

        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")
        if self._population_size < 2:
            raise ValueError("population_size must be >= 2")
        if self._population_size % 2 != 0:
            raise ValueError("population_size must be even")
        if self._n_generations < 1:
            raise ValueError("n_generations must be >= 1")
        if self._num_features_mutation_strength <= 0:
            raise ValueError("num_features_mutation_strength must be > 0")
        if not (0.0 <= self._init_temperature <= 1.0):
            raise ValueError("init_temperature must be between 0 and 1")
        if (
            self._num_features_mutation_strength_decay is not None
            and self._num_features_mutation_strength_decay <= 0
        ):
            raise ValueError("num_features_mutation_strength_decay must be > 0")
        if self._fitness_name != "gower":
            raise ValueError("fitness_name must currently be 'gower'")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for CogsMethod.fit()")

        with seed_context(self._seed):
            feature_df = trainset.get(target=False)
            try:
                train_array = feature_df.to_numpy(dtype=np.float64)
            except ValueError as error:
                raise ValueError(
                    "CogsMethod requires fully numeric input features; encode categorical "
                    "features before use"
                ) from error

            if np.isnan(train_array).any():
                raise ValueError("CogsMethod does not support NaN values in trainset")

            feature_groups = resolve_feature_groups(trainset)
            self._feature_names = list(feature_groups.feature_names)
            self._adapter = RecourseModelAdapter(self._target_model, self._feature_names)
            self._indices_categorical_features = [
                self._feature_names.index(feature_name)
                for feature_name in feature_groups.categorical
            ]
            self._plausibility_constraints = list(
                feature_groups.plausibility_constraints
            )
            self._feature_intervals: list[object] = []

            for feature_index, feature_name in enumerate(self._feature_names):
                feature_values = train_array[:, feature_index]
                if feature_name in feature_groups.categorical:
                    interval = np.unique(feature_values)
                else:
                    interval = (
                        float(np.min(feature_values)),
                        float(np.max(feature_values)),
                    )
                self._feature_intervals.append(interval)

            self._feature_intervals = np.asarray(self._feature_intervals, dtype=object)
            self._is_trained = True

    def _search_counterfactual(
        self,
        factual: np.ndarray,
        desired_class: int | str,
    ) -> np.ndarray | None:
        evolution = Evolution(
            x=factual,
            fitness_function=gower_fitness_function,
            fitness_function_kwargs={
                "blackbox": self._adapter,
                "desired_class": desired_class,
                "apply_fixes": self._apply_fixes,
            },
            feature_intervals=self._feature_intervals,
            indices_categorical_features=self._indices_categorical_features,
            plausibility_constraints=self._plausibility_constraints,
            evolution_type=self._evolution_type,
            population_size=self._population_size,
            n_generations=self._n_generations,
            mutation_probability=self._mutation_probability,
            num_features_mutation_strength=self._num_features_mutation_strength,
            num_features_mutation_strength_decay=self._num_features_mutation_strength_decay,
            num_features_mutation_strength_decay_generations=self._num_features_mutation_strength_decay_generations,
            init_temperature=self._init_temperature,
            selection_name=self._selection_name,
            noisy_evaluations=self._noisy_evaluations,
            verbose=self._verbose,
        )
        evolution.run()
        elite = evolution.elite
        if elite is None:
            return None
        candidate = np.asarray(elite, dtype=np.float64).copy()
        predicted = self._adapter.predict(candidate.reshape(1, -1))[0]
        if predicted != desired_class:
            return None
        return candidate

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("Input factuals cannot contain NaN")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        with seed_context(self._seed):
            original_predictions = self._adapter.predict_label_indices(factuals)
            desired_classes = resolve_target_classes(
                self._target_model,
                original_predictions,
                self._desired_class,
            )

            rows: list[np.ndarray] = []
            for row_index, (_, row) in enumerate(factuals.iterrows()):
                factual = row.to_numpy(dtype=np.float64)
                desired_class = desired_classes[row_index]
                candidate = self._search_counterfactual(factual, desired_class)
                if candidate is None:
                    rows.append(np.full(len(self._feature_names), np.nan, dtype=np.float64))
                else:
                    rows.append(candidate)

        candidates = pd.DataFrame(rows, index=factuals.index, columns=self._feature_names)
        return validate_counterfactuals(
            target_model=self._target_model,
            factuals=factuals,
            candidates=candidates,
            desired_class=self._desired_class,
        )
