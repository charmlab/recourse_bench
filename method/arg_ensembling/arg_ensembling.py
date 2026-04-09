from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from method.arg_ensembling.support import (
    build_baf_program,
    build_model_adapters,
    compute_model_accuracy_scores,
    compute_model_simplicity_scores,
    ensure_class_mapping_alignment,
    ensure_supported_target_model,
    instantiate_models_from_configs,
    nearest_neighbor_counterfactual,
    predict_label_indices,
    select_best_accepted_counterfactual,
    solve_argumentative_extension,
    validate_ensemble_models,
    validate_score_vector,
    BlackBoxModelTypes,
)
from method.method_object import MethodObject
from model.model_object import ModelObject
from utils.registry import register
from utils.seed import seed_context


@register("arg_ensembling")
class ArgEnsemblingMethod(MethodObject):
    def __init__(
        self,
        target_model: ModelObject,
        seed: int | None = None,
        device: str = "cpu",
        desired_class: int | str | None = None,
        ensemble_models: list[ModelObject] | None = None,
        ensemble_model_configs: list[dict] | None = None,
        semantics: str = "s",
        preference_mode: str = "none",
        accuracy_scores: list[float] | None = None,
        simplicity_scores: list[float] | None = None,
        **kwargs,
    ):
        ensure_supported_target_model(
            target_model,
            BlackBoxModelTypes,
            "ArgEnsemblingMethod",
        )

        self._target_model = target_model
        self._seed = seed
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._desired_class = desired_class
        self._semantics = str(semantics).lower()
        self._preference_mode = str(preference_mode).lower()
        self._accuracy_scores_cfg = (
            None if accuracy_scores is None else [float(value) for value in accuracy_scores]
        )
        self._simplicity_scores_cfg = (
            None
            if simplicity_scores is None
            else [float(value) for value in simplicity_scores]
        )

        if ensemble_models and ensemble_model_configs:
            raise ValueError(
                "Provide either ensemble_models or ensemble_model_configs, not both"
            )
        if self._semantics not in {"s", "d"}:
            raise ValueError("semantics must be 's' or 'd'")
        if self._preference_mode not in {
            "none",
            "accuracy",
            "simplicity",
            "accuracy_simplicity",
        }:
            raise ValueError(
                "preference_mode must be one of "
                "['none', 'accuracy', 'simplicity', 'accuracy_simplicity']"
            )
        if self._device != self._target_model._device:
            raise ValueError("Method device must match target model device")

        instantiated_models = (
            instantiate_models_from_configs(
                ensemble_model_configs or [],
                default_device=self._device,
            )
            if ensemble_model_configs is not None
            else list(ensemble_models or [])
        )
        self._ensemble_models = [target_model, *instantiated_models]
        validate_ensemble_models(
            self._ensemble_models,
            device=self._device,
            method_name="ArgEnsemblingMethod",
        )

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ArgEnsemblingMethod.fit()")

        with seed_context(self._seed):
            for model in self._ensemble_models:
                if not model._is_trained:
                    model.fit(trainset)

            validate_ensemble_models(
                self._ensemble_models,
                device=self._device,
                method_name="ArgEnsemblingMethod",
            )
            ensure_class_mapping_alignment(self._ensemble_models)

            class_to_index = self._target_model.get_class_to_index()
            if (
                self._desired_class is not None
                and self._desired_class not in class_to_index
            ):
                raise ValueError("desired_class is invalid for the trained target model")

            self._feature_names = list(trainset.get(target=False).columns)
            self._train_features = (
                trainset.get(target=False)
                .loc[:, self._feature_names]
                .copy(deep=True)
            )
            self._model_adapters = build_model_adapters(
                self._ensemble_models,
                self._feature_names,
            )
            self._train_prediction_indices = [
                predict_label_indices(adapter, self._train_features)
                for adapter in self._model_adapters
            ]
            self._desired_index = (
                None
                if self._desired_class is None
                else int(class_to_index[self._desired_class])
            )

            target_series = trainset.get(target=True).iloc[:, 0]
            encoded_target = np.asarray(
                [class_to_index[value] for value in target_series.tolist()],
                dtype=np.int64,
            )
            self._accuracy_scores = validate_score_vector(
                self._accuracy_scores_cfg,
                expected_size=len(self._ensemble_models),
                name="accuracy_scores",
            )
            if self._accuracy_scores is None:
                self._accuracy_scores = compute_model_accuracy_scores(
                    self._model_adapters,
                    self._train_features,
                    encoded_target,
                )

            self._simplicity_scores = validate_score_vector(
                self._simplicity_scores_cfg,
                expected_size=len(self._ensemble_models),
                name="simplicity_scores",
            )
            if self._simplicity_scores is None:
                self._simplicity_scores = compute_model_simplicity_scores(
                    self._ensemble_models
                )

            self._is_trained = True

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Method is not trained")
        if factuals.isna().any(axis=None):
            raise ValueError("factuals cannot contain NaN")

        factuals = factuals.loc[:, self._feature_names].copy(deep=True)
        rows: list[pd.Series] = []

        with seed_context(self._seed):
            for _, factual in factuals.iterrows():
                counterfactual = self._solve_instance(factual)
                if counterfactual is None:
                    rows.append(pd.Series(np.nan, index=self._feature_names))
                else:
                    rows.append(counterfactual.reindex(self._feature_names))

        return pd.DataFrame(rows, index=factuals.index, columns=self._feature_names)

    def _solve_instance(self, factual: pd.Series) -> pd.Series | None:
        factual_df = factual.to_frame().T.reindex(columns=self._feature_names)
        factual_predictions = np.asarray(
            [
                int(predict_label_indices(adapter, factual_df)[0])
                for adapter in self._model_adapters
            ],
            dtype=np.int64,
        )

        candidate_rows: list[pd.Series] = []
        for model_index, train_predictions in enumerate(self._train_prediction_indices):
            candidate = nearest_neighbor_counterfactual(
                factual=factual,
                train_features=self._train_features,
                train_predictions=train_predictions,
                original_prediction=int(factual_predictions[model_index]),
                desired_prediction=self._desired_index,
            )
            if candidate is None:
                return None
            candidate_rows.append(candidate.reindex(self._feature_names))

        candidate_frame = pd.DataFrame(candidate_rows, columns=self._feature_names)
        candidate_predictions = np.vstack(
            [
                predict_label_indices(adapter, candidate_frame)
                for adapter in self._model_adapters
            ]
        )

        extension = solve_argumentative_extension(
            build_baf_program(
                factual_predictions=factual_predictions,
                counterfactual_predictions=candidate_predictions,
                accuracy_scores=self._accuracy_scores,
                simplicity_scores=self._simplicity_scores,
                semantics=self._semantics,
                preference_mode=self._preference_mode,
            ),
            factual_predictions=factual_predictions,
        )
        if extension is None or not extension.ce_indices:
            return None

        best_counterfactual = select_best_accepted_counterfactual(
            factual=factual,
            counterfactuals=candidate_frame,
            accepted_ce_indices=extension.ce_indices,
        )
        if best_counterfactual is None:
            return None
        if not self._is_valid_for_target_model(
            factual=factual_df,
            counterfactual=best_counterfactual.to_frame().T,
        ):
            return None
        return best_counterfactual

    def _is_valid_for_target_model(
        self,
        factual: pd.DataFrame,
        counterfactual: pd.DataFrame,
    ) -> bool:
        adapter = self._model_adapters[0]
        original_prediction = int(predict_label_indices(adapter, factual)[0])
        candidate_prediction = int(predict_label_indices(adapter, counterfactual)[0])
        if self._desired_index is not None:
            return candidate_prediction == self._desired_index
        return candidate_prediction != original_prediction
