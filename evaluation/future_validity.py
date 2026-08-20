from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from utils.registry import get_registry, register


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml_dict(path: str | Path) -> dict:
    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path
    with resolved_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Future config at {resolved_path} must parse to a dictionary")
    return payload


def _build_dataset(cfg: dict):
    import dataset  # noqa: F401

    dataset_cfg = deepcopy(cfg["dataset"])
    name = dataset_cfg.pop("name")
    return get_registry("Dataset")[name](**dataset_cfg)


def _build_preprocess(cfg: dict) -> list:
    import preprocess  # noqa: F401

    preprocess_cfg = list(deepcopy(cfg.get("preprocess", [])))
    if not any(item.get("name", "").lower() == "finalize" for item in preprocess_cfg):
        preprocess_cfg.append({"name": "finalize"})

    registry = get_registry("PreProcess")
    steps = []
    for item in preprocess_cfg:
        item_cfg = deepcopy(item)
        name = item_cfg.pop("name")
        steps.append(registry[name](**item_cfg))
    return steps


def _materialize_train_test(raw_dataset, preprocess_steps: list):
    datasets = [raw_dataset]
    for preprocess_step in preprocess_steps:
        next_datasets = []
        for current_dataset in datasets:
            transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                next_datasets.append(transformed)
        datasets = next_datasets

    trainsets = [dataset for dataset in datasets if getattr(dataset, "trainset", False)]
    testsets = [dataset for dataset in datasets if getattr(dataset, "testset", False)]
    if len(trainsets) > 1 or len(testsets) > 1:
        raise ValueError("Future config must resolve at most one trainset and one testset")
    if trainsets and testsets:
        return trainsets[0], testsets[0]
    if len(datasets) == 1:
        return datasets[0], datasets[0]
    raise ValueError("Could not resolve future train/test datasets after preprocessing")


def _build_and_fit_future_model(cfg: dict):
    # Trigger registrations when this evaluator is used outside Experiment.
    import model  # noqa: F401

    raw_dataset = _build_dataset(cfg)
    preprocess_steps = _build_preprocess(cfg)
    trainset, _ = _materialize_train_test(raw_dataset, preprocess_steps)

    model_cfg = deepcopy(cfg["model"])
    name = model_cfg.pop("name")
    future_model = get_registry("TargetModel")[name](**model_cfg)
    future_model.fit(trainset)
    return future_model


@register("future_validity")
class FutureValidityEvaluation(EvaluationObject):
    def __init__(
        self,
        future_config_path: str | None = None,
        future_config: dict | None = None,
        future_model: Any | None = None,
        desired_class: int | str | None = None,
        prediction_batch_size: int = 512,
        **kwargs,
    ):
        del kwargs
        if future_model is None and future_config_path is None and future_config is None:
            raise ValueError(
                "future_validity requires future_model, future_config_path, or future_config"
            )
        if future_config_path is not None and future_config is not None:
            raise ValueError("Provide only one of future_config_path or future_config")
        if int(prediction_batch_size) < 1:
            raise ValueError("prediction_batch_size must be >= 1")

        self._future_model = future_model
        self._future_config = (
            _load_yaml_dict(future_config_path)
            if future_config_path is not None
            else deepcopy(future_config)
        )
        self._desired_class = desired_class
        self._prediction_batch_size = int(prediction_batch_size)

    def _ensure_future_model(self):
        if self._future_model is None:
            self._future_model = _build_and_fit_future_model(self._future_config)
        return self._future_model

    def _resolve_target_index(self, counterfactuals: DatasetObject) -> int:
        future_model = self._ensure_future_model()
        class_to_index = future_model.get_class_to_index()

        desired_class = self._desired_class
        if desired_class is None:
            try:
                raw_target = counterfactuals.attr("target_prediction_index")
            except AttributeError as error:
                raise ValueError(
                    "desired_class is required when counterfactuals do not provide "
                    "target_prediction_index"
                ) from error
            if isinstance(raw_target, pd.DataFrame):
                if raw_target.shape[1] == 0:
                    raise ValueError("target_prediction_index must contain a column")
                target_values = raw_target.iloc[:, 0].dropna().unique()
            elif isinstance(raw_target, pd.Series):
                target_values = raw_target.dropna().unique()
            else:
                raise TypeError("target_prediction_index must be a Series or DataFrame")
            if len(target_values) != 1:
                raise ValueError(
                    "future_validity requires a single target class per evaluation"
                )
            return int(target_values[0])

        if desired_class not in class_to_index:
            raise ValueError(
                f"desired_class '{desired_class}' is invalid for the future model"
            )
        return int(class_to_index[desired_class])

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        (
            _,
            counterfactual_features,
            evaluation_mask,
            success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)

        selected_mask = evaluation_mask.to_numpy()
        if int(selected_mask.sum()) == 0:
            return pd.DataFrame(
                [{"future_validity": float("nan"), "future_invalidation_rate": float("nan")}]
            )

        future_model = self._ensure_future_model()
        target_index = self._resolve_target_index(counterfactuals)
        selected_counterfactuals = counterfactual_features.loc[selected_mask].copy(
            deep=True
        )
        prediction_batches = []
        for start in range(0, selected_counterfactuals.shape[0], self._prediction_batch_size):
            batch = selected_counterfactuals.iloc[
                start : start + self._prediction_batch_size
            ]
            prediction_batches.append(future_model.get_prediction(batch, proba=False))
        predictions = (
            torch.cat(prediction_batches, dim=0)
            if prediction_batches
            else torch.empty(0)
        )
        prediction_indices = predictions.argmax(dim=1).detach().cpu().numpy()

        selected_success = success_mask.loc[evaluation_mask.to_numpy()].to_numpy()
        future_hits = (prediction_indices == target_index) & selected_success
        future_validity = float(np.mean(future_hits.astype(np.float64)))
        return pd.DataFrame(
            [
                {
                    "future_validity": future_validity,
                    "future_invalidation_rate": float(1.0 - future_validity),
                }
            ]
        )
