from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dataset.dataset_object import DatasetObject
from dataset.german.german import GermanDataset
from dataset.german_roar.german_roar import GermanRoarDataset
from dataset.sba_roar.sba_roar import SbaRoarDataset
from dataset.student_roar.student_roar import StudentRoarDataset
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import resolve_evaluation_inputs
from evaluation.validity import ValidityEvaluation
from experiment.utils import write_reproduction_report
from method.roar.roar import RoarMethod
from model.model_object import ModelObject, process_nan
from preprocess.common import EncodePreProcess, FinalizePreProcess, ScalePreProcess
from preprocess.preprocess_object import PreProcessObject
from utils.seed import seed_context

SEED = 0
SHUFFLE_SEED = 1
TOTAL_FOLDS = 5
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_LAMBDA_GRID = [round(value, 1) for value in np.arange(0.1, 1.1, 0.1)]
PAPER_METRICS = {
    ("correction", "linear", "l1"): {
        "cost": {"mean": 3.13, "std": 0.32},
        "m1_validity": {"mean": 1.00, "std": 0.00},
        "m2_validity": {"mean": 0.94, "std": 0.08},
    },
    ("temporal", "linear", "l1"): {
        "cost": {"mean": 3.14, "std": 0.25},
        "m1_validity": {"mean": 0.99, "std": 0.01},
        "m2_validity": {"mean": 0.98, "std": 0.02},
    },
    ("geospatial", "linear", "l1"): {
        "cost": {"mean": 10.88, "std": 1.67},
        "m1_validity": {"mean": 1.0, "std": 0.0},
        "m2_validity": {"mean": 0.67, "std": 0.19},
    },
    ("correction", "linear", "pfc"): {
        "cost": {"mean": 0.36, "std": 0.08},
        "m1_validity": {"mean": 1.00, "std": 0.00},
        "m2_validity": {"mean": 1.00, "std": 0.00},
    },
    ("temporal", "linear", "pfc"): {
        "cost": {"mean": 0.44, "std": 0.12},
        "m1_validity": {"mean": 0.99, "std": 0.01},
        "m2_validity": {"mean": 0.98, "std": 0.01},
    },
    ("geospatial", "linear", "pfc"): {
        "cost": {"mean": 1.2, "std": 0.1},
        "m1_validity": {"mean": 1.0, "std": 0.0},
        "m2_validity": {"mean": 0.91, "std": 0.07},
    },
    ("correction", "mlp", "l1"): {
        "cost": {"mean": 1.83, "std": 0.19},
        "m1_validity": {"mean": 0.78, "std": 0.06},
        "m2_validity": {"mean": 0.72, "std": 0.10},
    },
    ("temporal", "mlp", "l1"): {
        "cost": {"mean": 4.90, "std": 0.24},
        "m1_validity": {"mean": 0.98, "std": 0.02},
        "m2_validity": {"mean": 0.97, "std": 0.02},
    },
    ("geospatial", "mlp", "l1"): {
        "cost": {"mean": 21.05, "std": 3.58},
        "m1_validity": {"mean": 1.0, "std": 0.0},
        "m2_validity": {"mean": 0.97, "std": 0.03},
    },
    ("correction", "mlp", "pfc"): {
        "cost": {"mean": 0.64, "std": 0.08},
        "m1_validity": {"mean": 0.85, "std": 0.07},
        "m2_validity": {"mean": 0.82, "std": 0.05},
    },
    ("temporal", "mlp", "pfc"): {
        "cost": {"mean": 0.37, "std": 0.07},
        "m1_validity": {"mean": 0.99, "std": 0.01},
        "m2_validity": {"mean": 0.99, "std": 0.00},
    },
    ("geospatial", "mlp", "pfc"): {
        "cost": {"mean": 1.66, "std": 0.21},
        "m1_validity": {"mean": 1.0, "std": 0.0},
        "m2_validity": {"mean": 0.97, "std": 0.04},
    },
}
PAPER_METRIC_TO_AGGREGATE_KEY = {
    "cost": "cost",
    "m1_validity": "M1_validity",
    "m2_validity": "M2_validity",
}
REPORT_PATH = Path(__file__).with_name("reproduction_report.json")


@dataclass(frozen=True)
class TrialBundle:
    current_train: DatasetObject
    current_test: DatasetObject
    future_train: DatasetObject
    future_test: DatasetObject


class ShuffleRowsPreProcess(PreProcessObject):
    def __init__(self, seed: int | None = SHUFFLE_SEED, **kwargs):
        self._seed = seed

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            df = input.snapshot().sample(frac=1, random_state=self._seed)
            input.update("shuffled", True, df=df.copy(deep=True))
            return input


class FilterRowsPreProcess(PreProcessObject):
    def __init__(
        self,
        seed: int | None = None,
        column: str | None = None,
        equals: object | None = None,
        less_than: float | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._column = str(column) if column is not None else None
        self._equals = equals
        self._less_than = less_than
        if self._column is None:
            raise ValueError("column is required")
        if self._equals is None and self._less_than is None:
            raise ValueError("Either equals or less_than must be provided")

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            df = input.snapshot()
            if self._column not in df.columns:
                raise KeyError(f"Unknown filter column: {self._column}")
            mask = pd.Series(True, index=df.index, dtype=bool)
            if self._equals is not None:
                mask &= df[self._column] == self._equals
            if self._less_than is not None:
                mask &= df[self._column] < self._less_than
            input.update("row_filter", {self._column: True}, df=df.loc[mask].copy(deep=True))
            return input


class DropColumnsPreProcess(PreProcessObject):
    def __init__(
        self,
        seed: int | None = None,
        columns: list[str] | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._columns = list(columns or [])

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            df = input.snapshot()
            drop_columns = [column for column in self._columns if column in df.columns]
            if not drop_columns:
                return input
            input.update("dropped_columns", drop_columns, df=df.drop(columns=drop_columns))
            return input


class NormalizeCategoricalValuesPreProcess(PreProcessObject):
    def __init__(self, seed: int | None = None, columns: list[str] | None = None, **kwargs):
        self._seed = seed
        self._columns = list(columns or [])

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            df = input.snapshot().copy(deep=True)
            raw_feature_type = input.attr("raw_feature_type")

            if self._columns:
                columns = [column for column in self._columns if column in df.columns]
            else:
                columns = [
                    column
                    for column, feature_type in raw_feature_type.items()
                    if column in df.columns and str(feature_type).lower() == "categorical"
                ]

            for column in columns:
                df[column] = df[column].astype(str)

            input.update("normalized_categorical_values", columns, df=df)
            return input


class AlignColumnsPreProcess(PreProcessObject):
    def __init__(
        self,
        seed: int | None = None,
        reference: DatasetObject | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._reference = reference
        if self._reference is None:
            raise ValueError("reference dataset is required")

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            ref_df = self._reference.snapshot()
            df = input.snapshot()
            target_column = input.target_column
            ref_features = [column for column in ref_df.columns if column != target_column]
            current_features = [column for column in df.columns if column != target_column]

            aligned = df.copy(deep=True)
            for column in ref_features:
                if column not in aligned.columns:
                    aligned[column] = 0.0
            extra_columns = [
                column for column in current_features if column not in set(ref_features)
            ]
            if extra_columns:
                aligned = aligned.drop(columns=extra_columns)
            aligned = aligned.loc[:, ref_features + [target_column]].copy(deep=True)
            input.update("aligned_columns", ref_features, df=aligned)
            return input


class FoldChunkSplitPreProcess(PreProcessObject):
    def __init__(
        self,
        seed: int | None = None,
        fold: int = 0,
        total_folds: int = TOTAL_FOLDS,
        **kwargs,
    ):
        self._seed = seed
        self._fold = int(fold)
        self._total_folds = int(total_folds)
        if self._total_folds < 2:
            raise ValueError("total_folds must be >= 2")
        if self._fold < 0 or self._fold >= self._total_folds:
            raise ValueError("fold is out of range")

    def transform(self, input: DatasetObject) -> tuple[DatasetObject, DatasetObject]:
        with seed_context(self._seed):
            df = input.snapshot()
            chunks = []
            for index in range(self._total_folds):
                start = int(index / self._total_folds * len(df))
                end = int((index + 1) / self._total_folds * len(df))
                chunks.append(df.iloc[start:end].copy(deep=True))

            test_df = chunks.pop(self._fold)
            train_df = pd.concat(chunks).copy(deep=True)

            trainset = input
            testset = input.clone()
            trainset.update("trainset", True, df=train_df)
            testset.update("testset", True, df=test_df)
            return trainset, testset


class RoarLogisticModel(ModelObject):
    def __init__(self, seed: int | None = SEED, device: str = "cpu", **kwargs):
        self._seed = seed
        self._device = str(device).lower()
        self._need_grad = False
        self._is_trained = False
        self._sklearn_model: LogisticRegression | None = None
        self._model: torch.nn.Linear | None = None
        if self._device != "cpu":
            raise ValueError("RoarLogisticModel only supports cpu")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required")

        with seed_context(self._seed):
            X, labels, _ = self.extract_training_data(trainset)
            self._sklearn_model = LogisticRegression().fit(X, labels.cpu().numpy())
            linear = torch.nn.Linear(X.shape[1], 1)
            with torch.no_grad():
                linear.weight.copy_(
                    torch.tensor(self._sklearn_model.coef_, dtype=torch.float32)
                )
                linear.bias.copy_(
                    torch.tensor(self._sklearn_model.intercept_, dtype=torch.float32)
                )
            self._model = linear
            self._is_trained = True

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained or self._sklearn_model is None:
            raise RuntimeError("Target model is not trained")
        probabilities = torch.tensor(
            self._sklearn_model.predict_proba(X), dtype=torch.float32
        )
        if proba:
            return probabilities
        indices = probabilities.argmax(dim=1)
        return torch.nn.functional.one_hot(indices, num_classes=2).to(torch.float32)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Target model is not trained")
        logits = self._model(X.to(torch.float32))
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        return torch.cat([-logits, logits], dim=1)


class RoarNeuralModel(ModelObject):
    def __init__(
        self,
        seed: int | None = SEED,
        device: str = "cpu",
        epochs: int = DEFAULT_EPOCHS,
        learning_rate: float = DEFAULT_LR,
        **kwargs,
    ):
        self._seed = seed
        self._device = str(device).lower()
        self._need_grad = True
        self._is_trained = False
        self._epochs = int(epochs)
        self._learning_rate = float(learning_rate)
        self._model: torch.nn.Sequential | None = None
        if self._device != "cpu":
            raise ValueError("RoarNeuralModel only supports cpu")

    def _build_model(self, input_dim: int) -> torch.nn.Sequential:
        torch.manual_seed(int(self._seed or 0))
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 1),
            torch.nn.Sigmoid(),
        )

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required")

        with seed_context(self._seed):
            X, labels, _ = self.extract_training_data(trainset)
            self._model = self._build_model(X.shape[1])
            X_tensor = torch.tensor(X.to_numpy(dtype="float32"), dtype=torch.float32)
            y_tensor = labels.to(dtype=torch.float32)
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)

            self._model.train()
            for _ in range(self._epochs):
                optimizer.zero_grad()
                y_pred = self._model(X_tensor)[:, 0]
                loss = criterion(y_pred, y_tensor)
                loss.backward()
                optimizer.step()

            self._model.eval()
            self._is_trained = True

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Target model is not trained")
        self._model.eval()
        X_tensor = torch.tensor(X.to_numpy(dtype="float32"), dtype=torch.float32)
        with torch.no_grad():
            class1 = self._model(X_tensor)
            class0 = 1 - class1
            probabilities = torch.cat([class0, class1], dim=1)
        if proba:
            return probabilities.detach().cpu()
        indices = probabilities.argmax(dim=1)
        return torch.nn.functional.one_hot(indices, num_classes=2).to(torch.float32)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Target model is not trained")
        self._model.eval()
        with torch.no_grad():
            class1 = self._model(X.to(torch.float32))
            class0 = 1 - class1
            return torch.cat([class0, class1], dim=1)


class RoarCostEvaluation(EvaluationObject):
    def __init__(
        self,
        cost_name: str,
        feature_weights: np.ndarray | None = None,
        **kwargs,
    ):
        resolved = str(cost_name).lower()
        if resolved not in {"l1", "pfc"}:
            raise ValueError("cost_name must be 'l1' or 'pfc'")
        self._cost_name = resolved
        self._feature_weights = None if feature_weights is None else np.asarray(feature_weights, dtype=np.float32)

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        factual_features, counterfactual_features, evaluation_mask, _ = resolve_evaluation_inputs(
            factuals, counterfactuals
        )
        selected = evaluation_mask.to_numpy()
        if int(selected.sum()) == 0:
            return pd.DataFrame([{"cost": float("nan")}])

        factual_eval = factual_features.loc[selected].to_numpy(dtype=np.float32, copy=True)
        counterfactual_eval = counterfactual_features.loc[selected].to_numpy(
            dtype=np.float32, copy=True
        )
        deltas = np.abs(counterfactual_eval - factual_eval)
        if self._cost_name == "l1":
            row_cost = deltas.sum(axis=1)
        else:
            if self._feature_weights is None:
                raise ValueError("feature_weights are required for pfc cost")
            if self._feature_weights.shape[0] != deltas.shape[1]:
                raise ValueError("feature_weights must match the finalized feature width")
            row_cost = deltas @ self._feature_weights
        return pd.DataFrame([{"cost": float(np.mean(row_cost))}])


def _normalize_data_name(value: str) -> str:
    normalized = str(value).lower()
    aliases = {
        "all": "all",
        "correction": "correction",
        "german": "correction",
        "temporal": "temporal",
        "sba": "temporal",
        "geospatial": "geospatial",
        "student": "geospatial",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported data option: {value}")
    return aliases[normalized]


def _normalize_model_name(value: str) -> str:
    normalized = str(value).lower()
    aliases = {
        "all": "all",
        "lr": "linear",
        "linear": "linear",
        "nn": "mlp",
        "mlp": "mlp",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported model option: {value}")
    return aliases[normalized]


def _normalize_cost_name(value: str) -> str:
    normalized = str(value).lower()
    if normalized not in {"all", "l1", "pfc"}:
        raise ValueError(f"Unsupported cost option: {value}")
    return normalized


def _expand_options(data_name: str, model_name: str, cost_name: str) -> list[tuple[str, str, str]]:
    data_options = ["correction", "temporal", "geospatial"] if data_name == "all" else [data_name]
    model_options = ["linear", "mlp"] if model_name == "all" else [model_name]
    cost_options = ["l1", "pfc"] if cost_name == "all" else [cost_name]
    return [
        (data_option, model_option, cost_option)
        for data_option in data_options
        for model_option in model_options
        for cost_option in cost_options
    ]


def _simulate_pairwise_feature_costs(
    n_feat: int,
    n_cmps: int = 100,
    seed: int = SEED,
    alpha: float = 0.01,
) -> np.ndarray:
    if n_feat < 1:
        raise ValueError("n_feat must be >= 1")
    if n_feat == 1:
        return np.zeros(1, dtype=np.float32)

    rng = np.random.default_rng(seed)
    comparisons: list[tuple[int, int]] = []
    for i in range(n_feat):
        for j in range(n_feat):
            if i == j:
                continue
            for _ in range(int(n_cmps / 2)):
                if rng.uniform() < 0.5:
                    comparisons.append((i, j))
                else:
                    comparisons.append((j, i))

    def objective(free_params: np.ndarray) -> float:
        theta = np.concatenate([np.array([0.0]), free_params.astype(np.float64)])
        loss = 0.0
        for winner, loser in comparisons:
            loss += np.logaddexp(0.0, theta[loser] - theta[winner])
        loss += float(alpha) * float(np.sum(theta**2))
        return float(loss)

    from scipy.optimize import minimize

    result = minimize(objective, x0=np.zeros(n_feat - 1, dtype=np.float64), method="L-BFGS-B")
    theta = np.concatenate([np.array([0.0]), result.x.astype(np.float64)])
    theta = theta - float(theta.min())
    return theta.astype(np.float32)


def _apply_steps(dataset: DatasetObject, steps: list[PreProcessObject]) -> DatasetObject:
    current = dataset
    for step in steps:
        transformed = step.transform(current)
        if isinstance(transformed, tuple):
            raise TypeError("Tuple-producing preprocess is not supported in _apply_steps")
        current = transformed
    return current


def _freeze_dataset(dataset: DatasetObject) -> DatasetObject:
    return FinalizePreProcess().transform(dataset)


def _subset_frozen_dataset(
    dataset: DatasetObject,
    row_mask: np.ndarray,
) -> DatasetObject:
    combined = pd.concat([dataset.get(target=False), dataset.get(target=True)], axis=1)
    subset = combined.iloc[np.asarray(row_mask, dtype=int)].copy(deep=True)
    clone = dataset.clone()
    clone.update("subset", True, df=subset)
    clone.freeze()
    return clone


def _prepare_trial_bundle(data_name: str, fold: int) -> TrialBundle:
    current_raw: DatasetObject
    future_raw: DatasetObject
    if data_name == "correction":
        current_raw = GermanDataset()
        future_raw = GermanRoarDataset()
    elif data_name == "temporal":
        current_raw = SbaRoarDataset()
        future_raw = SbaRoarDataset()
    elif data_name == "geospatial":
        current_raw = StudentRoarDataset()
        future_raw = StudentRoarDataset()
    else:
        raise ValueError(f"Unsupported dataset option: {data_name}")

    shuffle = ShuffleRowsPreProcess(seed=SHUFFLE_SEED)
    current = shuffle.transform(current_raw)
    future = shuffle.transform(future_raw)

    if data_name == "correction":
        current_ref = current.clone()
        current = ScalePreProcess(scaling="standardize", range=False).transform(current)
        future = ScalePreProcess(
            scaling="standardize",
            range=False,
            refset=current_ref,
        ).transform(future)
        current = EncodePreProcess(encoding="onehot").transform(current)
        future = EncodePreProcess(encoding="onehot").transform(future)
        future = AlignColumnsPreProcess(reference=current).transform(future)
    elif data_name == "temporal":
        current = NormalizeCategoricalValuesPreProcess().transform(current)
        future = NormalizeCategoricalValuesPreProcess().transform(future)
        current = EncodePreProcess(encoding="onehot").transform(current)
        future = EncodePreProcess(encoding="onehot").transform(future)
        current = FilterRowsPreProcess(column="ApprovalFY", less_than=2006).transform(current)
        current_ref = current.clone()
        current = ScalePreProcess(scaling="standardize", range=False).transform(current)
        future = AlignColumnsPreProcess(reference=current_ref).transform(future)
        future = ScalePreProcess(
            scaling="standardize",
            range=False,
            refset=current_ref,
        ).transform(future)
        future = AlignColumnsPreProcess(reference=current).transform(future)
    else:
        current = EncodePreProcess(encoding="onehot").transform(current)
        future = EncodePreProcess(encoding="onehot").transform(future)
        current = FilterRowsPreProcess(column="school_cat_GP", equals=1.0).transform(current)
        current_ref = current.clone()
        current = ScalePreProcess(scaling="standardize", range=False).transform(current)
        future = AlignColumnsPreProcess(reference=current_ref).transform(future)
        future = ScalePreProcess(
            scaling="standardize",
            range=False,
            refset=current_ref,
        ).transform(future)
        drop_school = DropColumnsPreProcess(columns=["school_cat_GP", "school_cat_MS"])
        current = drop_school.transform(current)
        future = drop_school.transform(future)
        future = AlignColumnsPreProcess(reference=current).transform(future)

    splitter = FoldChunkSplitPreProcess(fold=fold, total_folds=TOTAL_FOLDS)
    current_train, current_test = splitter.transform(current)
    future_train, future_test = splitter.transform(future)
    return TrialBundle(
        current_train=_freeze_dataset(current_train),
        current_test=_freeze_dataset(current_test),
        future_train=_freeze_dataset(future_train),
        future_test=_freeze_dataset(future_test),
    )


def _build_model(model_name: str):
    if model_name == "linear":
        return RoarLogisticModel(seed=SEED, device="cpu")
    if model_name == "mlp":
        return RoarNeuralModel(
            seed=SEED,
            device="cpu",
            epochs=DEFAULT_EPOCHS,
            learning_rate=DEFAULT_LR,
        )
    raise ValueError(f"Unsupported model option: {model_name}")


def _compute_model_metrics(model: ModelObject, dataset: DatasetObject) -> tuple[float, float]:
    probabilities = model.get_prediction(dataset.get(target=False), proba=True).detach().cpu().numpy()
    targets = dataset.get(target=True).iloc[:, 0].to_numpy(dtype=int, copy=True)
    preds = probabilities.argmax(axis=1)
    accuracy = float(np.mean(preds == targets))
    auc = float(roc_auc_score(targets, probabilities[:, 1]))
    return accuracy, auc


def _recourse_needed_indices(model: ModelObject, dataset: DatasetObject, desired_class: int = 1) -> np.ndarray:
    predictions = model.predict(dataset).argmax(dim=1).detach().cpu().numpy()
    target_index = int(model.get_class_to_index()[desired_class])
    return np.where(predictions != target_index)[0]


def _build_roar_method(
    model: ModelObject,
    trainset: DatasetObject,
    lambda_value: float,
    feature_costs: np.ndarray | None,
    show_progress: bool,
    progress_desc: str | None,
) -> RoarMethod:
    method = RoarMethod(
        target_model=model,
        seed=SEED,
        device="cpu",
        desired_class=1,
        lr=DEFAULT_LR,
        lambda_=float(lambda_value),
        delta_max=0.1,
        norm=1,
        max_minutes=0.5,
        loss_type="BCE",
        loss_threshold=1e-4,
        lime_seed=SEED,
        discretize_continuous=False,
        enforce_encoding=False,
        sample_around_instance=False,
        feature_cost=None if feature_costs is None else feature_costs.tolist(),
        return_best_effort=True,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    method.fit(trainset)
    return method


def _choose_lambda(
    model: ModelObject,
    trainset: DatasetObject,
    feature_costs: np.ndarray | None,
    factual_limit: int | None,
    show_progress: bool,
    progress_prefix: str,
) -> float:
    candidate_indices = _recourse_needed_indices(model, trainset, desired_class=1)
    if factual_limit is not None:
        candidate_indices = candidate_indices[: max(0, int(factual_limit))]
    factual_subset = _subset_frozen_dataset(trainset, candidate_indices)
    evaluator = ValidityEvaluation()

    best_validity = 0.0
    best_lambda = float(DEFAULT_LAMBDA_GRID[0])
    for lambda_value in DEFAULT_LAMBDA_GRID:
        method = _build_roar_method(
            model=model,
            trainset=trainset,
            lambda_value=float(lambda_value),
            feature_costs=feature_costs,
            show_progress=show_progress,
            progress_desc=f"{progress_prefix} lambda={lambda_value:.1f}",
        )
        counterfactuals = method.predict(factual_subset)
        validity = float(
            evaluator.evaluate(factual_subset, counterfactuals).iloc[0]["validity"]
        )
        if np.isnan(validity):
            validity = 0.0
        if validity >= best_validity:
            best_validity = validity
            best_lambda = float(lambda_value)
        else:
            break
    return best_lambda


def _future_validity(
    factuals: DatasetObject,
    counterfactuals: DatasetObject,
    future_model: ModelObject,
    desired_class: int = 1,
) -> float:
    _, counterfactual_features, evaluation_mask, _ = resolve_evaluation_inputs(
        factuals, counterfactuals
    )
    selected = evaluation_mask.to_numpy()
    if int(selected.sum()) == 0:
        return float("nan")
    cf_eval = counterfactual_features.loc[selected].copy(deep=True)
    probabilities = future_model.get_prediction(cf_eval, proba=True).detach().cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    target_index = int(future_model.get_class_to_index()[desired_class])
    return float(np.mean(predictions == target_index))


def _run_case_fold(
    data_name: str,
    model_name: str,
    cost_name: str,
    fold: int,
    lambda_value: float | None,
    factual_limit: int | None,
    lambda_factual_limit: int | None,
    show_progress: bool,
) -> dict:
    bundle = _prepare_trial_bundle(data_name, fold)
    current_model = _build_model(model_name)
    future_model = _build_model(model_name)
    current_model.fit(bundle.current_train)
    future_model.fit(bundle.future_train)

    m1_acc, m1_auc = _compute_model_metrics(current_model, bundle.current_test)
    m2_acc, m2_auc = _compute_model_metrics(future_model, bundle.future_test)

    feature_costs = None
    if cost_name == "pfc":
        feature_costs = _simulate_pairwise_feature_costs(
            bundle.current_train.get(target=False).shape[1]
        )

    selected_lambda = float(lambda_value) if lambda_value is not None else _choose_lambda(
        model=current_model,
        trainset=bundle.current_train,
        feature_costs=feature_costs,
        factual_limit=lambda_factual_limit,
        show_progress=show_progress,
        progress_prefix=f"{data_name}/{model_name}/{cost_name} fold {fold}",
    )

    if factual_limit is not None:
        selected_indices = _recourse_needed_indices(current_model, bundle.current_test)[: max(0, int(factual_limit))]
        factuals = _subset_frozen_dataset(bundle.current_test, selected_indices)
    else:
        factuals = bundle.current_test

    method = _build_roar_method(
        model=current_model,
        trainset=bundle.current_train,
        lambda_value=selected_lambda,
        feature_costs=feature_costs,
        show_progress=show_progress,
        progress_desc=f"{data_name}/{model_name}/{cost_name} fold {fold}",
    )
    counterfactuals = method.predict(factuals)

    validity_eval = ValidityEvaluation()
    cost_eval = RoarCostEvaluation(cost_name=cost_name, feature_weights=feature_costs)
    m1_validity = float(validity_eval.evaluate(factuals, counterfactuals).iloc[0]["validity"])
    cost = float(cost_eval.evaluate(factuals, counterfactuals).iloc[0]["cost"])
    m2_validity = _future_validity(factuals, counterfactuals, future_model, desired_class=1)

    _, _, evaluation_mask, _ = resolve_evaluation_inputs(factuals, counterfactuals)
    return {
        "fold": int(fold),
        "lambda": float(selected_lambda),
        "m1_metrics": [float(m1_acc), float(m1_auc)],
        "m2_metrics": [float(m2_acc), float(m2_auc)],
        "m1_validity": float(m1_validity),
        "m2_validity": float(m2_validity),
        "cost": float(cost),
        "num_factuals": int(evaluation_mask.sum()),
    }


def _aggregate_case_results(results: dict[int, dict]) -> dict:
    ordered = [results[index] for index in sorted(results.keys())]

    def values(key: str) -> np.ndarray:
        return np.asarray([float(item[key]) for item in ordered], dtype=np.float64)

    def metric_values(metric_index: int, source_key: str) -> np.ndarray:
        return np.asarray(
            [float(item[source_key][metric_index]) for item in ordered], dtype=np.float64
        )

    aggregates = {}
    for label, array in [
        ("M1_acc", metric_values(0, "m1_metrics")),
        ("M1_auc", metric_values(1, "m1_metrics")),
        ("M2_acc", metric_values(0, "m2_metrics")),
        ("M2_auc", metric_values(1, "m2_metrics")),
        ("M1_validity", values("m1_validity")),
        ("M2_validity", values("m2_validity")),
        ("cost", values("cost")),
        ("lambda", values("lambda")),
    ]:
        aggregates[label] = {
            "values": array.tolist(),
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
        }
    return aggregates


def _compute_paper_comparison(
    data_name: str,
    model_name: str,
    cost_name: str,
    aggregate: dict,
) -> dict | None:
    paper_metrics = PAPER_METRICS.get((data_name, model_name, cost_name))
    if paper_metrics is None:
        return None

    comparison = {}
    for metric_name, paper_values in paper_metrics.items():
        observed = aggregate[PAPER_METRIC_TO_AGGREGATE_KEY[metric_name]]
        paper_mean = float(paper_values["mean"])
        observed_mean = float(observed["mean"])
        paper_std = float(paper_values["std"])
        delta = observed_mean - paper_mean
        relative_delta = abs(delta) / max(abs(paper_mean), abs(observed_mean), 1e-12)
        comparison[metric_name] = {
            "observed_mean": observed_mean,
            "observed_std": float(observed["std"]),
            "paper_mean": paper_mean,
            "paper_std": paper_std,
            "delta": float(delta),
            "relative_delta": float(relative_delta),
        }
    return comparison


def _save_case_text(
    data_name: str,
    model_name: str,
    cost_name: str,
    n_trials: int,
    case_result: dict,
) -> str:
    model_token = "lr" if model_name == "linear" else "nn"
    file_name = f"{data_name}_{model_token}_{cost_name}_robust_{n_trials}.txt"
    output_path = PROJECT_ROOT / "experiment" / "roar" / file_name
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(case_result, indent=2, sort_keys=True))
        handle.write("\n")
    return str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def run_reproduction(
    data_name: str,
    model_name: str,
    cost_name: str,
    n_trials: int,
    lambda_value: float | None,
    factual_limit: int | None,
    lambda_factual_limit: int | None,
    show_progress: bool,
) -> dict:
    if n_trials < 1 or n_trials > TOTAL_FOLDS:
        raise ValueError(f"n_trials must be between 1 and {TOTAL_FOLDS}")

    cases = []
    rows = []
    for case_data, case_model, case_cost in _expand_options(data_name, model_name, cost_name):
        trial_results: dict[int, dict] = {}
        for fold in range(n_trials):
            trial_results[int(fold)] = _run_case_fold(
                data_name=case_data,
                model_name=case_model,
                cost_name=case_cost,
                fold=int(fold),
                lambda_value=lambda_value,
                factual_limit=factual_limit,
                lambda_factual_limit=lambda_factual_limit,
                show_progress=show_progress,
            )

        aggregate = _aggregate_case_results(trial_results)
        recourse_name = "ROAR" if case_model == "linear" else "ROAR-LIME"
        case_result = {
            "dataset": case_data,
            "model": case_model,
            "cost": case_cost,
            "recourse": recourse_name,
            "n_trials": int(n_trials),
            "lambda_mode": "fixed" if lambda_value is not None else "chosen",
            "lambda_value": None if lambda_value is None else float(lambda_value),
            "factual_limit": None if factual_limit is None else int(factual_limit),
            "lambda_factual_limit": None
            if lambda_factual_limit is None
            else int(lambda_factual_limit),
            "trial_results": trial_results,
            "aggregate": aggregate,
            "paper_comparison": _compute_paper_comparison(
                case_data, case_model, case_cost, aggregate
            ),
            "paper_metrics_source": "User-supplied Table 1 values from reference/roar.pdf",
        }
        case_result["output_path"] = _save_case_text(
            data_name=case_data,
            model_name=case_model,
            cost_name=case_cost,
            n_trials=n_trials,
            case_result=case_result,
        )
        cases.append(case_result)
        rows.append(
            {
                "dataset": case_data,
                "model": case_model,
                "cost": case_cost,
                "recourse": recourse_name,
                "m1_mean": aggregate["M1_validity"]["mean"],
                "m1_std": aggregate["M1_validity"]["std"],
                "m2_mean": aggregate["M2_validity"]["mean"],
                "m2_std": aggregate["M2_validity"]["std"],
                "avg_cost_mean": aggregate["cost"]["mean"],
                "avg_cost_std": aggregate["cost"]["std"],
                "paper_comparison": case_result["paper_comparison"],
            }
        )

    output = {
        "results": cases,
        "rows": rows,
        "summary": {
            "datasets": sorted({case["dataset"] for case in cases}),
            "models": sorted({case["model"] for case in cases}),
            "costs": sorted({case["cost"] for case in cases}),
            "n_folds": int(n_trials),
            "lambda_grid": DEFAULT_LAMBDA_GRID,
            "device": "cpu",
        },
    }
    report_path = write_reproduction_report(
        output_path=REPORT_PATH,
        paper_id="roar_robust_recourse",
        reproduction_metadata={
            "timestamp": datetime.now(timezone.utc),
            "framework_version": "1.0.0",
            "source_script": Path(__file__).name,
            "n_trials": int(n_trials),
            "lambda_value": None if lambda_value is None else float(lambda_value),
            "factual_limit": None if factual_limit is None else int(factual_limit),
            "lambda_factual_limit": None
            if lambda_factual_limit is None
            else int(lambda_factual_limit),
        },
        experiments_data={
            f"{case['dataset']}_{case['model']}_{case['cost']}": {
                "configuration": {
                    "dataset": case["dataset"],
                    "model": case["model"],
                    "cost": case["cost"],
                    "recourse": case["recourse"],
                    "n_trials": case["n_trials"],
                    "lambda_mode": case["lambda_mode"],
                    "lambda_value": case["lambda_value"],
                    "factual_limit": case["factual_limit"],
                    "lambda_factual_limit": case["lambda_factual_limit"],
                },
                "metrics": {
                    "cost_mean": {
                        "original": PAPER_METRICS.get(
                            (case["dataset"], case["model"], case["cost"]), {}
                        ).get("cost", {}).get("mean"),
                        "reproduced": case["aggregate"]["cost"]["mean"],
                    },
                    "cost_std": {
                        "original": PAPER_METRICS.get(
                            (case["dataset"], case["model"], case["cost"]), {}
                        ).get("cost", {}).get("std"),
                        "reproduced": case["aggregate"]["cost"]["std"],
                    },
                    "m1_validity_mean": {
                        "original": PAPER_METRICS.get(
                            (case["dataset"], case["model"], case["cost"]), {}
                        ).get("m1_validity", {}).get("mean"),
                        "reproduced": case["aggregate"]["M1_validity"]["mean"],
                    },
                    "m1_validity_std": {
                        "original": PAPER_METRICS.get(
                            (case["dataset"], case["model"], case["cost"]), {}
                        ).get("m1_validity", {}).get("std"),
                        "reproduced": case["aggregate"]["M1_validity"]["std"],
                    },
                    "m2_validity_mean": {
                        "original": PAPER_METRICS.get(
                            (case["dataset"], case["model"], case["cost"]), {}
                        ).get("m2_validity", {}).get("mean"),
                        "reproduced": case["aggregate"]["M2_validity"]["mean"],
                    },
                    "m2_validity_std": {
                        "original": PAPER_METRICS.get(
                            (case["dataset"], case["model"], case["cost"]), {}
                        ).get("m2_validity", {}).get("std"),
                        "reproduced": case["aggregate"]["M2_validity"]["std"],
                    },
                    "lambda_mean": {
                        "original": None,
                        "reproduced": case["aggregate"]["lambda"]["mean"],
                    },
                    "lambda_std": {
                        "original": None,
                        "reproduced": case["aggregate"]["lambda"]["std"],
                    },
                },
            }
            for case in cases
        },
    )
    output["report_path"] = str(report_path)
    return output

@pytest.mark.slow
def test_reproduce() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "--dataset", default="all")
    parser.add_argument("--base-model", "--model", default="all")
    parser.add_argument("--cost", default="all")
    parser.add_argument("--n-trials", type=int, default=TOTAL_FOLDS)
    parser.add_argument("--lambda-value", type=float, default=None)
    parser.add_argument("--factual-limit", type=int, default=None)
    parser.add_argument("--lambda-factual-limit", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    output = run_reproduction(
        data_name=_normalize_data_name(args.data),
        model_name=_normalize_model_name(args.base_model),
        cost_name=_normalize_cost_name(args.cost),
        n_trials=int(args.n_trials),
        lambda_value=None if args.lambda_value is None else float(args.lambda_value),
        factual_limit=args.factual_limit,
        lambda_factual_limit=args.lambda_factual_limit,
        show_progress=not args.no_progress,
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    test_reproduce()
