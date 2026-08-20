from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback
from typing import Iterable

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_object import DatasetObject
from evaluation.distance import DistanceEvaluation
from evaluation.validity import ValidityEvaluation
from experiment.utils import write_reproduction_report
from method.larr.larr import LarrMethod
from method.larr.library.larr import RecourseCost
from model.linear.linear import LinearModel
from model.mlp.mlp import MlpModel
from model.model_object import ModelObject
from preprocess.common import EncodePreProcess, FinalizePreProcess, ScalePreProcess
from utils.seed import seed_context

SEED = 0
N_FOLDS = 5
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("logs")


class LocalDataFrameDataset(DatasetObject):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        name: str,
        target_column: str,
        raw_feature_type: dict[str, str],
        raw_feature_mutability: dict[str, bool] | None = None,
        raw_feature_actionability: dict[str, str] | None = None,
    ):
        self._rawdf = df.copy(deep=True)
        self._freeze = False
        self.name = name
        self.target_column = target_column
        self.feature_order = list(df.columns)
        self.raw_feature_type = dict(raw_feature_type)
        self.raw_feature_mutability = raw_feature_mutability or {
            column: column != target_column for column in df.columns
        }
        self.raw_feature_actionability = raw_feature_actionability or {
            column: ("none" if column == target_column else "any") for column in df.columns
        }

    def _read_df(self, path: str) -> pd.DataFrame:
        raise NotImplementedError("LocalDataFrameDataset is initialized from a DataFrame")


@dataclass(frozen=True)
class ExperimentConfig:
    profile: str
    datasets: tuple[str, ...]
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    betas: tuple[float, ...]
    alphas: tuple[float, ...]
    lambdas: tuple[float, ...]
    max_factuals: int
    output_dir: Path
    device: str


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _metadata_for_features(
    columns: Iterable[str],
    target_column: str,
    numerical: set[str],
    categorical: set[str],
    immutable: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
    immutable = immutable or set()
    feature_type: dict[str, str] = {}
    mutability: dict[str, bool] = {}
    actionability: dict[str, str] = {}
    for column in columns:
        if column == target_column:
            feature_type[column] = "binary"
            mutability[column] = False
            actionability[column] = "none"
        elif column in numerical:
            feature_type[column] = "numerical"
            mutability[column] = column not in immutable
            actionability[column] = "same" if column in immutable else "any"
        elif column in categorical:
            feature_type[column] = "categorical"
            mutability[column] = column not in immutable
            actionability[column] = "same" if column in immutable else "any"
        else:
            feature_type[column] = "binary"
            mutability[column] = column not in immutable
            actionability[column] = "same" if column in immutable else "any"
    return feature_type, mutability, actionability


def _synthetic_dataset(seed: int, shifted: bool = False, n: int = 1000) -> LocalDataFrameDataset:
    rng = np.random.default_rng(seed)
    shift = 0.1 if shifted else 0.0
    variance_multiplier = 1.15 if shifted else 1.0
    cov = 0.5 * variance_multiplier * np.eye(2)
    x0 = rng.multivariate_normal([-2 + shift, -2 + shift], cov, n // 2)
    x1 = rng.multivariate_normal([2 + shift, 2 + shift], cov, n // 2)
    data = np.vstack([x0, x1])
    labels = np.array([0] * (n // 2) + [1] * (n // 2), dtype=int)
    frame = pd.DataFrame(data, columns=["x0", "x1"])
    frame["label"] = labels
    frame = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    feature_type, mutability, actionability = _metadata_for_features(
        frame.columns,
        "label",
        numerical={"x0", "x1"},
        categorical=set(),
    )
    return LocalDataFrameDataset(
        frame,
        name="synthetic",
        target_column="label",
        raw_feature_type=feature_type,
        raw_feature_mutability=mutability,
        raw_feature_actionability=actionability,
    )


def _german_dataset(seed: int, shifted: bool = False) -> LocalDataFrameDataset:
    path = PROJECT_ROOT / "dataset" / "german" / "german.csv"
    frame = pd.read_csv(path)
    keep = ["duration", "amount", "age", "personal_status_sex", "credit_risk"]
    frame = frame.loc[:, keep].sample(frac=1, random_state=seed).reset_index(drop=True)
    if shifted:
        # Local deterministic proxy for the paper's corrected-German future model.
        frame = frame.copy(deep=True)
        frame["duration"] = frame["duration"] * 1.05
        frame["amount"] = frame["amount"] * 1.03
        frame["age"] = frame["age"] + 1.0
    feature_type, mutability, actionability = _metadata_for_features(
        frame.columns,
        "credit_risk",
        numerical={"duration", "amount", "age"},
        categorical={"personal_status_sex"},
        immutable={"age"},
    )
    return LocalDataFrameDataset(
        frame,
        name="german",
        target_column="credit_risk",
        raw_feature_type=feature_type,
        raw_feature_mutability=mutability,
        raw_feature_actionability=actionability,
    )


def _sba_dataset(seed: int, shifted: bool = False) -> LocalDataFrameDataset:
    path = PROJECT_ROOT / "dataset" / "sba_roar" / "SBAcase.11.13.17.csv"
    frame = pd.read_csv(path).fillna(-1)
    frame["NoDefault"] = 1 - frame["Default"].astype(int)
    drop_columns = {
        "Selected",
        "State",
        "Name",
        "BalanceGross",
        "LowDoc",
        "BankState",
        "LoanNr_ChkDgt",
        "MIS_Status",
        "Default",
        "Bank",
        "City",
    }
    frame = frame.drop(columns=[column for column in drop_columns if column in frame])
    target = "NoDefault"
    feature_columns = [column for column in frame.columns if column != target]
    if shifted:
        frame = frame.copy(deep=True)
    else:
        frame = frame.loc[frame["ApprovalFY"] < 2006].copy(deep=True)
    frame = frame.loc[:, feature_columns + [target]].sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    categorical = {
        column for column in feature_columns if frame[column].dtype == object
    }
    numerical = set(feature_columns) - categorical
    feature_type, mutability, actionability = _metadata_for_features(
        frame.columns,
        target,
        numerical=numerical,
        categorical=categorical,
    )
    return LocalDataFrameDataset(
        frame,
        name="sba",
        target_column=target,
        raw_feature_type=feature_type,
        raw_feature_mutability=mutability,
        raw_feature_actionability=actionability,
    )


def _load_dataset(name: str, seed: int, shifted: bool = False) -> LocalDataFrameDataset:
    if name == "synthetic":
        return _synthetic_dataset(seed=seed, shifted=shifted)
    if name == "german":
        return _german_dataset(seed=seed, shifted=shifted)
    if name == "sba":
        return _sba_dataset(seed=seed, shifted=shifted)
    raise ValueError(f"Unsupported LARR reproduction dataset: {name}")


def _preprocess(dataset: LocalDataFrameDataset, seed: int) -> LocalDataFrameDataset:
    dataset = ScalePreProcess(seed=seed, scaling="standardize", range=True).transform(dataset)
    dataset = EncodePreProcess(seed=seed, encoding="onehot").transform(dataset)
    dataset = FinalizePreProcess(seed=seed).transform(dataset)
    return dataset


def _feature_sets(dataset: LocalDataFrameDataset) -> tuple[list[str], list[str]]:
    categorical: list[str] = []
    numerical: list[str] = []
    for feature_name, feature_type in dataset.raw_feature_type.items():
        if feature_name == dataset.target_column:
            continue
        if str(feature_type).lower() == "categorical":
            categorical.append(feature_name)
        else:
            numerical.append(feature_name)
    return categorical, numerical


def _standardize_with_base_stats(
    base_df: pd.DataFrame,
    shifted_df: pd.DataFrame,
    numerical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float | str]]]:
    base = base_df.copy(deep=True)
    shifted = shifted_df.copy(deep=True)
    stats: dict[str, dict[str, float | str]] = {}
    for column in numerical_columns:
        if column not in base.columns:
            continue
        series = base[column].astype("float64")
        mean_value = float(series.mean())
        std_value = float(series.std(ddof=0))
        stats[column] = {
            "mode": "standardize",
            "mean": mean_value,
            "std": std_value,
        }
        if std_value == 0.0:
            base[column] = 0.0
            shifted[column] = 0.0
        else:
            base[column] = (base[column].astype("float64") - mean_value) / std_value
            shifted[column] = (
                shifted[column].astype("float64") - mean_value
            ) / std_value
    return base, shifted, stats


def _encode_pair_with_shared_schema(
    base_df: pd.DataFrame,
    shifted_df: pd.DataFrame,
    target_column: str,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], dict[str, str]]:
    base_features = base_df.drop(columns=[target_column]).copy(deep=True)
    shifted_features = shifted_df.drop(columns=[target_column]).copy(deep=True)
    for column in categorical_columns:
        if column in base_features.columns:
            base_features[column] = base_features[column].astype(str)
        if column in shifted_features.columns:
            shifted_features[column] = shifted_features[column].astype(str)

    combined = pd.concat(
        [base_features, shifted_features],
        axis=0,
        ignore_index=True,
    )
    encoded = pd.get_dummies(
        combined,
        columns=[column for column in categorical_columns if column in combined],
        dtype="float64",
    )
    encoded = encoded.astype("float64")
    base_encoded = encoded.iloc[: len(base_features)].copy(deep=True)
    shifted_encoded = encoded.iloc[len(base_features) :].copy(deep=True)
    base_encoded.index = base_df.index
    shifted_encoded.index = shifted_df.index

    encoding_map: dict[str, list[str]] = {}
    encoded_sources: dict[str, str] = {}
    for source in categorical_columns:
        prefix = f"{source}_"
        columns = [column for column in encoded.columns if column.startswith(prefix)]
        if columns:
            encoding_map[source] = columns
            for column in columns:
                encoded_sources[column] = source

    base_encoded[target_column] = base_df[target_column].to_numpy()
    shifted_encoded[target_column] = shifted_df[target_column].to_numpy()
    return base_encoded, shifted_encoded, encoding_map, encoded_sources


def _build_preprocessed_pair(
    dataset_name: str,
    seed: int,
) -> tuple[LocalDataFrameDataset, LocalDataFrameDataset]:
    raw_base = _load_dataset(dataset_name, seed=seed, shifted=False)
    raw_shifted = _load_dataset(dataset_name, seed=seed, shifted=True)
    target_column = raw_base.target_column
    categorical_columns, numerical_columns = _feature_sets(raw_base)

    base_df = raw_base.snapshot().copy(deep=True)
    shifted_df = raw_shifted.snapshot().copy(deep=True)

    if dataset_name == "synthetic":
        scaling_stats: dict[str, dict[str, float | str]] = {}
        encoded_base = base_df
        encoded_shifted = shifted_df
        encoding_map: dict[str, list[str]] = {}
        encoded_sources: dict[str, str] = {}
    else:
        encoded_base, encoded_shifted, encoding_map, encoded_sources = (
            _encode_pair_with_shared_schema(
                base_df=base_df,
                shifted_df=shifted_df,
                target_column=target_column,
                categorical_columns=categorical_columns,
            )
        )
        encoded_base, encoded_shifted, scaling_stats = _standardize_with_base_stats(
            encoded_base,
            encoded_shifted,
            numerical_columns=numerical_columns,
        )

    feature_type: dict[str, str] = {}
    feature_mutability: dict[str, bool] = {}
    feature_actionability: dict[str, str] = {}
    for column in encoded_base.columns:
        if column == target_column:
            feature_type[column] = "binary"
            feature_mutability[column] = False
            feature_actionability[column] = "none"
            continue
        source = encoded_sources.get(column, column)
        if source in categorical_columns:
            feature_type[column] = "binary"
        else:
            feature_type[column] = "numerical"
        feature_mutability[column] = bool(raw_base.raw_feature_mutability[source])
        feature_actionability[column] = str(raw_base.raw_feature_actionability[source])

    base = LocalDataFrameDataset(
        encoded_base,
        name=dataset_name,
        target_column=target_column,
        raw_feature_type=feature_type,
        raw_feature_mutability=feature_mutability,
        raw_feature_actionability=feature_actionability,
    )
    shifted = LocalDataFrameDataset(
        encoded_shifted,
        name=f"{dataset_name}_shifted",
        target_column=target_column,
        raw_feature_type=feature_type,
        raw_feature_mutability=feature_mutability,
        raw_feature_actionability=feature_actionability,
    )
    for dataset in (base, shifted):
        if encoding_map:
            dataset.update("encoding", encoding_map)
        if scaling_stats:
            dataset.update("scaling", {column: "standardize" for column in scaling_stats})
            dataset.update("scaling_stats", scaling_stats)
        dataset.update("paired_preprocessed", True)
        dataset.freeze()
    return base, shifted


def _split_fold(dataset: DatasetObject, fold_index: int) -> tuple[DatasetObject, DatasetObject]:
    full_df = pd.concat([dataset.get(target=False), dataset.get(target=True)], axis=1)
    start = int(fold_index / N_FOLDS * len(full_df))
    end = int((fold_index + 1) / N_FOLDS * len(full_df))
    test_df = full_df.iloc[start:end].copy(deep=True)
    train_df = pd.concat([full_df.iloc[:start], full_df.iloc[end:]], axis=0)

    trainset = dataset.clone()
    trainset.update("trainset", True, df=train_df.copy(deep=True))
    trainset.freeze()
    testset = dataset.clone()
    testset.update("testset", True, df=test_df.copy(deep=True))
    testset.freeze()
    return trainset, testset


def _model_for(name: str, seed: int, device: str, train_rows: int) -> ModelObject:
    if name == "linear":
        return LinearModel(
            seed=seed,
            device=device,
            epochs=200,
            learning_rate=0.03,
            batch_size=train_rows,
            optimizer="adam",
            criterion="bce",
            output_activation="sigmoid",
            save_name=None,
        )
    if name == "mlp":
        return MlpModel(
            seed=seed,
            device=device,
            epochs=100,
            learning_rate=0.001,
            batch_size=train_rows,
            layers=[50, 100, 200],
            optimizer="adam",
            criterion="bce",
            output_activation="sigmoid",
            save_name=None,
        )
    raise ValueError(f"Unsupported model: {name}")


def _prediction_indices(model: ModelObject, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.get_prediction(features, proba=True)
    return probabilities.detach().cpu().numpy().argmax(axis=1)


def _select_factuals(
    model: ModelObject,
    testset: DatasetObject,
    max_factuals: int,
) -> DatasetObject:
    features = testset.get(target=False)
    predictions = _prediction_indices(model, features)
    selected_index = features.index[predictions == 0][:max_factuals]
    if len(selected_index) == 0:
        raise RuntimeError("No test instances require recourse for target class 1")
    factual_df = pd.concat(
        [testset.get(target=False).loc[selected_index], testset.get(target=True).loc[selected_index]],
        axis=1,
    )
    factuals = testset.clone()
    factuals.update("factuals", True, df=factual_df)
    factuals.freeze()
    return factuals


def _fit_larr(
    model: ModelObject,
    trainset: DatasetObject,
    *,
    seed: int,
    device: str,
    alpha: float,
    beta: float,
) -> LarrMethod:
    method = LarrMethod(
        target_model=model,
        seed=seed,
        device=device,
        desired_class=1,
        alpha=alpha,
        beta=beta,
        lime_seed=seed,
    )
    method.fit(trainset)
    return method


def _make_future_theta(
    theta_0: tuple[np.ndarray, float],
    theta_r: tuple[np.ndarray, float],
    error_level: float,
    alpha: float,
) -> tuple[np.ndarray, float]:
    weights_0, bias_0 = theta_0
    weights_r, bias_r = theta_r
    weights = weights_0 + error_level * (weights_r - weights_0)
    bias = bias_0 + error_level * (bias_r - bias_0)
    weights = np.clip(weights, weights_0 - alpha, weights_0 + alpha).round(4)
    bias = float(np.clip(bias, bias_0 - alpha, bias_0 + alpha).round(4))
    return weights, bias


def _paper_metrics_for_method(
    method: LarrMethod,
    factuals: pd.DataFrame,
    betas: Iterable[float],
    prediction_error_levels: Iterable[float],
) -> list[dict[str, float | int | str]]:
    if method._adapter is None or method._train_features is None:
        raise RuntimeError("LarrMethod is not fully fitted")
    adapter = method._adapter
    larr = method._method
    records: list[dict[str, float | int | str]] = []

    for factual_position, (_, row) in enumerate(factuals.iterrows()):
        x0 = row.to_numpy(dtype="float32")
        target_index = 1
        weights_0, bias_0 = method._get_surrogate(x0, target_index)
        weights_0 = np.round(weights_0, 4)
        bias_0 = float(np.round(bias_0, 4))
        larr.weights = weights_0
        larr.bias = bias_0
        objective = RecourseCost(x0, larr.lamb)

        robust_x = larr.get_recourse(x0, beta=1.0)
        theta_r = larr.calc_theta_adv(robust_x)
        robust_opt_cost = float(np.asarray(objective.eval(robust_x, *theta_r)).reshape(-1)[0])

        consistent_opt_by_error: dict[float, tuple[np.ndarray, float]] = {}
        consistent_cost_by_error: dict[float, float] = {}
        smooth_reference_by_error: dict[float, np.ndarray] = {}
        for error_level in prediction_error_levels:
            theta_p = _make_future_theta((weights_0, bias_0), theta_r, error_level, larr.alpha)
            consistent_x = larr.get_recourse(x0, beta=0.0, theta_p=theta_p)
            consistent_opt_by_error[float(error_level)] = theta_p
            consistent_cost_by_error[float(error_level)] = float(
                np.asarray(objective.eval(consistent_x, *theta_p)).reshape(-1)[0]
            )
            if float(error_level) == 0.0:
                smooth_reference_by_error[float(error_level)] = consistent_x

        smooth_reference = smooth_reference_by_error.get(0.0, robust_x)

        for beta in betas:
            for error_level, theta_p in consistent_opt_by_error.items():
                x = larr.get_recourse(x0, beta=float(beta), theta_p=theta_p)
                theta_adv = larr.calc_theta_adv(x)
                robust_cost = float(
                    np.asarray(objective.eval(x, *theta_adv)).reshape(-1)[0]
                )
                consistent_cost = float(
                    np.asarray(objective.eval(x, *theta_p)).reshape(-1)[0]
                )
                prediction_df = pd.DataFrame([x], columns=method._feature_names)
                current_validity = float(adapter.predict_label_indices(prediction_df)[0] == 1)
                records.append(
                    {
                        "i": factual_position,
                        "beta": float(beta),
                        "prediction_error": float(error_level),
                        "lambda": float(larr.lamb),
                        "robustness": robust_cost - robust_opt_cost,
                        "consistency": consistent_cost
                        - consistent_cost_by_error[float(error_level)],
                        "smoothness_l1": float(np.linalg.norm(x - smooth_reference, ord=1)),
                        "current_validity": current_validity,
                        "cost_l1": float(np.linalg.norm(x - x0, ord=1)),
                    }
                )
    return records


def _run_one_setting(
    dataset_name: str,
    model_name: str,
    seed: int,
    alpha: float,
    config: ExperimentConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    base_dataset, shifted_dataset = _build_preprocessed_pair(dataset_name, seed=seed)
    trainset, testset = _split_fold(base_dataset, fold_index=seed % N_FOLDS)
    shifted_trainset, shifted_testset = _split_fold(
        shifted_dataset, fold_index=seed % N_FOLDS
    )

    model = _model_for(model_name, seed, config.device, len(trainset))
    model.fit(trainset)
    shifted_model = _model_for(model_name, seed, config.device, len(shifted_trainset))
    shifted_model.fit(shifted_trainset)

    test_features = testset.get(target=False)
    test_target = testset.get(target=True).iloc[:, 0].astype(int).to_numpy()
    test_prediction = _prediction_indices(model, test_features)
    train_prediction = _prediction_indices(model, trainset.get(target=False))
    shifted_prediction = _prediction_indices(
        shifted_model, shifted_testset.get(target=False)
    )
    shifted_target = shifted_testset.get(target=True).iloc[:, 0].astype(int).to_numpy()
    recourse_needed_test = int(np.sum(test_prediction == 0))
    recourse_needed_train = int(np.sum(train_prediction == 0))

    factuals = _select_factuals(model, testset, config.max_factuals)
    method = _fit_larr(
        model,
        trainset,
        seed=seed,
        device=config.device,
        alpha=alpha,
        beta=0.5,
    )

    metric_records = _paper_metrics_for_method(
        method,
        factuals.get(target=False),
        betas=config.betas,
        prediction_error_levels=(0.0, 0.5, 1.0),
    )

    generated_records: list[dict[str, object]] = []
    for lamb in config.lambdas:
        method_for_lambda = _fit_larr(
            model,
            trainset,
            seed=seed,
            device=config.device,
            alpha=alpha,
            beta=1.0,
        )
        method_for_lambda._method.lamb = float(lamb)
        method_for_lambda._lambda_ready = True
        start = perf_counter()
        counterfactuals = method_for_lambda.predict(factuals, batch_size=20)
        elapsed = perf_counter() - start
        validity = float(ValidityEvaluation().evaluate(factuals, counterfactuals)["validity"].iloc[0])
        distance = DistanceEvaluation(metrics=["l0", "l1"]).evaluate(
            factuals, counterfactuals
        )

        cf_features = counterfactuals.get(target=False)
        valid_rows = ~cf_features.isna().any(axis=1)
        future_validity = float("nan")
        if bool(valid_rows.any()):
            future_pred = _prediction_indices(shifted_model, cf_features.loc[valid_rows])
            future_validity = float(np.mean(future_pred == 1))

        generated_records.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "seed": seed,
                "alpha": alpha,
                "lambda": float(lamb),
                "factuals": len(factuals),
                "validity": validity,
                "future_validity": future_validity,
                "distance_l0": float(distance["distance_l0"].iloc[0]),
                "distance_l1": float(distance["distance_l1"].iloc[0]),
                "runtime_seconds": elapsed,
            }
        )

    for record in metric_records:
        record.update(
            {
                "dataset": dataset_name,
                "model": model_name,
                "seed": seed,
                "alpha": alpha,
                "factuals": len(factuals),
            }
        )
    diagnostics = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "alpha": alpha,
        "train_rows": len(trainset),
        "test_rows": len(testset),
        "shifted_train_rows": len(shifted_trainset),
        "shifted_test_rows": len(shifted_testset),
        "feature_count": int(test_features.shape[1]),
        "recourse_needed_train": recourse_needed_train,
        "recourse_needed_test": recourse_needed_test,
        "selected_factuals": len(factuals),
        "target_accuracy": float(np.mean(test_prediction == test_target)),
        "shifted_target_accuracy": float(np.mean(shifted_prediction == shifted_target)),
        "selected_lambda": float(method._method.lamb),
    }
    return metric_records, generated_records, diagnostics


def _aggregate(records: list[dict[str, object]], group_cols: list[str]) -> list[dict[str, object]]:
    if not records:
        return []
    frame = pd.DataFrame(records)
    numeric_cols = [
        column
        for column in frame.columns
        if column not in group_cols and pd.api.types.is_numeric_dtype(frame[column])
    ]
    grouped = frame.groupby(group_cols, dropna=False)[numeric_cols]
    mean_frame = grouped.mean().reset_index()
    std_frame = grouped.std(ddof=1).reset_index()
    output: list[dict[str, object]] = []
    for idx, row in mean_frame.iterrows():
        item = row.to_dict()
        std_row = std_frame.iloc[idx]
        for column in numeric_cols:
            item[f"{column}_std"] = std_row[column]
        output.append(item)
    return output


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_reproduction(config: ExperimentConfig) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metric_records: list[dict[str, object]] = []
    validity_cost_records: list[dict[str, object]] = []
    setting_diagnostics: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for dataset_name in config.datasets:
        for model_name in config.models:
            for alpha in config.alphas:
                for seed in config.seeds:
                    with seed_context(seed):
                        try:
                            metrics, validity_cost, diagnostics = _run_one_setting(
                                dataset_name,
                                model_name,
                                seed,
                                alpha,
                                config,
                            )
                            metric_records.extend(metrics)
                            validity_cost_records.extend(validity_cost)
                            setting_diagnostics.append(diagnostics)
                        except Exception as error:
                            failures.append(
                                {
                                    "dataset": dataset_name,
                                    "model": model_name,
                                    "alpha": alpha,
                                    "seed": seed,
                                    "error": repr(error),
                                    "traceback": traceback.format_exc(limit=8),
                                }
                            )

    metric_summary = _aggregate(
        metric_records,
        ["dataset", "model", "alpha", "beta", "prediction_error"],
    )
    validity_cost_summary = _aggregate(
        validity_cost_records,
        ["dataset", "model", "alpha", "lambda"],
    )

    payload = {
        "metadata": {
            "timestamp": started.isoformat().replace("+00:00", "Z"),
            "profile": config.profile,
            "datasets": list(config.datasets),
            "models": list(config.models),
            "seeds": list(config.seeds),
            "betas": list(config.betas),
            "alphas": list(config.alphas),
            "lambdas": list(config.lambdas),
            "max_factuals": config.max_factuals,
            "device": config.device,
            "source_script": Path(__file__).name,
        },
        "robustness_consistency_smoothness": metric_summary,
        "future_validity_cost": validity_cost_summary,
        "setting_diagnostics": setting_diagnostics,
        "failures": failures,
    }

    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    artifact_path = config.output_dir / f"larr_reproduction_{config.profile}_{timestamp}.json"
    _write_json(artifact_path, payload)

    report_experiments: dict[str, dict[str, object]] = {}
    for record in metric_summary:
        experiment_id = (
            f"robustness_consistency_smoothness_"
            f"{record['dataset']}_{record['model']}_alpha{record['alpha']}_"
            f"beta{record['beta']}_err{record['prediction_error']}"
        )
        report_experiments[experiment_id] = {
            "configuration": {
                "dataset": record["dataset"],
                "model": record["model"],
                "method": "larr",
                "alpha": record["alpha"],
                "beta": record["beta"],
                "prediction_error": record["prediction_error"],
                "profile": config.profile,
            },
            "metrics": {
                "average_robustness": {
                    "original": None,
                    "reproduced": record.get("robustness"),
                },
                "average_consistency": {
                    "original": None,
                    "reproduced": record.get("consistency"),
                },
                "average_smoothness_l1": {
                    "original": None,
                    "reproduced": record.get("smoothness_l1"),
                },
            },
        }
    for record in validity_cost_summary:
        experiment_id = (
            f"future_validity_cost_{record['dataset']}_{record['model']}_"
            f"alpha{record['alpha']}_lambda{record['lambda']}"
        )
        report_experiments[experiment_id] = {
            "configuration": {
                "dataset": record["dataset"],
                "model": record["model"],
                "method": "larr",
                "alpha": record["alpha"],
                "lambda": record["lambda"],
                "profile": config.profile,
            },
            "metrics": {
                "current_validity": {
                    "original": None,
                    "reproduced": record.get("validity"),
                },
                "future_validity": {
                    "original": None,
                    "reproduced": record.get("future_validity"),
                },
                "average_cost_l1": {
                    "original": None,
                    "reproduced": record.get("distance_l1"),
                },
            },
        }

    report_path = config.output_dir / "reproduction_report.json"
    write_reproduction_report(
        output_path=report_path,
        paper_id="larr",
        reproduction_metadata=payload["metadata"] | {"artifact_path": str(artifact_path)},
        experiments_data=report_experiments,
    )
    print(f"Wrote detailed artifact: {artifact_path}")
    print(f"Wrote reproduction report: {report_path}")
    if failures:
        print(f"Completed with {len(failures)} failed settings")
    return payload


def _parse_csv(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.profile == "smoke":
        datasets = ("synthetic",) if args.datasets is None else _parse_csv(args.datasets)
        models = ("linear",) if args.models is None else _parse_csv(args.models)
        seeds = (0,) if args.seeds is None else _parse_csv(args.seeds, int)
        betas = (0.0, 0.5, 1.0) if args.betas is None else _parse_csv(args.betas, float)
        alphas = (0.5,) if args.alphas is None else _parse_csv(args.alphas, float)
        lambdas = (0.1, 0.3) if args.lambdas is None else _parse_csv(args.lambdas, float)
        max_factuals = args.max_factuals or 3
    elif args.profile == "bounded":
        datasets = ("synthetic", "german") if args.datasets is None else _parse_csv(args.datasets)
        models = ("linear", "mlp") if args.models is None else _parse_csv(args.models)
        seeds = (0, 1) if args.seeds is None else _parse_csv(args.seeds, int)
        betas = (0.0, 0.25, 0.5, 0.75, 1.0) if args.betas is None else _parse_csv(args.betas, float)
        alphas = (0.5,) if args.alphas is None else _parse_csv(args.alphas, float)
        lambdas = (0.1, 0.2, 0.3) if args.lambdas is None else _parse_csv(args.lambdas, float)
        max_factuals = args.max_factuals or 10
    else:
        datasets = ("synthetic", "german", "sba") if args.datasets is None else _parse_csv(args.datasets)
        models = ("linear", "mlp") if args.models is None else _parse_csv(args.models)
        seeds = tuple(range(5)) if args.seeds is None else _parse_csv(args.seeds, int)
        betas = tuple(np.arange(0.0, 1.01, 0.01).round(2).tolist()) if args.betas is None else _parse_csv(args.betas, float)
        alphas = (0.5,) if args.alphas is None else _parse_csv(args.alphas, float)
        lambdas = (0.1, 0.2, 0.3) if args.lambdas is None else _parse_csv(args.lambdas, float)
        max_factuals = args.max_factuals or 100

    return ExperimentConfig(
        profile=args.profile,
        datasets=tuple(str(item) for item in datasets),
        models=tuple(str(item) for item in models),
        seeds=tuple(int(item) for item in seeds),
        betas=tuple(float(item) for item in betas),
        alphas=tuple(float(item) for item in alphas),
        lambdas=tuple(float(item) for item in lambdas),
        max_factuals=int(max_factuals),
        output_dir=Path(args.output_dir),
        device=_resolve_device(args.device),
    )


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Paper-aligned LARR reproduction")
    profile_group = parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--smoke",
        action="store_const",
        const="smoke",
        dest="profile",
        help="Run a tiny executable check.",
    )
    profile_group.add_argument(
        "--bounded",
        action="store_const",
        const="bounded",
        dest="profile",
        help="Run a reduced multi-dataset, multi-model reproduction.",
    )
    profile_group.add_argument(
        "--paper",
        action="store_const",
        const="paper",
        dest="profile",
        help="Run the full paper-aligned reproduction. This is the default.",
    )
    parser.set_defaults(profile="paper")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names")
    parser.add_argument("--models", default=None, help="Comma-separated model names")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds")
    parser.add_argument("--betas", default=None, help="Comma-separated beta values")
    parser.add_argument("--alphas", default=None, help="Comma-separated alpha values")
    parser.add_argument("--lambdas", default=None, help="Comma-separated lambda values")
    parser.add_argument("--max-factuals", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR.as_posix(),
        help="Directory for generated reproduction artifacts",
    )
    args = parser.parse_args(argv)
    return run_reproduction(build_config(args))


@pytest.mark.slow
def test_reproduce() -> None:
    payload = main(["--smoke"])
    assert not payload["failures"]
    assert payload["robustness_consistency_smoothness"]
    assert payload["future_validity_cost"]


if __name__ == "__main__":
    main()
