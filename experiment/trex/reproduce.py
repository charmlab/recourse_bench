from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from tqdm.auto import tqdm

from dataset.dataset_object import DatasetObject
from method.trex.trex import TrexMethod
from model.mlp.mlp import MlpModel

TARGET_COLUMN = "credit_risk"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yml")

NUMERIC_COLUMNS = [
    "duration",
    "amount",
    "installment_rate",
    "present_residence",
    "age",
    "number_credits",
    "people_liable",
]
CAT_COLUMNS = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]
EXPECTED_INPUT_DIM = 61
PAPER_TARGETS = {
    "l1": {
        "cost": 4.81,
        "lof": 0.72,
        "wi_validity_pct": 98.0,
        "lo_validity_pct": 96.5,
    },
    "l2": {
        "cost": 1.20,
        "lof": 0.75,
        "wi_validity_pct": 99.2,
        "lo_validity_pct": 98.7,
    },
}


class FrameDataset(DatasetObject):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        target_column: str = TARGET_COLUMN,
        name: str = "frame_dataset",
    ):
        self._rawdf = df.copy(deep=True)
        self._freeze = False
        self.name = name
        self.target_column = target_column
        self.feature_order = list(self._rawdf.columns)
        self.raw_feature_type = {
            column: ("binary" if column == target_column else "numerical")
            for column in self._rawdf.columns
        }
        self.raw_feature_mutability = {
            column: column != target_column for column in self._rawdf.columns
        }
        self.raw_feature_actionability = {
            column: ("none" if column == target_column else "any")
            for column in self._rawdf.columns
        }

    def _read_df(self, path: str) -> pd.DataFrame:
        raise NotImplementedError("FrameDataset reads from an in-memory DataFrame")


@dataclass
class ModelBundle:
    model: MlpModel
    seed: int
    variant: str


@dataclass
class RunMetrics:
    tau: float
    norm: int
    current_validity_pct: float
    cost: float
    lof: float
    wi_validity_pct: float
    lo_validity_pct: float
    num_factuals: int
    num_valid_counterfactuals: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau": float(self.tau),
            "norm": int(self.norm),
            "current_validity_pct": float(self.current_validity_pct),
            "cost": float(self.cost),
            "lof": float(self.lof),
            "wi_validity_pct": float(self.wi_validity_pct),
            "lo_validity_pct": float(self.lo_validity_pct),
            "num_factuals": int(self.num_factuals),
            "num_valid_counterfactuals": int(self.num_valid_counterfactuals),
        }


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _single_data_config(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.get("data")
    if not isinstance(data_cfg, list) or len(data_cfg) != 1:
        raise ValueError("TreX reproduction expects exactly one data section")
    item = data_cfg[0]
    if not isinstance(item, dict):
        raise ValueError("Data section must contain a dictionary entry")
    return item


def _resolve_settings(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    experiment_cfg = config.get("experiment", {})
    data_item = _single_data_config(config)
    data_overrides = data_item.get("overrides", {})
    model_overrides = config.get("model", {}).get("overrides", {})
    method_overrides = config.get("method", {}).get("overrides", {})
    evaluation_cfg = config.get("evaluation", {})
    evaluation_overrides = evaluation_cfg.get("overrides", {})

    split_seed = (
        int(args.split_seed)
        if args.split_seed is not None
        else int(data_overrides.get("split_seed", 27))
    )
    device = _resolve_device(args.device or str(experiment_cfg.get("device", "auto")))
    learning_rate = (
        float(args.learning_rate)
        if args.learning_rate is not None
        else float(model_overrides.get("learning_rate", 0.001))
    )
    train_batch_size = (
        int(args.train_batch_size)
        if args.train_batch_size is not None
        else int(model_overrides.get("batch_size", 32))
    )
    wi_models = (
        int(args.wi_models)
        if args.wi_models is not None
        else int(evaluation_overrides.get("wi_models", 50))
    )
    lo_models = (
        int(args.lo_models)
        if args.lo_models is not None
        else int(evaluation_overrides.get("lo_models", 50))
    )
    pilot_model_count = (
        int(args.pilot_model_count)
        if args.pilot_model_count is not None
        else int(evaluation_overrides.get("pilot_model_count", 10))
    )
    pilot_factual_limit = (
        int(args.pilot_factual_limit)
        if args.pilot_factual_limit is not None
        else int(evaluation_overrides.get("pilot_factual_limit", 64))
    )
    factual_limit = (
        int(args.factual_limit)
        if args.factual_limit is not None
        else evaluation_overrides.get("factual_limit")
    )
    if factual_limit is not None:
        factual_limit = int(factual_limit)

    tau_grid = (
        [float(value) for value in args.tau_grid]
        if args.tau_grid is not None
        else [float(value) for value in method_overrides.get("tau_grid", [0.70, 0.75, 0.80, 0.85])]
    )
    tau_l1 = (
        float(args.tau_l1)
        if args.tau_l1 is not None
        else method_overrides.get("tau_l1")
    )
    tau_l2 = (
        float(args.tau_l2)
        if args.tau_l2 is not None
        else method_overrides.get("tau_l2")
    )
    k = int(args.k) if args.k is not None else int(method_overrides.get("k", 1000))
    sigma = (
        float(args.sigma)
        if args.sigma is not None
        else float(method_overrides.get("sigma", 0.1))
    )
    cf_confidence = (
        float(args.cf_confidence)
        if args.cf_confidence is not None
        else float(method_overrides.get("cf_confidence", 0.5))
    )
    cf_steps = (
        int(args.cf_steps)
        if args.cf_steps is not None
        else int(method_overrides.get("cf_steps", 60))
    )
    cf_step_size = (
        float(args.cf_step_size)
        if args.cf_step_size is not None
        else float(method_overrides.get("cf_step_size", 0.02))
    )
    trex_step_size = (
        float(args.trex_step_size)
        if args.trex_step_size is not None
        else float(method_overrides.get("trex_step_size", 0.01))
    )
    trex_max_steps = (
        int(args.trex_max_steps)
        if args.trex_max_steps is not None
        else int(method_overrides.get("trex_max_steps", 200))
    )
    trex_epsilon = (
        float(args.trex_epsilon)
        if args.trex_epsilon is not None
        else float(method_overrides.get("trex_epsilon", 1.0))
    )
    lof_neighbors = int(evaluation_overrides.get("lof_n_neighbors", 1))
    desired_class = int(method_overrides.get("desired_class", 1))

    settings = {
        "config_path": str(Path(args.config).resolve()),
        "experiment_name": str(experiment_cfg.get("name", "german_trex_reproduce")),
        "seed": int(experiment_cfg.get("seed", 0)),
        "device": device,
        "data_name": str(data_item.get("name", "german_reconstructed_61d")),
        "raw_data_path": _resolve_project_path(
            str(data_overrides.get("raw_path", "./dataset/german/german.csv"))
        ),
        "split_seed": split_seed,
        "test_split": float(data_overrides.get("test_split", 0.33)),
        "label_flip": bool(data_overrides.get("label_flip", True)),
        "numeric_columns": list(data_overrides.get("numeric_columns", NUMERIC_COLUMNS)),
        "categorical_columns": list(
            data_overrides.get("categorical_columns", CAT_COLUMNS)
        ),
        "model": {
            "epochs": int(model_overrides.get("epochs", 50)),
            "learning_rate": learning_rate,
            "batch_size": train_batch_size,
            "layers": [int(value) for value in model_overrides.get("layers", [128, 128])],
            "optimizer": str(model_overrides.get("optimizer", "adam")),
            "criterion": str(model_overrides.get("criterion", "cross_entropy")),
            "output_activation": str(
                model_overrides.get("output_activation", "softmax")
            ),
        },
        "method": {
            "seed": int(method_overrides.get("seed", experiment_cfg.get("seed", 0))),
            "desired_class": desired_class,
            "cf_confidence": cf_confidence,
            "cf_steps": cf_steps,
            "cf_step_size": cf_step_size,
            "k": k,
            "sigma": sigma,
            "trex_step_size": trex_step_size,
            "trex_max_steps": trex_max_steps,
            "trex_epsilon": trex_epsilon,
            "trex_p": method_overrides.get("trex_p", 2),
            "batch_size": int(method_overrides.get("batch_size", 1)),
            "clamp": method_overrides.get("clamp", True),
            "tau_grid": tau_grid,
            "tau_l1": None if tau_l1 is None else float(tau_l1),
            "tau_l2": None if tau_l2 is None else float(tau_l2),
        },
        "evaluation": {
            "wi_models": wi_models,
            "lo_models": lo_models,
            "pilot_model_count": pilot_model_count,
            "pilot_factual_limit": pilot_factual_limit,
            "factual_limit": factual_limit,
            "lof_n_neighbors": lof_neighbors,
            "paper_targets": evaluation_cfg.get("paper_targets", PAPER_TARGETS),
        },
    }

    if settings["test_split"] <= 0 or settings["test_split"] >= 1:
        raise ValueError("data.overrides.test_split must satisfy 0 < split < 1")
    if settings["evaluation"]["wi_models"] < 1 or settings["evaluation"]["lo_models"] < 1:
        raise ValueError("WI and LO model counts must be >= 1")
    if settings["evaluation"]["pilot_model_count"] < 1:
        raise ValueError("pilot_model_count must be >= 1")
    if settings["evaluation"]["pilot_factual_limit"] < 1:
        raise ValueError("pilot_factual_limit must be >= 1")
    if settings["evaluation"]["factual_limit"] is not None and settings["evaluation"]["factual_limit"] < 1:
        raise ValueError("factual_limit must be >= 1 when provided")
    if settings["model"]["batch_size"] < 1:
        raise ValueError("model batch_size must be >= 1")
    if settings["model"]["learning_rate"] <= 0:
        raise ValueError("model learning_rate must be > 0")
    if not settings["method"]["tau_grid"] and settings["method"]["tau_l1"] is None:
        raise ValueError("Provide tau_grid or tau_l1 in config/CLI")

    return settings


def _resolve_device(device: str) -> str:
    device = device.lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in the current environment")
    return device


def _make_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    name: str,
    **flags: object,
) -> FrameDataset:
    combined = pd.concat([features, target.rename(TARGET_COLUMN)], axis=1)
    dataset = FrameDataset(combined, target_column=TARGET_COLUMN, name=name)
    for flag, value in flags.items():
        dataset.update(flag, value)
    dataset.freeze()
    return dataset


def _count_values(series: pd.Series) -> dict[int, int]:
    counts = series.astype(int).value_counts().sort_index()
    return {int(index): int(value) for index, value in counts.items()}


def _build_categorical_frame(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for column in categorical_columns:
        categories = sorted(pd.Index(df[column].unique()).tolist())
        cat_series = pd.Categorical(df[column], categories=categories)
        encoded = pd.get_dummies(cat_series, prefix=column, prefix_sep="_")
        ordered_columns = [f"{column}_{category}" for category in categories]
        encoded = encoded.reindex(columns=ordered_columns, fill_value=0)
        frames.append(encoded.astype(np.float32))
    return pd.concat(frames, axis=1)


def load_german_reproduction_data(
    *,
    raw_data_path: Path,
    split_seed: int,
    test_split: float,
    label_flip: bool,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    raw_df = pd.read_csv(raw_data_path)
    raw_target = raw_df[TARGET_COLUMN].astype(int)
    target = (1 - raw_target) if label_flip else raw_target

    numeric = raw_df.loc[:, numeric_columns].astype(np.float32)
    denom = numeric.max(axis=0) - numeric.min(axis=0)
    denom = denom.replace(0, 1)
    numeric = ((numeric - numeric.min(axis=0)) / denom).astype(np.float32)

    categorical = _build_categorical_frame(raw_df, categorical_columns)
    features = pd.concat([numeric, categorical], axis=1).astype(np.float32)
    feature_order = list(features.columns)

    if features.shape[1] != EXPECTED_INPUT_DIM:
        raise ValueError(
            f"Expected {EXPECTED_INPUT_DIM} processed features, "
            f"found {features.shape[1]}"
        )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        features,
        target,
        test_size=test_split,
        random_state=split_seed,
        shuffle=True,
    )

    trainset = _make_dataset(
        X_train,
        y_train,
        name="german_train_reproduction",
        trainset=True,
    )
    testset = _make_dataset(
        X_test,
        y_test,
        name="german_test_reproduction",
        testset=True,
    )

    return {
        "raw_df": raw_df,
        "feature_order": feature_order,
        "trainset": trainset,
        "testset": testset,
        "train_features": X_train.copy(deep=True),
        "test_features": X_test.copy(deep=True),
        "train_target": y_train.copy(deep=True),
        "test_target": y_test.copy(deep=True),
        "input_dim": int(features.shape[1]),
        "full_target_counts": _count_values(target),
        "train_target_counts": _count_values(y_train),
        "test_target_counts": _count_values(y_test),
    }


def _build_model(
    *,
    seed: int,
    device: str,
    model_cfg: dict[str, Any],
) -> MlpModel:
    return MlpModel(
        seed=seed,
        device=device,
        epochs=int(model_cfg["epochs"]),
        learning_rate=float(model_cfg["learning_rate"]),
        batch_size=int(model_cfg["batch_size"]),
        layers=[int(value) for value in model_cfg["layers"]],
        optimizer=str(model_cfg["optimizer"]),
        criterion=str(model_cfg["criterion"]),
        output_activation=str(model_cfg["output_activation"]),
        save_name=None,
    )


def train_model_bundle(
    trainset: FrameDataset,
    testset: FrameDataset,
    *,
    seed: int,
    device: str,
    model_cfg: dict[str, Any],
    variant: str,
) -> ModelBundle:
    model = _build_model(
        seed=seed,
        device=device,
        model_cfg=model_cfg,
    )
    model.fit(trainset)
    _ = compute_model_metrics(model, testset)
    return ModelBundle(model=model, seed=seed, variant=variant)


def compute_model_metrics(model: MlpModel, testset: FrameDataset) -> dict[str, float]:
    probabilities = model.predict_proba(testset).detach().cpu()
    prediction = probabilities.argmax(dim=1)

    y = testset.get(target=True).iloc[:, 0].astype(int)
    class_to_index = model.get_class_to_index()
    encoded_target = torch.tensor(
        [class_to_index[int(value)] for value in y.tolist()],
        dtype=torch.long,
    )

    accuracy = float((prediction == encoded_target).to(dtype=torch.float32).mean())
    unique_labels = sorted(set(encoded_target.tolist()))
    if len(unique_labels) < 2:
        auc = float("nan")
    else:
        positive_index = class_to_index.get(1, max(class_to_index.values()))
        auc = float(
            roc_auc_score(
                encoded_target.numpy(),
                probabilities[:, positive_index].numpy(),
            )
        )
    return {"test_accuracy": accuracy, "test_auc": auc}


def select_rejected_factuals(
    model: MlpModel,
    testset: FrameDataset,
    *,
    desired_class: int,
) -> FrameDataset:
    probabilities = model.predict_proba(testset).detach().cpu()
    predicted = probabilities.argmax(dim=1).numpy()
    labels = testset.get(target=True).iloc[:, 0].astype(int)
    undesired_class = 1 - int(desired_class)
    rejected_mask = pd.Series(
        (labels.to_numpy() == undesired_class) & (predicted == undesired_class),
        index=testset.get(target=False).index,
        dtype=bool,
    )

    factual_features = testset.get(target=False).loc[rejected_mask].copy(deep=True)
    factual_target = testset.get(target=True).iloc[:, 0].loc[rejected_mask].copy(deep=True)
    factuals = _make_dataset(
        factual_features,
        factual_target,
        name="german_rejected_factuals",
        testset=True,
    )
    return factuals


def build_lo_trainset(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    *,
    sample_seed: int,
) -> FrameDataset:
    sample_count = max(1, int(round(0.01 * len(train_features))))
    rng = np.random.default_rng(sample_seed)
    sampled_positions = rng.choice(len(train_features), size=sample_count, replace=True)
    unique_positions = np.unique(sampled_positions)
    keep_mask = np.ones(len(train_features), dtype=bool)
    keep_mask[unique_positions] = False

    lo_features = train_features.iloc[keep_mask].copy(deep=True)
    lo_target = train_target.iloc[keep_mask].copy(deep=True)
    return _make_dataset(
        lo_features,
        lo_target,
        name=f"german_lo_train_{sample_seed}",
        trainset=True,
    )


def train_changed_models(
    *,
    trainset: FrameDataset,
    testset: FrameDataset,
    train_features: pd.DataFrame,
    train_target: pd.Series,
    wi_models: int,
    lo_models: int,
    device: str,
    model_cfg: dict[str, Any],
) -> tuple[list[ModelBundle], list[ModelBundle]]:
    wi_bundles: list[ModelBundle] = []
    lo_bundles: list[ModelBundle] = []

    for offset in tqdm(
        range(wi_models),
        desc="Training WI models",
        unit="model",
        leave=False,
    ):
        seed = offset + 1
        wi_bundles.append(
            train_model_bundle(
                trainset,
                testset,
                seed=seed,
                device=device,
                model_cfg=model_cfg,
                variant="wi",
            )
        )

    for offset in tqdm(
        range(lo_models),
        desc="Training LO models",
        unit="model",
        leave=False,
    ):
        seed = 1001 + offset
        lo_trainset = build_lo_trainset(
            train_features,
            train_target,
            sample_seed=5001 + offset,
        )
        lo_bundles.append(
            train_model_bundle(
                lo_trainset,
                testset,
                seed=seed,
                device=device,
                model_cfg=model_cfg,
                variant="lo",
            )
        )

    return wi_bundles, lo_bundles


def _valid_counterfactual_mask(counterfactuals: FrameDataset) -> pd.Series:
    cf_features = counterfactuals.get(target=False)
    return ~cf_features.isna().any(axis=1)


def _distance_mean(
    factual_features: pd.DataFrame,
    counterfactual_features: pd.DataFrame,
    *,
    norm: int,
) -> float:
    diff = counterfactual_features.to_numpy(dtype=np.float32) - factual_features.to_numpy(
        dtype=np.float32
    )
    if norm == 1:
        values = np.linalg.norm(diff, ord=1, axis=1)
    elif norm == 2:
        values = np.linalg.norm(diff, ord=2, axis=1)
    else:
        raise ValueError("norm must be 1 or 2")
    return float(np.mean(values))


def _predict_counterfactuals_with_progress(
    method: TrexMethod,
    testset: FrameDataset,
    *,
    batch_size: int,
    desc: str,
) -> FrameDataset:
    if not method._is_trained:
        raise RuntimeError("Method is not trained")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if getattr(testset, "counterfactual", False):
        raise ValueError("testset must not already be marked as counterfactual")

    factuals = testset.get(target=False)
    counterfactual_batches: list[pd.DataFrame] = []

    with tqdm(
        total=factuals.shape[0],
        desc=desc,
        unit="cf",
        leave=False,
    ) as progress:
        for start in range(0, factuals.shape[0], batch_size):
            batch = factuals.iloc[start : start + batch_size]
            counterfactual_batch = method.get_counterfactuals(batch)

            if counterfactual_batch.shape[0] != batch.shape[0]:
                raise ValueError(
                    "get_counterfactuals() must preserve the input row count"
                )
            if set(counterfactual_batch.columns) != set(batch.columns):
                raise ValueError(
                    "get_counterfactuals() must preserve the input feature columns"
                )

            counterfactual_batch = counterfactual_batch.reindex(
                index=batch.index,
                columns=batch.columns,
            )
            counterfactual_batches.append(counterfactual_batch)
            progress.update(batch.shape[0])

    if counterfactual_batches:
        counterfactual_features = pd.concat(counterfactual_batches, axis=0)
        counterfactual_features = counterfactual_features.reindex(index=factuals.index)
    else:
        counterfactual_features = factuals.iloc[0:0].copy(deep=True)

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
    counterfactual_df = counterfactual_df.reindex(columns=testset.ordered_features())

    output = testset.clone()
    output.update("counterfactual", True, df=counterfactual_df)

    if method._desired_class is not None:
        class_to_index = method._target_model.get_class_to_index()
        prediction = method._target_model.predict(testset, batch_size=batch_size)
        predicted_label = prediction.argmax(dim=1).cpu().numpy()
        evaluation_filter = pd.DataFrame(
            predicted_label != class_to_index[method._desired_class],
            index=counterfactual_df.index,
            columns=["evaluation_filter"],
            dtype=bool,
        )
        output.update("evaluation_filter", evaluation_filter)

    output.freeze()
    return output


def evaluate_changed_model_validity(
    bundles: list[ModelBundle],
    *,
    valid_counterfactuals: pd.DataFrame,
    denominator: int,
    desired_class: int,
    desc: str,
) -> float:
    if denominator == 0:
        return float("nan")
    if valid_counterfactuals.shape[0] == 0:
        return 0.0

    results: list[float] = []
    for bundle in tqdm(
        bundles,
        desc=desc,
        unit="model",
        leave=False,
    ):
        probabilities = bundle.model.get_prediction(valid_counterfactuals, proba=True)
        predicted = probabilities.argmax(dim=1).detach().cpu().numpy()
        success_count = int(np.sum(predicted == int(desired_class)))
        results.append(100.0 * success_count / denominator)
    return float(np.mean(results)) if results else float("nan")


def evaluate_run(
    *,
    target_model: MlpModel,
    trainset: FrameDataset,
    factuals: FrameDataset,
    wi_bundles: list[ModelBundle],
    lo_bundles: list[ModelBundle],
    norm: int,
    tau: float,
    device: str,
    method_cfg: dict[str, Any],
    lof_n_neighbors: int,
) -> RunMetrics:
    method = TrexMethod(
        target_model=target_model,
        seed=int(method_cfg["seed"]),
        device=device,
        desired_class=int(method_cfg["desired_class"]),
        norm=norm,
        cf_confidence=float(method_cfg["cf_confidence"]),
        cf_steps=int(method_cfg["cf_steps"]),
        cf_step_size=float(method_cfg["cf_step_size"]),
        tau=tau,
        k=int(method_cfg["k"]),
        sigma=float(method_cfg["sigma"]),
        trex_step_size=float(method_cfg["trex_step_size"]),
        trex_max_steps=int(method_cfg["trex_max_steps"]),
        trex_epsilon=float(method_cfg["trex_epsilon"]),
        trex_p=method_cfg["trex_p"],
        batch_size=int(method_cfg["batch_size"]),
        clamp=method_cfg["clamp"],
    )
    method.fit(trainset)
    counterfactuals = _predict_counterfactuals_with_progress(
        method,
        factuals,
        batch_size=int(method_cfg["batch_size"]),
        desc=f"Generating L{norm} counterfactuals",
    )

    factual_features = factuals.get(target=False)
    counterfactual_features = counterfactuals.get(target=False)
    valid_mask = _valid_counterfactual_mask(counterfactuals)

    denominator = int(len(factual_features))
    num_valid = int(valid_mask.sum())
    current_validity_pct = (
        100.0 * num_valid / denominator if denominator > 0 else float("nan")
    )

    if num_valid == 0:
        cost = float("nan")
        lof_value = float("nan")
        valid_counterfactuals = counterfactual_features.iloc[0:0].copy(deep=True)
    else:
        factual_valid = factual_features.loc[valid_mask].copy(deep=True)
        valid_counterfactuals = counterfactual_features.loc[valid_mask].copy(deep=True)
        cost = _distance_mean(factual_valid, valid_counterfactuals, norm=norm)

        lof = LocalOutlierFactor(n_neighbors=lof_n_neighbors, novelty=True)
        lof.fit(trainset.get(target=False).to_numpy(dtype=np.float32))
        lof_value = float(
            lof.predict(valid_counterfactuals.to_numpy(dtype=np.float32)).mean()
        )

    wi_validity_pct = evaluate_changed_model_validity(
        wi_bundles,
        valid_counterfactuals=valid_counterfactuals,
        denominator=denominator,
        desired_class=int(method_cfg["desired_class"]),
        desc=f"WI validity L{norm}",
    )
    lo_validity_pct = evaluate_changed_model_validity(
        lo_bundles,
        valid_counterfactuals=valid_counterfactuals,
        denominator=denominator,
        desired_class=int(method_cfg["desired_class"]),
        desc=f"LO validity L{norm}",
    )

    return RunMetrics(
        tau=float(tau),
        norm=int(norm),
        current_validity_pct=float(current_validity_pct),
        cost=float(cost),
        lof=float(lof_value),
        wi_validity_pct=float(wi_validity_pct),
        lo_validity_pct=float(lo_validity_pct),
        num_factuals=denominator,
        num_valid_counterfactuals=num_valid,
    )


def _metric_gap(value: float, target: float) -> float:
    if math.isnan(value):
        return float("inf")
    return abs(value - target)


def _candidate_summary(metrics: RunMetrics, paper_target: dict[str, float]) -> dict[str, Any]:
    wi_gap = _metric_gap(metrics.wi_validity_pct, paper_target["wi_validity_pct"])
    lo_gap = _metric_gap(metrics.lo_validity_pct, paper_target["lo_validity_pct"])
    cost_gap = _metric_gap(metrics.cost, paper_target["cost"])
    lof_gap = _metric_gap(metrics.lof, paper_target["lof"])
    return {
        "metrics": metrics,
        "wi_gap": wi_gap,
        "lo_gap": lo_gap,
        "cost_gap": cost_gap,
        "lof_gap": lof_gap,
        "validity_gap_sum": wi_gap + lo_gap,
    }


def choose_tau(
    *,
    norm: int,
    tau_grid: list[float],
    paper_target: dict[str, float],
    target_model: MlpModel,
    trainset: FrameDataset,
    pilot_factuals: FrameDataset,
    pilot_wi_bundles: list[ModelBundle],
    pilot_lo_bundles: list[ModelBundle],
    device: str,
    method_cfg: dict[str, Any],
    lof_n_neighbors: int,
) -> tuple[float, list[dict[str, Any]]]:
    candidate_results: list[dict[str, Any]] = []

    for tau in tqdm(
        tau_grid,
        desc=f"Tuning tau (L{norm})",
        unit="tau",
        leave=False,
    ):
        metrics = evaluate_run(
            target_model=target_model,
            trainset=trainset,
            factuals=pilot_factuals,
            wi_bundles=pilot_wi_bundles,
            lo_bundles=pilot_lo_bundles,
            norm=norm,
            tau=tau,
            device=device,
            method_cfg=method_cfg,
            lof_n_neighbors=lof_n_neighbors,
        )
        candidate_results.append(_candidate_summary(metrics, paper_target))

    valid_candidates = [
        item
        for item in candidate_results
        if item["wi_gap"] <= 3.0 and item["lo_gap"] <= 3.0
    ]
    if valid_candidates:
        best = min(
            valid_candidates,
            key=lambda item: (
                item["cost_gap"],
                item["validity_gap_sum"],
                item["lof_gap"],
                item["metrics"].tau,
            ),
        )
    else:
        best = min(
            candidate_results,
            key=lambda item: (
                item["validity_gap_sum"],
                item["cost_gap"],
                item["lof_gap"],
                item["metrics"].tau,
            ),
        )

    serialized_candidates = []
    for item in candidate_results:
        payload = item["metrics"].to_dict()
        payload["gaps"] = {
            "wi_validity_pct": float(item["wi_gap"]),
            "lo_validity_pct": float(item["lo_gap"]),
            "cost": float(item["cost_gap"]),
            "lof": float(item["lof_gap"]),
            "validity_gap_sum": float(item["validity_gap_sum"]),
        }
        serialized_candidates.append(payload)

    return float(best["metrics"].tau), serialized_candidates


def _pilot_subset(factuals: FrameDataset, limit: int) -> FrameDataset:
    features = factuals.get(target=False)
    target = factuals.get(target=True).iloc[:, 0]
    selected_index = sorted(features.index.tolist())[: min(limit, len(features))]
    pilot_features = features.loc[selected_index].copy(deep=True)
    pilot_target = target.loc[selected_index].copy(deep=True)
    return _make_dataset(
        pilot_features,
        pilot_target,
        name="german_pilot_factuals",
        testset=True,
    )


def _subset_factuals(factuals: FrameDataset, limit: int | None, *, name: str) -> FrameDataset:
    if limit is None or limit >= len(factuals):
        return factuals

    features = factuals.get(target=False)
    target = factuals.get(target=True).iloc[:, 0]
    selected_index = sorted(features.index.tolist())[:limit]
    subset_features = features.loc[selected_index].copy(deep=True)
    subset_target = target.loc[selected_index].copy(deep=True)
    return _make_dataset(
        subset_features,
        subset_target,
        name=name,
        testset=True,
    )


def _report_row(metrics: RunMetrics, paper_target: dict[str, float]) -> dict[str, Any]:
    payload = metrics.to_dict()
    payload["paper_target"] = dict(paper_target)
    payload["gaps"] = {
        "cost": _metric_gap(metrics.cost, paper_target["cost"]),
        "lof": _metric_gap(metrics.lof, paper_target["lof"]),
        "wi_validity_pct": _metric_gap(
            metrics.wi_validity_pct, paper_target["wi_validity_pct"]
        ),
        "lo_validity_pct": _metric_gap(
            metrics.lo_validity_pct, paper_target["lo_validity_pct"]
        ),
    }
    return payload


def _print_summary(
    *,
    output: dict[str, Any],
) -> None:
    baseline = output["baseline"]
    data = output["data"]
    model_cfg = output["model"]
    method_cfg = output["method"]
    evaluation_cfg = output["evaluation"]

    print("TreX German reproduction")
    print(f"config: {output['config_path']}")
    print(f"device: {output['device']}")
    print(
        f"processed_dim: {data['input_dim']} | train/test: "
        f"{data['train_size']}/{data['test_size']} | rejected: {data['num_rejected_factuals']}"
    )
    print(
        f"baseline accuracy: {baseline['test_accuracy']:.4f} | "
        f"baseline auc: {baseline['test_auc']:.4f}"
    )
    print(
        "model cfg: "
        f"layers={model_cfg['layers']} | epochs={model_cfg['epochs']} | "
        f"lr={model_cfg['learning_rate']:.4f} | batch={model_cfg['batch_size']}"
    )
    print(
        "trex cfg: "
        f"tau_l1={method_cfg['tau_l1']:.4f} | sigma={method_cfg['sigma']:.4f} | "
        f"k={method_cfg['k']} | cf_steps={method_cfg['cf_steps']} | "
        f"cf_step_size={method_cfg['cf_step_size']:.4f} | "
        f"trex_steps={method_cfg['trex_max_steps']} | "
        f"trex_step_size={method_cfg['trex_step_size']:.4f}"
    )
    print(
        "eval cfg: "
        f"wi_models={evaluation_cfg['wi_models']} | "
        f"lo_models={evaluation_cfg['lo_models']} | "
        f"factual_limit={evaluation_cfg['factual_limit']}"
    )
    print("")

    rows = []
    for norm_key in ["l1"]:#, "l2"):
        row = output[norm_key]
        rows.append(
            {
                "norm": norm_key,
                "tau": row["tau"],
                "current_validity_pct": row["current_validity_pct"],
                "cost": row["cost"],
                "lof": row["lof"],
                "wi_validity_pct": row["wi_validity_pct"],
                "lo_validity_pct": row["lo_validity_pct"],
            }
        )
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("")
    for norm_key in ["l1"]:#, "l2"):
        row = output[norm_key]
        gaps = row["gaps"]
        print(
            f"{norm_key} target gaps | cost: {gaps['cost']:.4f} | lof: {gaps['lof']:.4f} | "
            f"wi: {gaps['wi_validity_pct']:.4f} | lo: {gaps['lo_validity_pct']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--wi-models", type=int, default=None)
    parser.add_argument("--lo-models", type=int, default=None)
    parser.add_argument("--pilot-model-count", type=int, default=None)
    parser.add_argument("--pilot-factual-limit", type=int, default=None)
    parser.add_argument("--factual-limit", type=int, default=None)
    parser.add_argument(
        "--tau-grid",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument("--tau-l1", type=float, default=None)
    parser.add_argument("--tau-l2", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--cf-confidence", type=float, default=None)
    parser.add_argument("--cf-steps", type=int, default=None)
    parser.add_argument("--cf-step-size", type=float, default=None)
    parser.add_argument("--trex-step-size", type=float, default=None)
    parser.add_argument("--trex-max-steps", type=int, default=None)
    parser.add_argument("--trex-epsilon", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    config = _load_config(Path(args.config).resolve())
    settings = _resolve_settings(config, args)

    with tqdm(total=5, desc="Reproduction stages", unit="stage") as stage_bar:
        stage_bar.set_postfix_str("load data")
        data = load_german_reproduction_data(
            raw_data_path=settings["raw_data_path"],
            split_seed=settings["split_seed"],
            test_split=settings["test_split"],
            label_flip=settings["label_flip"],
            numeric_columns=settings["numeric_columns"],
            categorical_columns=settings["categorical_columns"],
        )
        trainset = data["trainset"]
        testset = data["testset"]
        stage_bar.update(1)

        stage_bar.set_postfix_str("train baseline")
        baseline_bundle = train_model_bundle(
            trainset,
            testset,
            seed=settings["seed"],
            device=settings["device"],
            model_cfg=settings["model"],
            variant="baseline",
        )
        baseline_metrics = compute_model_metrics(baseline_bundle.model, testset)
        stage_bar.update(1)

        stage_bar.set_postfix_str("select factuals")
        factuals = select_rejected_factuals(
            baseline_bundle.model,
            testset,
            desired_class=int(settings["method"]["desired_class"]),
        )
        if len(factuals) == 0:
            raise RuntimeError("No rejected factuals were found for reproduction")
        factuals = _subset_factuals(
            factuals,
            settings["evaluation"]["factual_limit"],
            name="german_rejected_factuals_subset",
        )
        stage_bar.update(1)

        stage_bar.set_postfix_str("train changed models")
        wi_bundles, lo_bundles = train_changed_models(
            trainset=trainset,
            testset=testset,
            train_features=data["train_features"],
            train_target=data["train_target"],
            wi_models=settings["evaluation"]["wi_models"],
            lo_models=settings["evaluation"]["lo_models"],
            device=settings["device"],
            model_cfg=settings["model"],
        )
        stage_bar.update(1)

        stage_bar.set_postfix_str("evaluate norms")
        pilot_factuals = _pilot_subset(
            factuals,
            settings["evaluation"]["pilot_factual_limit"],
        )
        pilot_wi_bundles = wi_bundles[
            : min(settings["evaluation"]["pilot_model_count"], len(wi_bundles))
        ]
        pilot_lo_bundles = lo_bundles[
            : min(settings["evaluation"]["pilot_model_count"], len(lo_bundles))
        ]

        norm_results: dict[str, dict[str, Any]] = {}
        manual_taus = {
            "l1": settings["method"]["tau_l1"],
            #"l2": settings["method"]["tau_l2"],
        }
        for norm_key, norm in tqdm(
            [("l1", 1)],#, ("l2", 2)],
            desc="Final norm evaluations",
            unit="norm",
            leave=False,
        ):
            paper_target = settings["evaluation"]["paper_targets"][norm_key]
            if manual_taus[norm_key] is None:
                tau, pilot_candidates = choose_tau(
                    norm=norm,
                    tau_grid=[float(value) for value in settings["method"]["tau_grid"]],
                    paper_target=paper_target,
                    target_model=baseline_bundle.model,
                    trainset=trainset,
                    pilot_factuals=pilot_factuals,
                    pilot_wi_bundles=pilot_wi_bundles,
                    pilot_lo_bundles=pilot_lo_bundles,
                    device=settings["device"],
                    method_cfg=settings["method"],
                    lof_n_neighbors=settings["evaluation"]["lof_n_neighbors"],
                )
            else:
                tau = float(manual_taus[norm_key])
                pilot_candidates = []

            metrics = evaluate_run(
                target_model=baseline_bundle.model,
                trainset=trainset,
                factuals=factuals,
                wi_bundles=wi_bundles,
                lo_bundles=lo_bundles,
                norm=norm,
                tau=tau,
                device=settings["device"],
                method_cfg=settings["method"],
                lof_n_neighbors=settings["evaluation"]["lof_n_neighbors"],
            )
            report = _report_row(metrics, paper_target)
            report["pilot_candidates"] = pilot_candidates
            norm_results[norm_key] = report
        stage_bar.update(1)

    elapsed_seconds = time.perf_counter() - start_time
    output = {
        "config_path": settings["config_path"],
        "experiment_name": settings["experiment_name"],
        "device": settings["device"],
        "elapsed_seconds": elapsed_seconds,
        "baseline": baseline_metrics,
        "model": {
            "epochs": int(settings["model"]["epochs"]),
            "learning_rate": float(settings["model"]["learning_rate"]),
            "batch_size": int(settings["model"]["batch_size"]),
            "layers": [int(value) for value in settings["model"]["layers"]],
        },
        "method": {
            "tau_l1": float(settings["method"]["tau_l1"]),
            "sigma": float(settings["method"]["sigma"]),
            "k": int(settings["method"]["k"]),
            "cf_steps": int(settings["method"]["cf_steps"]),
            "cf_step_size": float(settings["method"]["cf_step_size"]),
            "trex_max_steps": int(settings["method"]["trex_max_steps"]),
            "trex_step_size": float(settings["method"]["trex_step_size"]),
        },
        "evaluation": {
            "wi_models": int(settings["evaluation"]["wi_models"]),
            "lo_models": int(settings["evaluation"]["lo_models"]),
            "factual_limit": settings["evaluation"]["factual_limit"],
        },
        "data": {
            "raw_path": str(settings["raw_data_path"]),
            "data_name": settings["data_name"],
            "input_dim": data["input_dim"],
            "train_size": int(len(trainset)),
            "test_size": int(len(testset)),
            "full_target_counts": data["full_target_counts"],
            "train_target_counts": data["train_target_counts"],
            "test_target_counts": data["test_target_counts"],
            "num_rejected_factuals": int(len(factuals)),
            "num_pilot_factuals": int(len(pilot_factuals)),
            "factual_limit": settings["evaluation"]["factual_limit"],
            "split_seed": int(settings["split_seed"]),
            "test_split": float(settings["test_split"]),
            "label_transform": (
                "paper_target = 1 - raw_credit_risk"
                if settings["label_flip"]
                else "paper_target = raw_credit_risk"
            ),
        },
        "l1": norm_results["l1"],
        # "l2": norm_results["l2"],
    }

    _print_summary(output=output)


if __name__ == "__main__":
    main()
