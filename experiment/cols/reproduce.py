from __future__ import annotations

import argparse
import copy
import sys
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
from sklearn.model_selection import ShuffleSplit

from dataset.compas_carla.compas_carla import CompasCarlaDataset
from method.cols.cols import ColsMethod
from method.cols.support import decode_feature_dataframe
from model.mlp.mlp import MlpModel
from model.model_utils import build_optimizer
from model.model_object import process_nan
from utils.seed import seed_context

EPSILON = 1e-7


@dataclass(frozen=True)
class ScalingStats:
    minimum: dict[str, float]
    maximum: dict[str, float]


@dataclass(frozen=True)
class SplitArtifacts:
    balanced_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    provisional_factual_df: pd.DataFrame
    provisional_unique_count: int


@dataclass(frozen=True)
class PaperMetricContext:
    feature_names: list[str]
    continuous_feature_names: list[str]
    categorical_feature_names: list[str]
    continuous_feature_indexes: list[int]
    categorical_feature_indexes: list[int]
    feature_types: dict[str, str]
    feature_change_restriction: dict[str, int]
    feature_values: dict[str, list[object]]
    original_ranges: dict[str, list[object]]
    percentiles: dict[str, dict[object, float]]
    mads: dict[str, float]
    cost_map: dict[str, dict[object, int]]
    invalid_cost: float
    variance: float


class ReferenceCompasMlpModel(MlpModel):
    def __init__(
        self,
        *args,
        reference_checkpoint_path: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._best_val_accuracy: float | None = None
        self._best_val_loss: float | None = None
        self._reference_checkpoint_path = reference_checkpoint_path

    def _encode_target_series(self, y: pd.Series) -> torch.Tensor:
        class_to_index = self.get_class_to_index()
        encoded = []
        for value in y.tolist():
            if isinstance(value, float) and float(value).is_integer():
                encoded.append(class_to_index[int(value)])
            else:
                encoded.append(class_to_index[value])
        return torch.tensor(encoded, dtype=torch.long, device=self._device)

    def _build_model(self, input_dim: int, output_dim: int) -> torch.nn.Module:
        hidden_dim = int(self._layers[0]) if self._layers else 20
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.Softmax(dim=1),
        )

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Target model is not trained")
        with seed_context(self._seed):
            self._model.eval()
            X_tensor = torch.tensor(
                X.to_numpy(dtype="float32"),
                dtype=torch.float32,
                device=self._device,
            )
            with torch.no_grad():
                probabilities = self._model(X_tensor)
            if proba:
                return probabilities.detach().cpu()
            indices = probabilities.argmax(dim=1)
            return torch.nn.functional.one_hot(
                indices,
                num_classes=probabilities.shape[1],
            ).to(dtype=torch.float32).detach().cpu()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._is_trained or self._model is None:
            raise RuntimeError("Target model is not trained")
        with seed_context(self._seed):
            self._model.eval()
            return self._model(X.to(self._device))

    def fit(
        self,
        trainset,
        valset=None,
    ):
        if trainset is None:
            raise ValueError("trainset is required for ReferenceCompasMlpModel.fit()")
        if valset is None:
            raise ValueError("valset is required for ReferenceCompasMlpModel.fit()")

        with seed_context(self._seed):
            X_train, labels_train, output_dim = self.extract_training_data(trainset)
            input_dim = X_train.shape[1]
            self._output_dim = output_dim
            self._model = self._build_model(input_dim, output_dim).to(self._device)

            optimizer = build_optimizer(
                self._optimizer_name,
                self._model.parameters(),
                self._learning_rate,
            )
            criterion = torch.nn.CrossEntropyLoss()

            X_train_tensor = torch.tensor(
                X_train.to_numpy(dtype="float32"),
                dtype=torch.float32,
                device=self._device,
            )
            y_train_tensor = labels_train.to(self._device)

            X_val = valset.get(target=False)
            y_val = valset.get(target=True).iloc[:, 0]
            X_val_tensor = torch.tensor(
                X_val.to_numpy(dtype="float32"),
                dtype=torch.float32,
                device=self._device,
            )
            y_val_tensor = self._encode_target_series(y_val)

            if self._reference_checkpoint_path is not None:
                checkpoint = torch.load(
                    self._reference_checkpoint_path,
                    map_location=self._device,
                )
                reference_state = checkpoint["state_dict"]
                mapped_state = {
                    "0.weight": reference_state["model.layers.0.weight"],
                    "0.bias": reference_state["model.layers.0.bias"],
                    "2.weight": reference_state["model.layers.2.weight"],
                    "2.bias": reference_state["model.layers.2.bias"],
                }
                self._model.load_state_dict(mapped_state)
                self._model.eval()
                with torch.no_grad():
                    val_probs = self._model(X_val_tensor)
                    val_loss = float(criterion(val_probs, y_val_tensor).item())
                    val_prediction = val_probs.argmax(dim=1)
                    val_accuracy = float(
                        (val_prediction == y_val_tensor)
                        .to(dtype=torch.float32)
                        .mean()
                        .item()
                    )
                self._is_trained = True
                self._best_val_accuracy = val_accuracy
                self._best_val_loss = val_loss
                return

            best_state: dict[str, torch.Tensor] | None = None
            best_val_accuracy = float("-inf")
            best_val_loss = float("inf")

            for _ in range(self._epochs):
                self._model.train()
                permutation = torch.randperm(
                    X_train_tensor.shape[0],
                    device=self._device,
                )
                for start in range(0, X_train_tensor.shape[0], self._batch_size):
                    batch_indices = permutation[start : start + self._batch_size]
                    batch_X = X_train_tensor[batch_indices]
                    batch_y = y_train_tensor[batch_indices]
                    optimizer.zero_grad()
                    probs = self._model(batch_X)
                    loss = criterion(probs, batch_y)
                    loss.backward()
                    optimizer.step()

                self._model.eval()
                with torch.no_grad():
                    val_probs = self._model(X_val_tensor)
                    val_loss = float(criterion(val_probs, y_val_tensor).item())
                    val_prediction = val_probs.argmax(dim=1)
                    val_accuracy = float(
                        (val_prediction == y_val_tensor)
                        .to(dtype=torch.float32)
                        .mean()
                        .item()
                    )

                if (
                    val_accuracy > best_val_accuracy + 1e-12
                    or (
                        abs(val_accuracy - best_val_accuracy) <= 1e-12
                        and val_loss < best_val_loss
                    )
                ):
                    best_state = copy.deepcopy(self._model.state_dict())
                    best_val_accuracy = val_accuracy
                    best_val_loss = val_loss

            if best_state is None:
                raise RuntimeError("Validation checkpoint selection failed")

            self._model.load_state_dict(best_state)
            self._model.eval()
            self._is_trained = True
            self._best_val_accuracy = best_val_accuracy
            self._best_val_loss = best_val_loss


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name.lower()


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    if args.max_runs is not None:
        cfg["reproduction"]["run_seeds"] = cfg["reproduction"]["run_seeds"][
            : int(args.max_runs)
        ]
    if args.max_factuals is not None:
        cfg["data"]["split"]["target_factual_count"] = int(args.max_factuals)
    if args.override_epochs is not None:
        cfg["model"]["epochs"] = int(args.override_epochs)
    if args.override_budget is not None:
        for method_cfg in cfg["methods"]:
            method_cfg["budget"] = int(args.override_budget)
    if args.override_num_mcmc is not None:
        for method_cfg in cfg["methods"]:
            method_cfg["num_mcmc"] = int(args.override_num_mcmc)
    if args.methods is not None:
        requested_methods = {
            method_name.strip()
            for method_name in str(args.methods).split(",")
            if method_name.strip()
        }
        cfg["methods"] = [
            method_cfg
            for method_cfg in cfg["methods"]
            if str(method_cfg["name"]) in requested_methods
        ]
        cfg["reproduction"]["paper_targets"] = {
            method_name: metric_cfg
            for method_name, metric_cfg in cfg["reproduction"]["paper_targets"].items()
            if method_name in requested_methods
        }
    if args.use_reference_checkpoint:
        cfg["model"]["use_reference_checkpoint"] = True
    return cfg


def _ordered_mapping(
    names: list[str],
    values: dict[str, Any],
) -> dict[str, Any]:
    return {name: values[name] for name in names}


def _load_raw_compas_dataframe(data_cfg: dict[str, Any]) -> pd.DataFrame:
    raw_path = (PROJECT_ROOT / data_cfg["raw_path"]).resolve()
    df = pd.read_csv(raw_path).dropna().reset_index(drop=True)
    target_column = data_cfg["target_column"]
    ordered_columns = list(data_cfg["raw_feature_order"]) + [target_column]
    return df.loc[:, ordered_columns].copy(deep=True)


def _uses_scalar_categorical_encoding(data_cfg: dict[str, Any]) -> bool:
    return str(data_cfg.get("encoding_mode", "onehot")).lower() == "scalar"


def _build_reference_split(
    raw_df: pd.DataFrame,
    data_cfg: dict[str, Any],
) -> SplitArtifacts:
    target_column = data_cfg["target_column"]
    balance_seed = int(data_cfg["balance"]["seed"])
    split_seed = int(data_cfg["split"]["split_seed"])
    val_fraction = float(data_cfg["split"]["val_fraction"])
    nominal_provisional_count = int(
        data_cfg["split"]["nominal_provisional_factual_count"]
    )

    labels = raw_df[target_column].to_numpy(dtype=int)
    rng = np.random.RandomState(balance_seed)
    balanced_indices = np.array([], dtype=int)
    minimum_class_count = int(np.min(np.bincount(labels)))
    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        balanced_indices = np.hstack(
            (balanced_indices, rng.choice(label_indices, minimum_class_count))
        )

    balanced_df = raw_df.iloc[balanced_indices].reset_index(drop=True)

    split_positions = np.arange(len(balanced_df))
    remaining_positions, val_positions = next(
        ShuffleSplit(n_splits=1, test_size=val_fraction, random_state=split_seed).split(
            split_positions
        )
    )

    remaining_labels = balanced_df.iloc[remaining_positions][target_column].to_numpy(
        dtype=int
    )
    negative_local_positions = np.where(remaining_labels == 0)[0]
    provisional_local_positions = rng.choice(
        negative_local_positions,
        nominal_provisional_count,
    )
    provisional_positions = remaining_positions[provisional_local_positions]
    provisional_unique_positions = set(int(position) for position in provisional_positions)
    train_positions = np.array(
        [
            int(position)
            for position in remaining_positions
            if int(position) not in provisional_unique_positions
        ],
        dtype=int,
    )

    train_df = balanced_df.iloc[train_positions].reset_index(drop=True)
    val_df = balanced_df.iloc[val_positions].reset_index(drop=True)
    provisional_factual_df = balanced_df.iloc[provisional_positions].reset_index(
        drop=True
    )

    return SplitArtifacts(
        balanced_df=balanced_df,
        train_df=train_df,
        val_df=val_df,
        provisional_factual_df=provisional_factual_df,
        provisional_unique_count=len(provisional_unique_positions),
    )


def _compute_scaling_stats(
    train_df: pd.DataFrame,
    continuous_feature_names: list[str],
) -> ScalingStats:
    minimum = {
        feature_name: float(train_df[feature_name].min())
        for feature_name in continuous_feature_names
    }
    maximum = {
        feature_name: float(train_df[feature_name].max())
        for feature_name in continuous_feature_names
    }
    return ScalingStats(minimum=minimum, maximum=maximum)


def _normalize_feature(
    values: pd.Series,
    minimum: float,
    maximum: float,
) -> pd.Series:
    if maximum == minimum:
        return pd.Series(0.0, index=values.index, dtype="float64")
    return (values.astype("float64") - minimum) / (maximum - minimum)


def _encode_raw_features(
    raw_df: pd.DataFrame,
    data_cfg: dict[str, Any],
    scaling_stats: ScalingStats,
) -> pd.DataFrame:
    encoded_columns: dict[str, pd.Series] = {}
    for feature_name in data_cfg["continuous_feature_order"]:
        encoded_columns[feature_name] = _normalize_feature(
            raw_df[feature_name],
            scaling_stats.minimum[feature_name],
            scaling_stats.maximum[feature_name],
        )

    for feature_name in data_cfg["categorical_feature_order"]:
        values = raw_df[feature_name].astype(str)
        if _uses_scalar_categorical_encoding(data_cfg):
            category_to_index = {
                str(category): index
                for index, category in enumerate(data_cfg["categorical_values"][feature_name])
            }
            encoded_columns[feature_name] = values.map(category_to_index).astype("float64")
            continue
        for category in data_cfg["categorical_values"][feature_name]:
            column_name = f"{feature_name}_cat_{category}"
            encoded_columns[column_name] = (values == str(category)).astype("float64")

    encoded_df = pd.DataFrame(encoded_columns, index=raw_df.index)
    return encoded_df.loc[:, data_cfg["encoded_feature_order"]].copy(deep=True)


def _decode_counterfactual_set(
    encoded_set: pd.DataFrame,
    method: ColsMethod,
    data_cfg: dict[str, Any],
    scaling_stats: ScalingStats,
) -> pd.DataFrame:
    decoded = decode_feature_dataframe(encoded_set, method._schema)
    decoded = decoded.loc[:, data_cfg["raw_feature_order"]].copy(deep=True)

    for feature_name in data_cfg["continuous_feature_order"]:
        minimum = scaling_stats.minimum[feature_name]
        maximum = scaling_stats.maximum[feature_name]
        if maximum == minimum:
            decoded[feature_name] = int(round(minimum))
            continue
        raw_values = (
            decoded[feature_name].astype("float64") * (maximum - minimum) + minimum
        )
        decoded[feature_name] = np.rint(raw_values).astype(int)

    for feature_name in data_cfg["categorical_feature_order"]:
        if _uses_scalar_categorical_encoding(data_cfg):
            categories = list(data_cfg["categorical_values"][feature_name])
            decoded[feature_name] = decoded[feature_name].map(
                lambda value: categories[
                    max(0, min(len(categories) - 1, int(round(float(value)))))
                ]
            )
        decoded[feature_name] = decoded[feature_name].astype(str)

    return decoded


def _build_dataset_template(data_cfg: dict[str, Any]) -> CompasCarlaDataset:
    template = CompasCarlaDataset(path=str((PROJECT_ROOT / "dataset/compas_carla").resolve()))
    raw_feature_names = list(data_cfg["raw_feature_order"]) + [data_cfg["target_column"]]
    template.update(
        "raw_feature_type",
        _ordered_mapping(raw_feature_names, data_cfg["raw_feature_type"]),
    )
    template.update(
        "raw_feature_mutability",
        _ordered_mapping(raw_feature_names, data_cfg["raw_feature_mutability"]),
    )
    template.update(
        "raw_feature_actionability",
        _ordered_mapping(raw_feature_names, data_cfg["raw_feature_actionability"]),
    )
    template.update("feature_order", list(data_cfg["encoded_feature_order"]) + [data_cfg["target_column"]])
    return template


def _build_encoding_map(data_cfg: dict[str, Any]) -> dict[str, list[str]]:
    return {
        feature_name: [
            f"{feature_name}_cat_{category}"
            for category in data_cfg["categorical_values"][feature_name]
        ]
        for feature_name in data_cfg["categorical_feature_order"]
    }


def _build_frozen_dataset(
    template: CompasCarlaDataset,
    feature_df: pd.DataFrame,
    target: pd.Series,
    data_cfg: dict[str, Any],
    marker: str,
):
    dataset = template.clone()
    if not _uses_scalar_categorical_encoding(data_cfg):
        dataset.update("encoding", _build_encoding_map(data_cfg))
    combined = pd.concat(
        [feature_df.reset_index(drop=True), target.reset_index(drop=True)],
        axis=1,
    )
    combined.columns = list(feature_df.columns) + [data_cfg["target_column"]]
    dataset.update(marker, True, df=combined)
    dataset.freeze()
    return dataset


def _predict_label_indices(model: MlpModel, X: pd.DataFrame) -> np.ndarray:
    return (
        model.get_prediction(X.loc[:, list(X.columns)], proba=True)
        .detach()
        .cpu()
        .numpy()
        .argmax(axis=1)
    )


def _build_metric_context(
    balanced_df: pd.DataFrame,
    train_df: pd.DataFrame,
    data_cfg: dict[str, Any],
    evaluation_cfg: dict[str, Any],
) -> PaperMetricContext:
    feature_names = list(data_cfg["raw_feature_order"])
    continuous_feature_names = list(data_cfg["continuous_feature_order"])
    categorical_feature_names = list(data_cfg["categorical_feature_order"])

    original_ranges: dict[str, list[object]] = {}
    for feature_name in feature_names:
        if feature_name in continuous_feature_names:
            minimum = int(balanced_df[feature_name].min())
            maximum = int(balanced_df[feature_name].max())
            original_ranges[feature_name] = list(range(minimum, maximum + 1))
        else:
            original_ranges[feature_name] = list(data_cfg["categorical_values"][feature_name])

    percentiles: dict[str, dict[object, float]] = {}
    for feature_name in feature_names:
        if data_cfg["reference_feature_types"][feature_name] != "ordered":
            continue
        if feature_name in continuous_feature_names:
            values = np.sort(balanced_df[feature_name].astype(int).to_numpy())
            percentiles[feature_name] = {
                state: float(np.searchsorted(values, state, side="right") / values.size)
                for state in original_ranges[feature_name]
            }
            continue

        value_counts = balanced_df[feature_name].astype(str).value_counts()
        running = 0.0
        total = float(max(1, int(balanced_df.shape[0])))
        feature_percentiles: dict[object, float] = {}
        for state in original_ranges[feature_name]:
            running += float(value_counts.get(state, 0))
            feature_percentiles[state] = running / total
        percentiles[feature_name] = feature_percentiles

    mads: dict[str, float] = {}
    for feature_name in continuous_feature_names:
        values = train_df[feature_name].astype(float).to_numpy()
        mad = float(np.median(np.abs(values - np.median(values))))
        if mad <= 0.0:
            mad = 1.0
        mads[feature_name] = mad

    cost_map = {
        feature_name: {
            state: index for index, state in enumerate(original_ranges[feature_name])
        }
        for feature_name in feature_names
    }

    return PaperMetricContext(
        feature_names=feature_names,
        continuous_feature_names=continuous_feature_names,
        categorical_feature_names=categorical_feature_names,
        continuous_feature_indexes=[
            feature_names.index(feature_name) for feature_name in continuous_feature_names
        ],
        categorical_feature_indexes=[
            feature_names.index(feature_name) for feature_name in categorical_feature_names
        ],
        feature_types=_ordered_mapping(feature_names, data_cfg["reference_feature_types"]),
        feature_change_restriction=_ordered_mapping(
            feature_names, data_cfg["reference_feature_change_restriction"]
        ),
        feature_values={
            feature_name: list(data_cfg["categorical_values"][feature_name])
            for feature_name in categorical_feature_names
        },
        original_ranges=original_ranges,
        percentiles=percentiles,
        mads=mads,
        cost_map=cost_map,
        invalid_cost=float(evaluation_cfg["invalid_cost"]),
        variance=float(evaluation_cfg["variance"]),
    )


def _sample_editable_features(
    context: PaperMetricContext,
    rng: np.random.Generator,
) -> set[str]:
    non_fixed_features = [
        feature_name
        for feature_name in context.feature_names
        if context.feature_types[feature_name] != "fixed"
    ]
    subset_size = int(rng.integers(1, len(non_fixed_features) + 1))
    sampled = rng.choice(
        np.array(non_fixed_features, dtype=object),
        size=subset_size,
        replace=False,
    )
    return {str(feature_name) for feature_name in sampled.tolist()}


def _sample_preference_scores(
    context: PaperMetricContext,
    editable_features: set[str],
    rng: np.random.Generator,
) -> dict[str, float]:
    concentration = np.array(
        [
            1.0 if feature_name in editable_features else EPSILON
            for feature_name in context.feature_names
        ],
        dtype="float64",
    )
    preference = rng.dirichlet(concentration)
    return {
        feature_name: (
            float(preference[index]) if feature_name in editable_features else 0.0
        )
        for index, feature_name in enumerate(context.feature_names)
    }


def _get_valid_ranges(
    query_row: pd.Series,
    editable_features: set[str],
    context: PaperMetricContext,
) -> dict[str, list[object]]:
    valid_ranges: dict[str, list[object]] = {}
    for feature_name in context.feature_names:
        restriction = int(context.feature_change_restriction[feature_name])
        current_value = query_row[feature_name]
        feature_range = list(context.original_ranges[feature_name])

        if feature_name in context.continuous_feature_names:
            current_value = int(current_value)
        else:
            current_value = str(current_value)

        if restriction == -2 or feature_name not in editable_features:
            valid_ranges[feature_name] = [current_value]
            continue

        if restriction == 0:
            valid_ranges[feature_name] = feature_range
            continue

        if feature_name not in context.continuous_feature_names and feature_name not in context.categorical_feature_names:
            raise ValueError(f"Unknown feature type for {feature_name}")

        if restriction == 1:
            if current_value not in feature_range:
                raise ValueError(f"Unknown current value for {feature_name}: {current_value}")
            valid_ranges[feature_name] = feature_range[feature_range.index(current_value) :]
            continue

        if restriction == -1:
            if current_value not in feature_range:
                raise ValueError(f"Unknown current value for {feature_name}: {current_value}")
            valid_ranges[feature_name] = feature_range[: feature_range.index(current_value) + 1]
            continue

        raise ValueError(f"Unsupported change restriction: {restriction}")

    return valid_ranges


def _linear_cost_means(
    feature_name: str,
    valid_states: list[object],
    current_value: object,
    preference_score: float,
    context: PaperMetricContext,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    full_range = list(context.original_ranges[feature_name])
    restriction = int(context.feature_change_restriction[feature_name])
    feature_type = context.feature_types[feature_name]

    means = np.full(len(full_range), context.invalid_cost, dtype="float64")
    variances = np.zeros(len(full_range), dtype="float64")
    current_index = full_range.index(current_value)

    if preference_score == 0.0 or len(valid_states) == 1 or feature_type == "fixed":
        flat_mean = np.array([0.0], dtype="float64")
    elif feature_type == "ordered":
        if restriction == 1:
            flat_mean = np.linspace(0.0, 1.0, len(valid_states), dtype="float64")
        elif restriction == -1:
            flat_mean = np.linspace(0.0, 1.0, len(valid_states), dtype="float64")[::-1]
        elif restriction == 0:
            current_valid_index = valid_states.index(current_value)
            post = np.linspace(
                0.0,
                1.0,
                len(valid_states[current_valid_index:]),
                dtype="float64",
            )
            pre = np.linspace(
                0.0,
                1.0,
                len(valid_states[: current_valid_index + 1]),
                dtype="float64",
            )[::-1]
            flat_mean = np.concatenate([pre[:-1], post])
        else:
            raise ValueError(f"Unsupported change restriction: {restriction}")
        flat_mean = flat_mean * (1.0 - preference_score)
    elif feature_type == "unordered":
        flat_mean = rng.uniform(0.0, 1.0, len(valid_states)).astype("float64")
        flat_mean[valid_states.index(current_value)] = 0.0
        flat_mean = flat_mean * (1.0 - preference_score)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    for index, state in enumerate(valid_states):
        full_index = full_range.index(state)
        means[full_index] = float(flat_mean[index])
        variances[full_index] = context.variance

    means[current_index] = 0.0
    variances[current_index] = 0.0
    return means, variances


def _percentile_cost_means(
    feature_name: str,
    valid_states: list[object],
    current_value: object,
    preference_score: float,
    context: PaperMetricContext,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    full_range = list(context.original_ranges[feature_name])
    restriction = int(context.feature_change_restriction[feature_name])
    feature_type = context.feature_types[feature_name]

    means = np.full(len(full_range), context.invalid_cost, dtype="float64")
    variances = np.zeros(len(full_range), dtype="float64")
    current_index = full_range.index(current_value)

    if preference_score == 0.0 or len(valid_states) == 1 or feature_type == "fixed":
        flat_mean = np.array([0.0], dtype="float64")
    elif feature_type == "ordered":
        feature_percentiles = context.percentiles[feature_name]
        current_percentile = feature_percentiles[current_value]
        flat_mean = np.array(
            [
                abs(feature_percentiles[state] - current_percentile)
                for state in valid_states
            ],
            dtype="float64",
        )
        flat_mean = flat_mean * (1.0 - preference_score)
    elif feature_type == "unordered":
        flat_mean = rng.uniform(0.0, 1.0, len(valid_states)).astype("float64")
        flat_mean[valid_states.index(current_value)] = 0.0
        flat_mean = flat_mean * (1.0 - preference_score)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    for index, state in enumerate(valid_states):
        full_index = full_range.index(state)
        means[full_index] = float(flat_mean[index])
        variances[full_index] = context.variance

    means[current_index] = 0.0
    variances[current_index] = 0.0
    return means, variances


def _combine_cost_means(
    linear_means: np.ndarray,
    linear_vars: np.ndarray,
    percentile_means: np.ndarray,
    percentile_vars: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if alpha == 0.0:
        return linear_means, linear_vars
    if alpha == 1.0:
        return percentile_means, percentile_vars
    means = linear_means * alpha + percentile_means * (1.0 - alpha)
    return means, percentile_vars


def _sample_cost_vector(
    means: np.ndarray,
    variances: np.ndarray,
    invalid_cost: float,
    rng: np.random.Generator,
) -> np.ndarray:
    samples = np.full(means.shape, invalid_cost, dtype="float64")
    zero_mask = means == 0.0
    positive_mask = (means > 0.0) & (means <= 1.0)
    if positive_mask.any():
        mean_values = means[positive_mask] + EPSILON
        mean_values = np.clip(mean_values, EPSILON, 1.0 - EPSILON)
        variance_values = variances[positive_mask] + EPSILON
        alpha_values = (
            ((1.0 - mean_values) / variance_values) - (1.0 / mean_values)
        ) * np.square(mean_values)
        alpha_values = np.maximum(alpha_values, EPSILON)
        beta_values = alpha_values * ((1.0 / mean_values) - 1.0)
        beta_values = np.maximum(beta_values, EPSILON)
        samples[positive_mask] = rng.beta(alpha_values, beta_values)
    samples[zero_mask] = 0.0
    return samples


def _sample_user_cost(
    query_row: pd.Series,
    context: PaperMetricContext,
    alpha: float | None,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    editable_features = _sample_editable_features(context, rng)
    preference_scores = _sample_preference_scores(context, editable_features, rng)
    valid_ranges = _get_valid_ranges(query_row, editable_features, context)
    sample_alpha = float(alpha) if alpha is not None else float(np.round(rng.uniform(0.0, 1.0), 2))

    costs: dict[str, np.ndarray] = {}
    for feature_name in context.feature_names:
        current_value = query_row[feature_name]
        if feature_name in context.continuous_feature_names:
            current_value = int(current_value)
        else:
            current_value = str(current_value)

        valid_states = valid_ranges[feature_name]
        linear_means, linear_vars = _linear_cost_means(
            feature_name,
            valid_states,
            current_value,
            preference_scores[feature_name],
            context,
            rng,
        )
        percentile_means, percentile_vars = _percentile_cost_means(
            feature_name,
            valid_states,
            current_value,
            preference_scores[feature_name],
            context,
            rng,
        )
        means, variances = _combine_cost_means(
            linear_means,
            linear_vars,
            percentile_means,
            percentile_vars,
            sample_alpha,
        )
        costs[feature_name] = _sample_cost_vector(
            means,
            variances,
            context.invalid_cost,
            rng,
        )
    return costs


def _compute_cf_cost(
    cf_values: np.ndarray,
    user_cost: dict[str, np.ndarray],
    context: PaperMetricContext,
) -> float:
    total_cost = 0.0
    for feature_index, feature_name in enumerate(context.feature_names):
        value = cf_values[feature_index]
        if feature_name in context.continuous_feature_names:
            key = int(float(value))
        else:
            key = str(value)
        total_cost += float(user_cost[feature_name][context.cost_map[feature_name][key]])
    return total_cost


def _pairwise_distance(
    cfs: np.ndarray,
    query: np.ndarray,
    context: PaperMetricContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfs_array = np.asarray(cfs)
    query_array = np.asarray(query)
    cat_idx = context.categorical_feature_indexes
    cont_idx = context.continuous_feature_indexes

    cat_count_dists = (
        cfs_array[..., cat_idx] != query_array[..., cat_idx]
    ).astype(int).mean(axis=-1)

    cfs_cont = cfs_array[..., cont_idx].astype(float).astype(int)
    query_cont = query_array[..., cont_idx].astype(float).astype(int)
    cont_count_dists = (cfs_cont != query_cont).astype(int).mean(axis=-1)

    cont_mads = np.array(
        [context.mads[feature_name] for feature_name in context.continuous_feature_names],
        dtype="float64",
    )
    cont_ranges = np.array(
        [
            max(
                1.0,
                float(
                    context.original_ranges[feature_name][-1]
                    - context.original_ranges[feature_name][0]
                ),
            )
            for feature_name in context.continuous_feature_names
        ],
        dtype="float64",
    )

    cont_dists_mad = (np.abs(cfs_cont - query_cont) / cont_mads).mean(axis=-1)
    cont_dists_norm = (np.abs(cfs_cont - query_cont) / cont_ranges).mean(axis=-1)
    return cat_count_dists, cont_dists_mad, cont_count_dists, cont_dists_norm


def _merge_cat_cont(cat_value: float, cont_value: float, context: PaperMetricContext) -> float:
    num_cat = len(context.categorical_feature_names)
    num_cont = len(context.continuous_feature_names)
    weights = np.array([num_cat, num_cont], dtype="float64")
    weights = weights / weights.sum()
    return float(np.dot(weights, np.array([cat_value, cont_value], dtype="float64")))


def _validity_and_unique_valid_sets(
    cfs: np.ndarray,
    original_class: int,
    pred_classes: np.ndarray,
) -> tuple[float, np.ndarray]:
    if len(cfs) == 0:
        return 0.0, np.empty((0, cfs.shape[1] if cfs.ndim == 2 else 0), dtype=str)

    target_class = 1 - int(original_class)
    cfs_df = pd.DataFrame(cfs.astype(str, copy=False))
    unique_cfs_df = cfs_df.drop_duplicates(keep="first")
    unique_idx = unique_cfs_df.index.to_numpy(dtype=int)
    valid_unique_mask = pred_classes[unique_idx].astype(int) == target_class
    validity = float(valid_unique_mask.sum() / len(cfs_df))
    return validity, unique_cfs_df.loc[valid_unique_mask].to_numpy(dtype=str)


def _compute_set_metrics(
    query_values: np.ndarray,
    cf_values: np.ndarray,
    pred_classes: np.ndarray,
    original_class: int,
    context: PaperMetricContext,
) -> dict[str, float | bool]:
    validity, unique_valid_cfs = _validity_and_unique_valid_sets(
        cf_values,
        original_class,
        pred_classes,
    )
    if len(unique_valid_cfs) == 0:
        return {
            "Val": validity,
            "Prox": context.invalid_cost,
            "Div": 0.0,
            "Spars": 0.0,
            "include": False,
        }

    cat_dists, _cont_mad, cont_count_dists, cont_norm_dists = _pairwise_distance(
        unique_valid_cfs,
        query_values,
        context,
    )
    cat_proximity = float(1.0 - cat_dists.mean())
    cont_proximity = float(1.0 - cont_norm_dists.mean())
    proximity = _merge_cat_cont(cat_proximity, cont_proximity, context)

    if len(unique_valid_cfs) <= 1:
        diversity = 0.0
    else:
        pairwise_cat = []
        pairwise_cont_norm = []
        for first_index in range(len(unique_valid_cfs)):
            for second_index in range(first_index + 1, len(unique_valid_cfs)):
                (
                    pair_cat,
                    _pair_cont_mad,
                    _pair_cont_count,
                    pair_cont_norm,
                ) = _pairwise_distance(
                    unique_valid_cfs[first_index],
                    unique_valid_cfs[second_index],
                    context,
                )
                pairwise_cat.append(float(np.asarray(pair_cat).reshape(-1)[0]))
                pairwise_cont_norm.append(
                    float(np.asarray(pair_cont_norm).reshape(-1)[0])
                )
        diversity = _merge_cat_cont(
            float(np.mean(pairwise_cat)),
            float(np.mean(pairwise_cont_norm)),
            context,
        )

    num_cat = len(context.categorical_feature_names)
    num_cont = len(context.continuous_feature_names)
    sample_sparsity = (
        cont_count_dists * num_cont + cat_dists * num_cat
    ) / max(1, num_cat + num_cont)
    sparsity = float(1.0 - sample_sparsity.mean())

    return {
        "Val": validity,
        "Prox": proximity,
        "Div": diversity,
        "Spars": sparsity,
        "include": True,
    }


def _compute_cost_metrics(
    cf_values: np.ndarray,
    pred_classes: np.ndarray,
    original_class: int,
    user_cost: dict[str, np.ndarray],
    context: PaperMetricContext,
    cost_threshold: float,
    coverage_threshold: float,
) -> dict[str, float | bool]:
    target_class = 1 - int(original_class)
    validity_mask = pred_classes.astype(int) == target_class

    if len(cf_values) == 0 or not bool(validity_mask.any()):
        final_cost = float(context.invalid_cost)
    else:
        valid_costs = [
            _compute_cf_cost(cf_values[index], user_cost, context)
            for index in np.flatnonzero(validity_mask)
        ]
        final_cost = float(min(valid_costs)) if valid_costs else float(context.invalid_cost)

    covered = bool(final_cost < coverage_threshold)
    return {
        "PAC": final_cost,
        "Cov": float(covered),
        "FS@1": float(final_cost <= cost_threshold),
        "covered": covered,
    }


def _scale_metric_for_report(metric_name: str, value: float) -> float:
    if metric_name == "PAC":
        return float(value)
    return float(value * 100.0)


def _evaluate_method(
    method_name: str,
    method_cfg: dict[str, Any],
    model: ReferenceCompasMlpModel,
    trainset,
    factual_encoded: pd.DataFrame,
    factual_raw: pd.DataFrame,
    user_costs: list[dict[str, np.ndarray]],
    factual_predictions: np.ndarray,
    metric_context: PaperMetricContext,
    data_cfg: dict[str, Any],
    scaling_stats: ScalingStats,
    device: str,
) -> tuple[dict[str, float], dict[str, float]]:
    cols_method = ColsMethod(
        target_model=model,
        seed=int(method_cfg["seed"]),
        device=device,
        desired_class=method_cfg.get("desired_class"),
        num_cfs=int(method_cfg["num_cfs"]),
        num_mcmc=int(method_cfg["num_mcmc"]),
        budget=int(method_cfg["budget"]),
        num_parallel_runs=int(method_cfg["num_parallel_runs"]),
        hamming_dist=int(method_cfg["hamming_dist"]),
        perturb_type=str(method_cfg["perturb_type"]),
        init_type=str(method_cfg["init_type"]),
        iter_type=str(method_cfg["iter_type"]),
        alpha=method_cfg.get("alpha"),
        variance=float(method_cfg["variance"]),
        invalid_cost=float(method_cfg["invalid_cost"]),
    )
    cols_method.fit(trainset)

    cf_sets_encoded = cols_method.get_counterfactual_sets(factual_encoded)
    cf_validity_masks = [mask.to_numpy(copy=True) for mask in cols_method._last_counterfactual_validity]
    search_stats = list(cols_method._last_search_stats)

    cost_rows = []
    set_rows = []
    for factual_index, cf_encoded in enumerate(cf_sets_encoded):
        cf_pred_classes = _predict_label_indices(model, cf_encoded)
        validity_mask = cf_validity_masks[factual_index].astype(bool)
        target_index = int(method_cfg["desired_class"])
        if not np.array_equal(validity_mask, cf_pred_classes == target_index):
            raise AssertionError("Stored COLS validity mask does not match model predictions")

        cf_raw = _decode_counterfactual_set(cf_encoded, cols_method, data_cfg, scaling_stats)
        cf_values = cf_raw.loc[:, metric_context.feature_names].astype(str).to_numpy()
        query_values = (
            factual_raw.iloc[factual_index]
            .loc[metric_context.feature_names]
            .astype(str)
            .to_numpy()
        )
        original_class = int(factual_predictions[factual_index])

        cost_rows.append(
            _compute_cost_metrics(
                cf_values=cf_values,
                pred_classes=cf_pred_classes,
                original_class=original_class,
                user_cost=user_costs[factual_index],
                context=metric_context,
                cost_threshold=float(data_cfg["evaluation"]["cost_threshold"]),
                coverage_threshold=float(data_cfg["evaluation"]["coverage_threshold"]),
            )
        )
        set_rows.append(
            _compute_set_metrics(
                query_values=query_values,
                cf_values=cf_values,
                pred_classes=cf_pred_classes,
                original_class=original_class,
                context=metric_context,
            )
        )

    covered_mask = [bool(row["covered"]) for row in cost_rows]
    pac_values = [float(row["PAC"]) for row in cost_rows if bool(row["covered"])]
    included_set_rows = [row for row in set_rows if bool(row["include"])]

    aggregated = {
        "FS@1": float(np.mean([row["FS@1"] for row in cost_rows])),
        "PAC": float(np.mean(pac_values)) if pac_values else float("nan"),
        "Cov": float(np.mean([row["Cov"] for row in cost_rows])),
        "Div": (
            float(np.mean([float(row["Div"]) for row in included_set_rows]))
            if included_set_rows
            else float("nan")
        ),
        "Prox": (
            float(np.mean([float(row["Prox"]) for row in included_set_rows]))
            if included_set_rows
            else float("nan")
        ),
        "Spars": (
            float(np.mean([float(row["Spars"]) for row in included_set_rows]))
            if included_set_rows
            else float("nan")
        ),
        "Val": (
            float(np.mean([float(row["Val"]) for row in included_set_rows]))
            if included_set_rows
            else float("nan")
        ),
    }
    aggregated = {
        metric_name: _scale_metric_for_report(metric_name, metric_value)
        for metric_name, metric_value in aggregated.items()
    }

    search_summary = {
        "mean_queries": float(
            np.mean([float(stats["num_queries"]) for stats in search_stats])
        ),
        "mean_emc": float(np.mean([float(stats["emc"]) for stats in search_stats])),
        "mean_valid_cfs": float(
            np.mean([float(stats["num_valid"]) for stats in search_stats])
        ),
        "coverage_fraction": float(np.mean(covered_mask)),
    }
    return aggregated, search_summary


def _run_single_seed(
    run_seed: int,
    config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    data_cfg = config["data"]
    model_cfg = config["model"]

    raw_df = _load_raw_compas_dataframe(data_cfg)
    split_artifacts = _build_reference_split(raw_df, data_cfg)
    scaling_stats = _compute_scaling_stats(
        split_artifacts.train_df,
        data_cfg["continuous_feature_order"],
    )

    template = _build_dataset_template(data_cfg)
    train_features = _encode_raw_features(split_artifacts.train_df, data_cfg, scaling_stats)
    val_features = _encode_raw_features(split_artifacts.val_df, data_cfg, scaling_stats)
    provisional_features = _encode_raw_features(
        split_artifacts.provisional_factual_df,
        data_cfg,
        scaling_stats,
    )

    trainset = _build_frozen_dataset(
        template,
        train_features,
        split_artifacts.train_df[data_cfg["target_column"]],
        data_cfg,
        "trainset",
    )
    valset = _build_frozen_dataset(
        template,
        val_features,
        split_artifacts.val_df[data_cfg["target_column"]],
        data_cfg,
        "valset",
    )

    model = ReferenceCompasMlpModel(
        seed=int(model_cfg.get("seed", 1234)),
        device=device,
        epochs=int(model_cfg["epochs"]),
        learning_rate=float(model_cfg["learning_rate"]),
        batch_size=int(model_cfg["batch_size"]),
        layers=[int(width) for width in model_cfg["hidden_layers"]],
        optimizer=str(model_cfg["optimizer"]),
        criterion=str(model_cfg["criterion"]),
        output_activation=str(model_cfg["output_activation"]),
        reference_checkpoint_path=(
            str((PROJECT_ROOT / model_cfg["reference_checkpoint_path"]).resolve())
            if bool(model_cfg.get("use_reference_checkpoint", False))
            else None
        ),
        save_name=None,
    )
    model.fit(trainset, valset=valset)

    provisional_predictions = _predict_label_indices(model, provisional_features)
    undesired_class = int(data_cfg["undesired_class"])
    keep_mask = provisional_predictions == undesired_class

    factual_raw = split_artifacts.provisional_factual_df.loc[keep_mask].reset_index(drop=True)
    factual_encoded = provisional_features.loc[keep_mask].reset_index(drop=True)

    target_factual_count = int(data_cfg["split"]["target_factual_count"])
    if factual_raw.shape[0] > target_factual_count:
        factual_raw = factual_raw.iloc[:target_factual_count].reset_index(drop=True)
        factual_encoded = factual_encoded.iloc[:target_factual_count].reset_index(drop=True)

    if factual_raw.empty:
        raise RuntimeError("Model filtering produced no COMPAS factuals for recourse")

    metric_context = _build_metric_context(
        split_artifacts.balanced_df.loc[:, list(data_cfg["raw_feature_order"]) + [data_cfg["target_column"]]],
        split_artifacts.train_df.loc[:, list(data_cfg["raw_feature_order"]) + [data_cfg["target_column"]]],
        data_cfg,
        data_cfg["evaluation"],
    )

    evaluation_rng = np.random.default_rng(run_seed)
    user_costs = [
        _sample_user_cost(
            factual_raw.iloc[row_index].loc[metric_context.feature_names],
            metric_context,
            data_cfg["evaluation"].get("alpha"),
            evaluation_rng,
        )
        for row_index in range(factual_raw.shape[0])
    ]
    factual_predictions = _predict_label_indices(model, factual_encoded)

    method_results: dict[str, dict[str, Any]] = {}
    for base_method_cfg in config["methods"]:
        method_cfg = copy.deepcopy(base_method_cfg)
        method_cfg["seed"] = run_seed
        metrics, search_summary = _evaluate_method(
            method_name=str(method_cfg["name"]),
            method_cfg=method_cfg,
            model=model,
            trainset=trainset,
            factual_encoded=factual_encoded,
            factual_raw=factual_raw,
            user_costs=user_costs,
            factual_predictions=factual_predictions,
            metric_context=metric_context,
            data_cfg=data_cfg,
            scaling_stats=scaling_stats,
            device=device,
        )
        method_results[str(method_cfg["name"])] = {
            "metrics": metrics,
            "search": search_summary,
        }

    return {
        "seed": run_seed,
        "balanced_rows": int(split_artifacts.balanced_df.shape[0]),
        "train_rows": int(split_artifacts.train_df.shape[0]),
        "val_rows": int(split_artifacts.val_df.shape[0]),
        "provisional_rows": int(split_artifacts.provisional_factual_df.shape[0]),
        "provisional_unique_rows": int(split_artifacts.provisional_unique_count),
        "factual_rows": int(factual_raw.shape[0]),
        "val_accuracy": (
            float(model._best_val_accuracy)
            if model._best_val_accuracy is not None
            else float("nan")
        ),
        "methods": method_results,
    }


def _build_comparison_table(
    run_results: list[dict[str, Any]],
    paper_targets: dict[str, dict[str, float]],
) -> pd.DataFrame:
    records = []
    for method_name, metric_targets in paper_targets.items():
        for metric_name, paper_value in metric_targets.items():
            metric_values = [
                float(result["methods"][method_name]["metrics"][metric_name])
                for result in run_results
            ]
            reproduced_mean = float(np.nanmean(metric_values))
            reproduced_std = float(np.nanstd(metric_values, ddof=0))
            records.append(
                {
                    "method": method_name,
                    "metric": metric_name,
                    "paper": float(paper_value),
                    "reproduced_mean": reproduced_mean,
                    "reproduced_std": reproduced_std,
                    "absolute_gap": float(abs(reproduced_mean - float(paper_value))),
                }
            )
    return pd.DataFrame.from_records(records)


def _print_run_summary(run_result: dict[str, Any]) -> None:
    print(
        "seed={seed} balanced={balanced_rows} train={train_rows} val={val_rows} "
        "provisional={provisional_rows} provisional_unique={provisional_unique_rows} "
        "factuals={factual_rows} val_acc={val_accuracy:.4f}".format(**run_result)
    )
    for method_name, method_result in run_result["methods"].items():
        search = method_result["search"]
        print(
            "  {name}: mean_queries={queries:.2f} mean_emc={emc:.4f} "
            "mean_valid_cfs={valid:.2f} coverage={coverage:.4f}".format(
                name=method_name,
                queries=search["mean_queries"],
                emc=search["mean_emc"],
                valid=search["mean_valid_cfs"],
                coverage=search["coverage_fraction"],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("reproduce_configs.yml")),
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-factuals", type=int, default=None)
    parser.add_argument("--override-epochs", type=int, default=None)
    parser.add_argument("--override-budget", type=int, default=None)
    parser.add_argument("--override-num-mcmc", type=int, default=None)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--use-reference-checkpoint", action="store_true")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    if not config_path.exists():
        config_path = Path(args.config).resolve()
    config = _apply_cli_overrides(_load_config(config_path), args)
    device = _resolve_device(str(config["model"]["device"]).lower())

    run_results = []
    for run_seed in [int(seed) for seed in config["reproduction"]["run_seeds"]]:
        run_result = _run_single_seed(run_seed, config, device)
        run_results.append(run_result)
        _print_run_summary(run_result)

    comparison = _build_comparison_table(
        run_results,
        config["reproduction"]["paper_targets"],
    )
    print("\nComparison Table")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
