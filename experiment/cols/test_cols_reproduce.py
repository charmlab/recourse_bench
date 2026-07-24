from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.model_selection import ShuffleSplit
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.adult_carla.adult_carla import AdultCarlaDataset
from dataset.compas_carla.compas_carla import CompasCarlaDataset
from experiment.utils import write_reproduction_report
from method.cols.cols import ColsMethod
from method.cols.support import (
    RuntimeSearchContext,
    build_runtime_context,
    compute_candidate_cost_matrix,
    compute_emc,
    decode_feature_dataframe,
    encode_state_dataframe,
    resolve_target_index,
)
from model.mlp.mlp import MlpModel
from model.model_object import process_nan
from model.model_utils import build_optimizer
from utils.seed import seed_context


DEFAULT_CONFIG_PATH = Path(__file__).with_name("reproduce_configs.yaml")
REPORT_PATH = Path(__file__).with_name("reproduction_report.json")
EPSILON = 1e-7


def _resolve_progress_mode(progress: str) -> str:
    resolved = str(progress).lower()
    if resolved not in {"none", "standard", "all"}:
        raise ValueError("progress must be one of: none, standard, all")
    return resolved


def _progress_enabled(progress_mode: str) -> bool:
    return progress_mode != "none"


def _search_progress_enabled(progress_mode: str) -> bool:
    return progress_mode == "all"


def _build_progress_bar(*args, **kwargs):
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(*args, **kwargs)


def _maybe_write_heartbeat(
    *,
    enabled: bool,
    heartbeat_seconds: int,
    last_heartbeat: float,
    message: str,
) -> float:
    if (not enabled) or heartbeat_seconds <= 0:
        return last_heartbeat
    now = time.monotonic()
    if now - last_heartbeat >= heartbeat_seconds:
        tqdm.write(message)
        return now
    return last_heartbeat


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


@dataclass(frozen=True)
class PerFactualResult:
    subgroup_values: dict[str, str]
    final_cost: float
    fs_at_1: float
    cov: float


@dataclass(frozen=True)
class TableRunResult:
    seed: int
    dataset: str
    methods: dict[str, dict[str, Any]]
    val_accuracy: float
    factual_rows: int


@dataclass(frozen=True)
class SearchRunArtifacts:
    encoded_set: pd.DataFrame
    pred_classes: np.ndarray
    valid_mask: np.ndarray
    score: float
    num_queries: int


class ReferenceBinaryMlpModel(MlpModel):
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
        show_progress: bool = False,
        progress_desc: str = "model-fit",
        progress_position: int = 0,
        progress_leave: bool = False,
        heartbeat_seconds: int = 60,
    ):
        if trainset is None:
            raise ValueError("trainset is required")
        if valset is None:
            raise ValueError("valset is required")

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
            last_heartbeat = time.monotonic()

            epoch_iterator = _build_progress_bar(
                range(self._epochs),
                total=self._epochs,
                desc=progress_desc,
                position=progress_position,
                leave=progress_leave,
                disable=not show_progress,
            )
            for epoch_index in epoch_iterator:
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

                if show_progress:
                    epoch_iterator.set_postfix(
                        val_acc=f"{val_accuracy:.4f}",
                        best_acc=f"{best_val_accuracy:.4f}",
                        val_loss=f"{val_loss:.4f}",
                    )
                last_heartbeat = _maybe_write_heartbeat(
                    enabled=show_progress,
                    heartbeat_seconds=heartbeat_seconds,
                    last_heartbeat=last_heartbeat,
                    message=(
                        f"[train] epoch={epoch_index + 1}/{self._epochs} "
                        f"best_val_acc={best_val_accuracy:.4f} "
                        f"best_val_loss={best_val_loss:.4f}"
                    ),
                )
            epoch_iterator.close()

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


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _apply_profile(config: dict[str, Any], profile_name: str | None) -> dict[str, Any]:
    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        return copy.deepcopy(config)

    resolved_profile = profile_name or config.get("active_profile")
    if resolved_profile is None:
        return copy.deepcopy(config)
    if resolved_profile not in profiles:
        raise ValueError(f"Unknown reproduction profile: {resolved_profile}")

    cfg = copy.deepcopy(config)
    profile = profiles[resolved_profile]
    for section_name in ("reproduction", "datasets"):
        section_overrides = profile.get(section_name)
        if isinstance(section_overrides, dict):
            cfg[section_name] = _deep_merge(cfg[section_name], section_overrides)
    return cfg


def _resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name.lower()


def _ordered_mapping(names: list[str], values: dict[str, Any]) -> dict[str, Any]:
    return {name: values[name] for name in names}


def _bucket_cap_gain_like(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    positive = numeric[numeric > 0]
    if positive.empty:
        return pd.Series(["0"] * len(values), index=values.index, dtype="object")
    bins = np.digitize(
        numeric.to_numpy(dtype="float64"),
        [0, float(np.median(positive)), float("inf")],
        right=True,
    )
    return pd.Series(bins.astype(str), index=values.index, dtype="object")


def _load_raw_dataframe(data_cfg: dict[str, Any]) -> pd.DataFrame:
    raw_path = (PROJECT_ROOT / data_cfg["raw_path"]).resolve()
    df = pd.read_csv(raw_path).dropna().reset_index(drop=True)

    variant = str(data_cfg.get("variant", data_cfg["dataset_name"])).lower()
    if variant == "adult_binary":
        if "fnlwgt" in df.columns:
            df = df.drop(columns=["fnlwgt"])
        df["capital-gain"] = _bucket_cap_gain_like(df["capital-gain"])
        df["capital-loss"] = _bucket_cap_gain_like(df["capital-loss"])

    ordered_columns = list(data_cfg["raw_feature_order"]) + [data_cfg["target_column"]]
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

    return SplitArtifacts(
        balanced_df=balanced_df,
        train_df=balanced_df.iloc[train_positions].reset_index(drop=True),
        val_df=balanced_df.iloc[val_positions].reset_index(drop=True),
        provisional_factual_df=balanced_df.iloc[provisional_positions].reset_index(drop=True),
        provisional_unique_count=len(provisional_unique_positions),
    )


def _compute_scaling_stats(
    train_df: pd.DataFrame,
    continuous_feature_names: list[str],
) -> ScalingStats:
    return ScalingStats(
        minimum={
            feature_name: float(train_df[feature_name].min())
            for feature_name in continuous_feature_names
        },
        maximum={
            feature_name: float(train_df[feature_name].max())
            for feature_name in continuous_feature_names
        },
    )


def _normalize_feature(values: pd.Series, minimum: float, maximum: float) -> pd.Series:
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
        else:
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
        raw_values = decoded[feature_name].astype("float64") * (maximum - minimum) + minimum
        decoded[feature_name] = np.rint(raw_values).astype(int)

    for feature_name in data_cfg["categorical_feature_order"]:
        if _uses_scalar_categorical_encoding(data_cfg):
            categories = list(data_cfg["categorical_values"][feature_name])
            decoded[feature_name] = decoded[feature_name].map(
                lambda value: categories[max(0, min(len(categories) - 1, int(round(float(value)))))]
            )
        decoded[feature_name] = decoded[feature_name].astype(str)

    return decoded


def _build_dataset_template(data_cfg: dict[str, Any]):
    source_dataset_name = str(data_cfg.get("source_dataset", data_cfg["dataset_name"]))
    if source_dataset_name == "adult_carla":
        template = AdultCarlaDataset(
            path=str((PROJECT_ROOT / "dataset/adult_carla").resolve())
        )
    elif source_dataset_name == "compas_carla":
        template = CompasCarlaDataset(
            path=str((PROJECT_ROOT / "dataset/compas_carla").resolve())
        )
    else:
        raise ValueError(f"Unsupported source_dataset: {source_dataset_name}")

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
    template.update(
        "feature_order",
        list(data_cfg["encoded_feature_order"]) + [data_cfg["target_column"]],
    )
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
    template,
    feature_df: pd.DataFrame,
    target: pd.Series,
    data_cfg: dict[str, Any],
    marker: str,
    extra_attrs: dict[str, object] | None = None,
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
    if extra_attrs is not None:
        for attr_name, attr_value in extra_attrs.items():
            dataset.update(attr_name, attr_value)
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

            def _percentile_rank(score: int) -> float:
                left = int(np.searchsorted(values, int(score), side="left"))
                right = int(np.searchsorted(values, int(score), side="right"))
                if right > left:
                    return float(((left + right + 1) / 2.0) / values.size)
                return float(right / values.size)

            percentiles[feature_name] = {
                state: _percentile_rank(state)
                for state in original_ranges[feature_name]
            }
        else:
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
        mads[feature_name] = mad if mad > 0.0 else 1.0

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


def _build_cols_state_space_overrides(
    balanced_df: pd.DataFrame,
    scaling_stats: ScalingStats,
    data_cfg: dict[str, Any],
) -> dict[str, list[float]]:
    overrides: dict[str, list[float]] = {}
    for feature_name in data_cfg["continuous_feature_order"]:
        minimum = scaling_stats.minimum[feature_name]
        maximum = scaling_stats.maximum[feature_name]
        raw_min = int(balanced_df[feature_name].min())
        raw_max = int(balanced_df[feature_name].max())
        if maximum == minimum:
            overrides[feature_name] = [0.0]
            continue
        overrides[feature_name] = [
            float((raw_value - minimum) / (maximum - minimum))
            for raw_value in range(raw_min, raw_max + 1)
        ]
    return overrides


def _sample_editable_features(
    context: PaperMetricContext,
    rng: np.random.Generator,
) -> set[str]:
    editable_candidates = [
        feature_name
        for feature_name in context.feature_names
        if int(context.feature_change_restriction[feature_name]) != -2
    ]
    num_features = int(rng.integers(1, len(editable_candidates) + 1))
    chosen = rng.choice(
        np.array(editable_candidates, dtype=object),
        size=num_features,
        replace=False,
    )
    return {str(feature_name) for feature_name in chosen.tolist()}


def _sample_preference_scores(
    context: PaperMetricContext,
    editable_features: set[str],
    rng: np.random.Generator,
) -> dict[str, float]:
    if not editable_features:
        return {feature_name: 0.0 for feature_name in context.feature_names}
    concentration = np.array(
        [
            1.0 if feature_name in editable_features else EPSILON
            for feature_name in context.feature_names
        ],
        dtype="float64",
    )
    preference_vector = rng.dirichlet(concentration)
    return {
        feature_name: (
            float(preference_vector[index])
            if feature_name in editable_features
            else 0.0
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
        current_value = query_row[feature_name]
        if feature_name in context.continuous_feature_names:
            current_value = int(current_value)
        else:
            current_value = str(current_value)

        states = list(context.original_ranges[feature_name])
        restriction = int(context.feature_change_restriction[feature_name])
        if restriction == -2 or feature_name not in editable_features:
            valid_ranges[feature_name] = [current_value]
        elif restriction == 0:
            valid_ranges[feature_name] = states
        elif restriction == 1:
            valid_ranges[feature_name] = states[states.index(current_value) :]
        elif restriction == -1:
            valid_ranges[feature_name] = states[: states.index(current_value) + 1]
        else:
            raise ValueError(f"Unsupported feature restriction: {restriction}")
    return valid_ranges


def _ordered_linear_cost(
    states: list[object],
    current_value: object,
    restriction: int,
) -> np.ndarray:
    current_index = states.index(current_value)
    flat_mean = np.full(len(states), np.inf, dtype="float64")
    flat_mean[current_index] = 0.0

    if restriction in {0, 1}:
        after = states[current_index:]
        if len(after) > 1:
            flat_mean[current_index:] = np.linspace(0.0, 1.0, len(after))
    if restriction in {0, -1}:
        before = states[: current_index + 1]
        if len(before) > 1:
            flat_mean[: current_index + 1] = np.linspace(0.0, 1.0, len(before))[::-1]
    return flat_mean


def _ordered_percentile_cost(
    feature_name: str,
    states: list[object],
    current_value: object,
    restriction: int,
    context: PaperMetricContext,
) -> np.ndarray:
    flat_mean = np.full(len(states), np.inf, dtype="float64")
    current_percentile = context.percentiles[feature_name][current_value]
    for index, state in enumerate(states):
        if restriction == 1 and index < states.index(current_value):
            continue
        if restriction == -1 and index > states.index(current_value):
            continue
        flat_mean[index] = abs(
            context.percentiles[feature_name][state] - current_percentile
        )
    flat_mean[states.index(current_value)] = 0.0
    return flat_mean


def _unordered_cost(
    states: list[object],
    current_value: object,
    rng: np.random.Generator,
) -> np.ndarray:
    flat_mean = rng.uniform(0.0, 1.0, len(states)).astype("float64")
    flat_mean[states.index(current_value)] = 0.0
    return flat_mean


def _linear_cost_means(
    feature_name: str,
    valid_states: list[object],
    current_value: object,
    preference_score: float,
    context: PaperMetricContext,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    restriction = int(context.feature_change_restriction[feature_name])
    feature_type = str(context.feature_types[feature_name])

    if preference_score == 0.0 or len(valid_states) == 1 or feature_type == "fixed":
        means = np.full(len(context.original_ranges[feature_name]), context.invalid_cost, dtype="float64")
        variances = np.zeros_like(means)
        means[context.cost_map[feature_name][current_value]] = 0.0
        return means, variances

    if feature_type == "ordered":
        flat_mean = _ordered_linear_cost(valid_states, current_value, restriction)
    else:
        flat_mean = _unordered_cost(valid_states, current_value, rng)
    flat_mean = flat_mean * (1.0 - preference_score)

    means = np.full(len(context.original_ranges[feature_name]), context.invalid_cost, dtype="float64")
    variances = np.zeros_like(means)
    for index, state in enumerate(valid_states):
        means[context.cost_map[feature_name][state]] = flat_mean[index]
        if flat_mean[index] > 0.0 and np.isfinite(flat_mean[index]):
            variances[context.cost_map[feature_name][state]] = context.variance
    means[context.cost_map[feature_name][current_value]] = 0.0
    variances[context.cost_map[feature_name][current_value]] = 0.0
    return np.round(means, 4), np.round(variances, 4)


def _percentile_cost_means(
    feature_name: str,
    valid_states: list[object],
    current_value: object,
    preference_score: float,
    context: PaperMetricContext,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    restriction = int(context.feature_change_restriction[feature_name])
    feature_type = str(context.feature_types[feature_name])

    if preference_score == 0.0 or len(valid_states) == 1 or feature_type == "fixed":
        means = np.full(len(context.original_ranges[feature_name]), context.invalid_cost, dtype="float64")
        variances = np.zeros_like(means)
        means[context.cost_map[feature_name][current_value]] = 0.0
        return means, variances

    if feature_type == "ordered":
        flat_mean = _ordered_percentile_cost(
            feature_name,
            valid_states,
            current_value,
            restriction,
            context,
        )
    else:
        flat_mean = _unordered_cost(valid_states, current_value, rng)
    flat_mean = flat_mean * (1.0 - preference_score)

    means = np.full(len(context.original_ranges[feature_name]), context.invalid_cost, dtype="float64")
    variances = np.zeros_like(means)
    for index, state in enumerate(valid_states):
        means[context.cost_map[feature_name][state]] = flat_mean[index]
        if flat_mean[index] > 0.0 and np.isfinite(flat_mean[index]):
            variances[context.cost_map[feature_name][state]] = context.variance
    means[context.cost_map[feature_name][current_value]] = 0.0
    variances[context.cost_map[feature_name][current_value]] = 0.0
    return np.round(means, 4), np.round(variances, 4)


def _combine_cost_means(
    linear_means: np.ndarray,
    linear_vars: np.ndarray,
    percentile_means: np.ndarray,
    percentile_vars: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    means = linear_means * alpha + percentile_means * (1.0 - alpha)
    variances = linear_vars * alpha + percentile_vars * (1.0 - alpha)
    invalid_mask = (linear_means >= 99999.0) | (percentile_means >= 99999.0)
    means[invalid_mask] = 99999.0
    variances[invalid_mask] = 0.0
    return means, variances


def _sample_cost_vector(
    means: np.ndarray,
    variances: np.ndarray,
    invalid_cost: float,
    rng: np.random.Generator,
) -> np.ndarray:
    samples = np.full(means.shape, invalid_cost, dtype="float64")
    zero_mask = means == 0.0
    positive_mask = np.isfinite(means) & (means > 0.0) & (means < invalid_cost)
    if positive_mask.any():
        mean_values = means[positive_mask] + EPSILON
        mean_values[mean_values >= 1.0] = 1.0 - (EPSILON * 1.1)
        variance_values = variances[positive_mask] + EPSILON
        alpha_values = (
            ((1.0 - mean_values) / variance_values) - (1.0 / mean_values)
        ) * np.square(mean_values)
        alpha_values[alpha_values < 0.0] = EPSILON
        beta_values = alpha_values * ((1.0 / mean_values) - 1.0)
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
        key = int(float(value)) if feature_name in context.continuous_feature_names else str(value)
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

    cat_count_dists = (cfs_array[..., cat_idx] != query_array[..., cat_idx]).astype(int).mean(axis=-1)
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
    proximity = _merge_cat_cont(
        float(1.0 - cat_dists.mean()),
        float(1.0 - cont_norm_dists.mean()),
        context,
    )

    if len(unique_valid_cfs) <= 1:
        diversity = 0.0
    else:
        pairwise_cat = []
        pairwise_cont_norm = []
        for first_index in range(len(unique_valid_cfs)):
            for second_index in range(first_index + 1, len(unique_valid_cfs)):
                pair_cat, _pair_cont_mad, _pair_cont_count, pair_cont_norm = _pairwise_distance(
                    unique_valid_cfs[first_index],
                    unique_valid_cfs[second_index],
                    context,
                )
                pairwise_cat.append(float(np.asarray(pair_cat).reshape(-1)[0]))
                pairwise_cont_norm.append(float(np.asarray(pair_cont_norm).reshape(-1)[0]))
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
        "final_cost": final_cost,
    }


def _scale_metric_for_report(metric_name: str, value: float) -> float:
    if metric_name == "PAC":
        return float(value)
    return float(value * 100.0)


def _build_runtime_and_method(
    *,
    model: ReferenceBinaryMlpModel,
    trainset,
    data_cfg: dict[str, Any],
    method_cfg: dict[str, Any],
    factual_encoded: pd.DataFrame,
    factual_raw: pd.DataFrame,
    scaling_stats: ScalingStats,
    evaluation_rng: np.random.Generator,
    row_index: int,
) -> tuple[ColsMethod, dict[str, object], RuntimeSearchContext, int, int]:
    cols_method = ColsMethod(
        target_model=model,
        seed=int(method_cfg["seed"]),
        device=str(method_cfg["device"]),
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

    factual_encoded_row = factual_encoded.iloc[[row_index]].reset_index(drop=True)
    factual_state = decode_feature_dataframe(factual_encoded_row, cols_method._schema).iloc[0].to_dict()
    original_prediction = int(_predict_label_indices(model, factual_encoded_row)[0])
    target_index = resolve_target_index(
        model,
        original_prediction=original_prediction,
        desired_class=method_cfg.get("desired_class"),
    )
    runtime_context = build_runtime_context(
        schema=cols_method._schema,
        decoded_training=cols_method._decoded_training,
        base_state_spaces=cols_method._base_state_spaces,
        factual_state=factual_state,
        num_mcmc=int(method_cfg["num_mcmc"]),
        alpha=method_cfg.get("alpha"),
        variance=float(method_cfg["variance"]),
        invalid_cost=float(method_cfg["invalid_cost"]),
        rng=evaluation_rng,
    )
    return cols_method, factual_state, runtime_context, target_index, original_prediction


def _objective_score_from_set_metrics(
    objective_name: str,
    set_metrics: dict[str, float | bool],
) -> float:
    objective_name = str(objective_name).lower()
    validity = float(set_metrics["Val"])
    if objective_name == "proximity":
        return (-float(set_metrics["Prox"]) - validity) / 2.0
    if objective_name == "diversity":
        return (-float(set_metrics["Div"]) - validity) / 2.0
    if objective_name == "sparsity":
        return (-float(set_metrics["Spars"]) - validity) / 2.0
    raise ValueError(f"Unsupported distance objective: {objective_name}")


def _evaluate_search_candidate(
    *,
    model: ReferenceBinaryMlpModel,
    cols_method: ColsMethod,
    candidate_states: pd.DataFrame,
    runtime_context: RuntimeSearchContext,
    factual_raw_row: pd.Series,
    original_prediction: int,
    metric_context: PaperMetricContext,
    data_cfg: dict[str, Any],
    scaling_stats: ScalingStats,
    objective_name: str,
    query_count: int,
) -> SearchRunArtifacts:
    encoded_set = encode_state_dataframe(candidate_states, runtime_context.schema)
    pred_classes = _predict_label_indices(model, encoded_set)
    query_count += int(encoded_set.shape[0])
    target_index = 1 - int(original_prediction)
    valid_mask = pred_classes.astype(int) == int(target_index)

    if objective_name == "cost_simple":
        cost_matrix = compute_candidate_cost_matrix(candidate_states, runtime_context, valid_mask)
        score = float(compute_emc(cost_matrix, invalid_cost=metric_context.invalid_cost))
    else:
        cf_raw = _decode_counterfactual_set(
            encoded_set,
            cols_method,
            data_cfg,
            scaling_stats,
        )
        cf_values = cf_raw.loc[:, metric_context.feature_names].astype(str).to_numpy()
        query_values = factual_raw_row.loc[metric_context.feature_names].astype(str).to_numpy()
        set_metrics = _compute_set_metrics(
            query_values=query_values,
            cf_values=cf_values,
            pred_classes=pred_classes,
            original_class=original_prediction,
            context=metric_context,
        )
        score = float(_objective_score_from_set_metrics(objective_name, set_metrics))

    return SearchRunArtifacts(
        encoded_set=encoded_set.reset_index(drop=True),
        pred_classes=pred_classes,
        valid_mask=valid_mask.astype(bool, copy=False),
        score=score,
        num_queries=query_count,
    )


def _run_local_search_objective(
    *,
    model: ReferenceBinaryMlpModel,
    trainset,
    method_cfg: dict[str, Any],
    factual_encoded_row: pd.DataFrame,
    factual_raw_row: pd.Series,
    metric_context: PaperMetricContext,
    data_cfg: dict[str, Any],
    scaling_stats: ScalingStats,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    local_cfg = copy.deepcopy(method_cfg)
    evaluation_rng = np.random.default_rng(int(local_cfg["seed"]))
    cols_method, factual_state, runtime_context, target_index, original_prediction = _build_runtime_and_method(
        model=model,
        trainset=trainset,
        data_cfg=data_cfg,
        method_cfg=local_cfg,
        factual_encoded=factual_encoded_row,
        factual_raw=factual_raw_row.to_frame().T,
        scaling_stats=scaling_stats,
        evaluation_rng=evaluation_rng,
        row_index=0,
    )

    rng = np.random.default_rng(int(local_cfg["seed"]))
    objective_name = str(local_cfg["objective"]).lower()
    budget = int(local_cfg["budget"])
    query_count = 0
    best_states: pd.DataFrame | None = None
    best_encoded: pd.DataFrame | None = None
    best_pred_classes: np.ndarray | None = None
    best_valid_mask: np.ndarray | None = None
    best_score = float("inf")
    current_states: pd.DataFrame | None = None
    init_type = str(local_cfg["init_type"])

    for attempt in range(16):
        candidate_states = cols_method._initialize_state_set(
            factual_state,
            runtime_context,
            rng,
            init_type=init_type,
        )
        candidate = _evaluate_search_candidate(
            model=model,
            cols_method=cols_method,
            candidate_states=candidate_states,
            runtime_context=runtime_context,
            factual_raw_row=factual_raw_row,
            original_prediction=original_prediction,
            metric_context=metric_context,
            data_cfg=data_cfg,
            scaling_stats=scaling_stats,
            objective_name=objective_name,
            query_count=query_count,
        )
        query_count = candidate.num_queries
        if candidate.valid_mask.sum() > 0:
            best_states = candidate_states.reset_index(drop=True)
            current_states = candidate_states.reset_index(drop=True)
            best_encoded = candidate.encoded_set
            best_pred_classes = candidate.pred_classes
            best_valid_mask = candidate.valid_mask
            best_score = float(candidate.score)
            break
        if attempt >= 9:
            init_type = "random"

    if best_states is None or best_encoded is None or best_pred_classes is None or best_valid_mask is None:
        raise RuntimeError("Local-search initialization failed to find any valid counterfactuals")

    while query_count < budget:
        base_states = best_states if str(local_cfg["iter_type"]).lower() == "best" else current_states
        if base_states is None:
            break
        candidate_states = cols_method._perturb_state_set(
            base_states,
            runtime_context,
            rng,
        )
        current_states = candidate_states.reset_index(drop=True)
        candidate = _evaluate_search_candidate(
            model=model,
            cols_method=cols_method,
            candidate_states=candidate_states,
            runtime_context=runtime_context,
            factual_raw_row=factual_raw_row,
            original_prediction=original_prediction,
            metric_context=metric_context,
            data_cfg=data_cfg,
            scaling_stats=scaling_stats,
            objective_name=objective_name,
            query_count=query_count,
        )
        query_count = candidate.num_queries
        if float(candidate.score) < best_score:
            best_states = candidate_states.reset_index(drop=True)
            best_encoded = candidate.encoded_set
            best_pred_classes = candidate.pred_classes
            best_valid_mask = candidate.valid_mask
            best_score = float(candidate.score)

    return best_encoded, best_pred_classes, best_valid_mask, best_score


def _evaluate_method(
    *,
    method_name: str,
    method_cfg: dict[str, Any],
    model: ReferenceBinaryMlpModel,
    trainset,
    factual_encoded: pd.DataFrame,
    factual_raw: pd.DataFrame,
    user_costs: list[dict[str, np.ndarray]],
    factual_predictions: np.ndarray,
    metric_context: PaperMetricContext,
    data_cfg: dict[str, Any],
    scaling_stats: ScalingStats,
    device: str,
    progress_mode: str,
    heartbeat_seconds: int,
    progress_position: int = 0,
) -> tuple[dict[str, float], dict[str, float], list[PerFactualResult]]:
    resolved_method_cfg = copy.deepcopy(method_cfg)
    resolved_method_cfg["device"] = device

    if str(resolved_method_cfg.get("search", "cols")).lower() == "cols":
        cols_method = ColsMethod(
            target_model=model,
            seed=int(resolved_method_cfg["seed"]),
            device=device,
            desired_class=resolved_method_cfg.get("desired_class"),
            num_cfs=int(resolved_method_cfg["num_cfs"]),
            num_mcmc=int(resolved_method_cfg["num_mcmc"]),
            budget=int(resolved_method_cfg["budget"]),
            num_parallel_runs=int(resolved_method_cfg["num_parallel_runs"]),
            hamming_dist=int(resolved_method_cfg["hamming_dist"]),
            perturb_type=str(resolved_method_cfg["perturb_type"]),
            init_type=str(resolved_method_cfg["init_type"]),
            iter_type=str(resolved_method_cfg["iter_type"]),
            alpha=resolved_method_cfg.get("alpha"),
            variance=float(resolved_method_cfg["variance"]),
            invalid_cost=float(resolved_method_cfg["invalid_cost"]),
        )
        cols_method.fit(trainset)
        cf_sets_encoded = cols_method.get_counterfactual_sets(
            factual_encoded,
            show_progress=_progress_enabled(progress_mode),
            search_progress=_search_progress_enabled(progress_mode),
            heartbeat_seconds=heartbeat_seconds,
            progress_desc=f"{method_name} factuals",
            progress_position=progress_position,
        )
        cf_pred_classes_list = [
            _predict_label_indices(model, cf_encoded) for cf_encoded in cf_sets_encoded
        ]
        cf_validity_masks = [
            mask.to_numpy(copy=True)
            for mask in cols_method._last_counterfactual_validity
        ]
        search_stats = list(cols_method._last_search_stats)
    else:
        cf_sets_encoded = []
        cf_pred_classes_list = []
        cf_validity_masks = []
        search_stats = []
        factual_iterator = _build_progress_bar(
            range(factual_encoded.shape[0]),
            total=factual_encoded.shape[0],
            desc=f"{method_name} factuals",
            position=progress_position,
            leave=False,
            disable=not _progress_enabled(progress_mode),
        )
        for row_index in factual_iterator:
            cf_encoded, cf_pred_classes, validity_mask, score = _run_local_search_objective(
                model=model,
                trainset=trainset,
                method_cfg=resolved_method_cfg,
                factual_encoded_row=factual_encoded.iloc[[row_index]].reset_index(drop=True),
                factual_raw_row=factual_raw.iloc[row_index],
                metric_context=metric_context,
                data_cfg=data_cfg,
                scaling_stats=scaling_stats,
            )
            cf_sets_encoded.append(cf_encoded)
            cf_pred_classes_list.append(cf_pred_classes)
            cf_validity_masks.append(validity_mask)
            search_stats.append(
                {
                    "row_index": row_index,
                    "target_index": 1 - int(factual_predictions[row_index]),
                    "emc": score,
                    "num_valid": int(validity_mask.sum()),
                    "num_queries": int(cf_encoded.shape[0]),
                }
            )
        factual_iterator.close()
        cols_method = ColsMethod(
            target_model=model,
            seed=int(resolved_method_cfg["seed"]),
            device=device,
            desired_class=resolved_method_cfg.get("desired_class"),
            num_cfs=int(resolved_method_cfg["num_cfs"]),
            num_mcmc=int(resolved_method_cfg["num_mcmc"]),
            budget=int(resolved_method_cfg["budget"]),
            num_parallel_runs=int(resolved_method_cfg["num_parallel_runs"]),
            hamming_dist=int(resolved_method_cfg["hamming_dist"]),
            perturb_type=str(resolved_method_cfg["perturb_type"]),
            init_type=str(resolved_method_cfg["init_type"]),
            iter_type=str(resolved_method_cfg["iter_type"]),
            alpha=resolved_method_cfg.get("alpha"),
            variance=float(resolved_method_cfg["variance"]),
            invalid_cost=float(resolved_method_cfg["invalid_cost"]),
        )
        cols_method.fit(trainset)

    cost_rows = []
    set_rows = []
    fairness_rows: list[PerFactualResult] = []
    evaluation_iterator = _build_progress_bar(
        range(len(cf_sets_encoded)),
        total=len(cf_sets_encoded),
        desc=f"{method_name} evaluate",
        position=progress_position,
        leave=False,
        disable=not _progress_enabled(progress_mode),
    )
    last_heartbeat = time.monotonic()
    for factual_index in evaluation_iterator:
        cf_encoded = cf_sets_encoded[factual_index]
        cf_pred_classes = cf_pred_classes_list[factual_index]
        validity_mask = cf_validity_masks[factual_index].astype(bool)
        target_index = int(resolved_method_cfg["desired_class"])
        if not np.array_equal(validity_mask, cf_pred_classes == target_index):
            raise AssertionError("Stored validity mask does not match model predictions")

        cf_raw = _decode_counterfactual_set(cf_encoded, cols_method, data_cfg, scaling_stats)
        cf_values = cf_raw.loc[:, metric_context.feature_names].astype(str).to_numpy()
        query_values = factual_raw.iloc[factual_index].loc[metric_context.feature_names].astype(str).to_numpy()
        original_class = int(factual_predictions[factual_index])

        cost_metrics = _compute_cost_metrics(
            cf_values=cf_values,
            pred_classes=cf_pred_classes,
            original_class=original_class,
            user_cost=user_costs[factual_index],
            context=metric_context,
            cost_threshold=float(data_cfg["evaluation"]["cost_threshold"]),
            coverage_threshold=float(data_cfg["evaluation"]["coverage_threshold"]),
        )
        set_metrics = _compute_set_metrics(
            query_values=query_values,
            cf_values=cf_values,
            pred_classes=cf_pred_classes,
            original_class=original_class,
            context=metric_context,
        )
        cost_rows.append(cost_metrics)
        set_rows.append(set_metrics)
        fairness_rows.append(
            PerFactualResult(
                subgroup_values={
                    feature_name: str(factual_raw.iloc[factual_index][feature_name])
                    for feature_name in data_cfg["fairness"]["subgroup_features"]
                },
                final_cost=float(cost_metrics["final_cost"]),
                fs_at_1=float(cost_metrics["FS@1"]),
                cov=float(cost_metrics["Cov"]),
            )
        )

        if _progress_enabled(progress_mode):
            evaluation_iterator.set_postfix(
                valid=int(search_stats[factual_index]["num_valid"]),
                queries=int(search_stats[factual_index]["num_queries"]),
                score=f"{float(search_stats[factual_index]['emc']):.4f}",
            )
        last_heartbeat = _maybe_write_heartbeat(
            enabled=_progress_enabled(progress_mode),
            heartbeat_seconds=heartbeat_seconds,
            last_heartbeat=last_heartbeat,
            message=(
                f"[evaluate] method={method_name} "
                f"factual={factual_index + 1}/{len(cf_sets_encoded)} "
                f"queries={int(search_stats[factual_index]['num_queries'])}"
            ),
        )
    evaluation_iterator.close()

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
        "mean_queries": float(np.mean([float(stats["num_queries"]) for stats in search_stats])),
        "mean_score": float(np.mean([float(stats["emc"]) for stats in search_stats])),
        "mean_valid_cfs": float(np.mean([float(stats["num_valid"]) for stats in search_stats])),
        "coverage_fraction": float(np.mean(covered_mask)),
    }
    return aggregated, search_summary, fairness_rows


def _run_single_seed_for_dataset(
    *,
    run_seed: int,
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    method_map: dict[str, dict[str, Any]],
    device: str,
    progress_mode: str,
    heartbeat_seconds: int,
) -> TableRunResult:
    raw_df = _load_raw_dataframe(dataset_cfg)
    split_artifacts = _build_reference_split(raw_df, dataset_cfg)
    scaling_stats = _compute_scaling_stats(
        split_artifacts.train_df,
        list(dataset_cfg["continuous_feature_order"]),
    )

    template = _build_dataset_template(dataset_cfg)
    balanced_features = _encode_raw_features(split_artifacts.balanced_df, dataset_cfg, scaling_stats)
    train_features = _encode_raw_features(split_artifacts.train_df, dataset_cfg, scaling_stats)
    val_features = _encode_raw_features(split_artifacts.val_df, dataset_cfg, scaling_stats)
    provisional_features = _encode_raw_features(
        split_artifacts.provisional_factual_df,
        dataset_cfg,
        scaling_stats,
    )

    trainset = _build_frozen_dataset(
        template,
        train_features,
        split_artifacts.train_df[dataset_cfg["target_column"]],
        dataset_cfg,
        "trainset",
        extra_attrs={
            "cols_feature_space_df": balanced_features,
            "cols_state_space_overrides": _build_cols_state_space_overrides(
                split_artifacts.balanced_df,
                scaling_stats,
                dataset_cfg,
            ),
        },
    )
    valset = _build_frozen_dataset(
        template,
        val_features,
        split_artifacts.val_df[dataset_cfg["target_column"]],
        dataset_cfg,
        "valset",
    )

    model_cfg = dataset_cfg["model"]
    model = ReferenceBinaryMlpModel(
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
    model.fit(
        trainset,
        valset=valset,
        show_progress=_progress_enabled(progress_mode),
        progress_desc=f"{dataset_name} seed {run_seed} train",
        heartbeat_seconds=heartbeat_seconds,
    )

    provisional_predictions = _predict_label_indices(model, provisional_features)
    keep_mask = provisional_predictions == int(dataset_cfg["undesired_class"])
    factual_raw = split_artifacts.provisional_factual_df.loc[keep_mask].reset_index(drop=True)
    factual_encoded = provisional_features.loc[keep_mask].reset_index(drop=True)

    target_factual_count = int(dataset_cfg["split"]["target_factual_count"])
    if factual_raw.shape[0] > target_factual_count:
        factual_raw = factual_raw.iloc[:target_factual_count].reset_index(drop=True)
        factual_encoded = factual_encoded.iloc[:target_factual_count].reset_index(drop=True)
    if factual_raw.empty:
        raise RuntimeError(f"Model filtering produced no factuals for {dataset_name}")

    metric_context = _build_metric_context(
        split_artifacts.balanced_df.loc[:, list(dataset_cfg["raw_feature_order"]) + [dataset_cfg["target_column"]]],
        split_artifacts.train_df.loc[:, list(dataset_cfg["raw_feature_order"]) + [dataset_cfg["target_column"]]],
        dataset_cfg,
        dataset_cfg["evaluation"],
    )

    evaluation_rng = np.random.default_rng(run_seed)
    user_costs = [
        _sample_user_cost(
            factual_raw.iloc[row_index].loc[metric_context.feature_names],
            metric_context,
            dataset_cfg["evaluation"].get("alpha"),
            evaluation_rng,
        )
        for row_index in range(factual_raw.shape[0])
    ]
    factual_predictions = _predict_label_indices(model, factual_encoded)

    method_results: dict[str, dict[str, Any]] = {}
    for method_name, base_method_cfg in method_map.items():
        method_cfg = copy.deepcopy(base_method_cfg)
        method_cfg["seed"] = run_seed
        metrics, search_summary, fairness_rows = _evaluate_method(
            method_name=method_name,
            method_cfg=method_cfg,
            model=model,
            trainset=trainset,
            factual_encoded=factual_encoded,
            factual_raw=factual_raw,
            user_costs=user_costs,
            factual_predictions=factual_predictions,
            metric_context=metric_context,
            data_cfg=dataset_cfg,
            scaling_stats=scaling_stats,
            device=device,
            progress_mode=progress_mode,
            heartbeat_seconds=heartbeat_seconds,
        )
        method_results[method_name] = {
            "metrics": metrics,
            "search": search_summary,
            "fairness_rows": fairness_rows,
        }

    return TableRunResult(
        seed=run_seed,
        dataset=dataset_name,
        methods=method_results,
        val_accuracy=float(model._best_val_accuracy) if model._best_val_accuracy is not None else float("nan"),
        factual_rows=int(factual_raw.shape[0]),
    )


def _aggregate_table1(
    run_results: list[TableRunResult],
    targets: dict[str, dict[str, dict[str, float]]],
) -> pd.DataFrame:
    records = []
    for dataset_name, method_map in targets.items():
        dataset_runs = [result for result in run_results if result.dataset == dataset_name]
        for method_name, metric_targets in method_map.items():
            for metric_name, paper_value in metric_targets.items():
                values = [
                    float(result.methods[method_name]["metrics"][metric_name])
                    for result in dataset_runs
                ]
                records.append(
                    {
                        "table": "Table 1",
                        "dataset": dataset_name,
                        "method": method_name,
                        "metric": metric_name,
                        "paper": float(paper_value),
                        "reproduced_mean": float(np.nanmean(values)),
                        "reproduced_std": float(np.nanstd(values, ddof=0)),
                        "absolute_gap": float(abs(np.nanmean(values) - float(paper_value))),
                    }
                )
    return pd.DataFrame.from_records(records)


def _aggregate_table2(
    run_results: list[TableRunResult],
    targets: dict[str, dict[str, float]],
) -> pd.DataFrame:
    records = []
    for method_name, metric_targets in targets.items():
        for metric_name, paper_value in metric_targets.items():
            values = [
                float(result.methods[method_name]["metrics"][metric_name])
                for result in run_results
            ]
            records.append(
                {
                    "table": "Table 2",
                    "method": method_name,
                    "metric": metric_name,
                    "paper": float(paper_value),
                    "reproduced_mean": float(np.nanmean(values)),
                    "reproduced_std": float(np.nanstd(values, ddof=0)),
                    "absolute_gap": float(abs(np.nanmean(values) - float(paper_value))),
                }
            )
    return pd.DataFrame.from_records(records)


def _aggregate_table3(
    run_results: list[TableRunResult],
    subgroup_feature: str,
    subgroup_order: list[str],
    targets: dict[str, dict[str, dict[str, float]]],
) -> pd.DataFrame:
    records = []
    for method_name, paper_rows in targets.items():
        female_label, male_label = subgroup_order
        female_fs = []
        female_cov = []
        male_fs = []
        male_cov = []
        for result in run_results:
            rows = result.methods[method_name]["fairness_rows"]
            female_rows = [row for row in rows if row.subgroup_values[subgroup_feature] == female_label]
            male_rows = [row for row in rows if row.subgroup_values[subgroup_feature] == male_label]
            female_fs.append(float(np.mean([row.fs_at_1 for row in female_rows])) * 100.0)
            female_cov.append(float(np.mean([row.cov for row in female_rows])) * 100.0)
            male_fs.append(float(np.mean([row.fs_at_1 for row in male_rows])) * 100.0)
            male_cov.append(float(np.mean([row.cov for row in male_rows])) * 100.0)

        reproduced = {
            female_label: {
                "FS@1": float(np.mean(female_fs)),
                "Cov": float(np.mean(female_cov)),
            },
            male_label: {
                "FS@1": float(np.mean(male_fs)),
                "Cov": float(np.mean(male_cov)),
            },
        }
        reproduced["DIR"] = {
            "DIR-FS": (
                reproduced[male_label]["FS@1"] / reproduced[female_label]["FS@1"]
                if reproduced[female_label]["FS@1"] > 0.0
                else float("nan")
            ),
            "DIR-Cov": (
                reproduced[male_label]["Cov"] / reproduced[female_label]["Cov"]
                if reproduced[female_label]["Cov"] > 0.0
                else float("nan")
            ),
        }

        for subgroup_name, metric_targets in paper_rows.items():
            if subgroup_name == "DIR":
                source = reproduced["DIR"]
            else:
                source = reproduced[subgroup_name]
            for metric_name, paper_value in metric_targets.items():
                value = float(source[metric_name])
                records.append(
                    {
                        "table": "Table 3",
                        "method": method_name,
                        "subgroup": subgroup_name,
                        "metric": metric_name,
                        "paper": float(paper_value),
                        "reproduced_mean": value,
                        "reproduced_std": 0.0,
                        "absolute_gap": float(abs(value - float(paper_value))),
                    }
                )
    return pd.DataFrame.from_records(records)


def _build_report_entries(
    comparison_tables: list[pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    experiments: dict[str, dict[str, Any]] = {}
    for table in comparison_tables:
        for row in table.to_dict(orient="records"):
            key_parts = [
                str(row["table"]).replace(" ", "_"),
                str(row.get("dataset", "")),
                str(row["method"]),
                str(row.get("subgroup", "")),
                str(row["metric"]),
            ]
            key = "_".join([part for part in key_parts if part])
            experiments[key] = {
                "configuration": {
                    "table": row["table"],
                    "dataset": row.get("dataset"),
                    "method": row["method"],
                    "subgroup": row.get("subgroup"),
                    "metric": row["metric"],
                },
                "metrics": {
                    "reproduced_mean": {
                        "original": row["paper"],
                        "reproduced": row["reproduced_mean"],
                    },
                    "reproduced_std": {
                        "original": None,
                        "reproduced": row["reproduced_std"],
                    },
                },
            }
    return experiments


def _parse_method_filter(methods_arg: str | None) -> set[str] | None:
    if methods_arg is None:
        return None
    requested = {method.strip() for method in methods_arg.split(",") if method.strip()}
    if not requested:
        raise ValueError("--methods must include at least one method name")
    return requested


def _normalize_method_filter_name(method_name: str) -> str:
    return "".join(ch for ch in method_name.lower() if ch.isalnum())


def _filter_table_methods(
    table_name: str,
    table_cfg: dict[str, Any],
    requested: set[str],
) -> None:
    methods = table_cfg.get("methods", {})
    available = set(methods)
    requested_aliases = {_normalize_method_filter_name(name) for name in requested}
    selected = {
        name: cfg
        for name, cfg in methods.items()
        if name in requested or _normalize_method_filter_name(name) in requested_aliases
    }
    if not selected:
        raise ValueError(
            f"No requested methods are available for {table_name}. "
            f"Requested {sorted(requested)}, available {sorted(available)}."
        )
    table_cfg["methods"] = selected

    paper_targets = table_cfg.get("paper_targets", {})
    if table_name == "table1":
        table_cfg["paper_targets"] = {
            dataset_name: {
                method_name: targets
                for method_name, targets in dataset_targets.items()
                if method_name in selected
            }
            for dataset_name, dataset_targets in paper_targets.items()
        }
        return

    table_cfg["paper_targets"] = {
        method_name: targets
        for method_name, targets in paper_targets.items()
        if method_name in selected
    }


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = _apply_profile(config, getattr(args, "profile", None))
    requested_methods = _parse_method_filter(getattr(args, "methods", None))
    if requested_methods is not None:
        if args.tables is not None:
            selected_table_names = set(args.tables)
        elif args.all_tables:
            selected_table_names = set(cfg["reproduction"]["tables"].keys())
        else:
            selected_table_names = {"table1"}
        for table_name in selected_table_names:
            _filter_table_methods(
                table_name,
                cfg["reproduction"]["tables"][table_name],
                requested_methods,
            )
    if args.max_runs is not None:
        cfg["reproduction"]["run_seeds"] = cfg["reproduction"]["run_seeds"][: int(args.max_runs)]
    if args.max_factuals is not None:
        for dataset_cfg in cfg["datasets"].values():
            dataset_cfg["split"]["target_factual_count"] = int(args.max_factuals)
    if args.override_epochs is not None:
        for dataset_cfg in cfg["datasets"].values():
            dataset_cfg["model"]["epochs"] = int(args.override_epochs)
    if args.override_budget is not None:
        for table_cfg in cfg["reproduction"]["tables"].values():
            for method_cfg in table_cfg["methods"].values():
                method_cfg["budget"] = int(args.override_budget)
    if args.override_num_mcmc is not None:
        for table_cfg in cfg["reproduction"]["tables"].values():
            for method_cfg in table_cfg["methods"].values():
                method_cfg["num_mcmc"] = int(args.override_num_mcmc)
    if args.use_reference_checkpoint:
        for dataset_cfg in cfg["datasets"].values():
            dataset_cfg["model"]["use_reference_checkpoint"] = True
    if getattr(args, "train_model_from_scratch", False):
        for dataset_cfg in cfg["datasets"].values():
            dataset_cfg["model"]["use_reference_checkpoint"] = False
    return cfg


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--tables", nargs="+", choices=["table1", "table2", "table3"], default=None)
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated method rows to run, for example: cols-emc or 'COLS,P-COLS'.",
    )
    parser.add_argument("--all-tables", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-factuals", type=int, default=None)
    parser.add_argument("--override-epochs", type=int, default=None)
    parser.add_argument("--override-budget", type=int, default=None)
    parser.add_argument("--override-num-mcmc", type=int, default=None)
    parser.add_argument("--use-reference-checkpoint", action="store_true")
    parser.add_argument("--train-model-from-scratch", action="store_true")
    parser.add_argument("--progress", type=str, default="standard")
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    return parser.parse_args(argv)


def _method_subset_matches(
    required: dict[str, dict[str, Any]],
    available: dict[str, dict[str, Any]],
) -> bool:
    for method_name, required_cfg in required.items():
        available_cfg = available.get(method_name)
        if available_cfg != required_cfg:
            return False
    return True


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = (PROJECT_ROOT / args.config).resolve()
    if not config_path.exists():
        config_path = Path(args.config).resolve()
    report_path_arg = Path(args.report_path)
    report_output_path = (
        (PROJECT_ROOT / report_path_arg).resolve()
        if not report_path_arg.is_absolute()
        else report_path_arg.resolve()
    )
    config = _apply_cli_overrides(_load_config(config_path), args)
    progress_mode = _resolve_progress_mode(args.progress)
    heartbeat_seconds = max(1, int(args.heartbeat_seconds))

    if args.tables is not None:
        selected_tables = list(args.tables)
    elif args.all_tables:
        selected_tables = list(config["reproduction"]["tables"].keys())
    else:
        # Match the original workflow more closely by defaulting to the main result table only.
        selected_tables = ["table1"]
    run_seeds = [int(seed) for seed in config["reproduction"]["run_seeds"]]
    comparison_tables: list[pd.DataFrame] = []
    table1_run_results_by_dataset: dict[str, list[TableRunResult]] = {}
    table1_method_map: dict[str, dict[str, Any]] | None = None

    if "table1" in selected_tables:
        table_cfg = config["reproduction"]["tables"]["table1"]
        table1_method_map = copy.deepcopy(table_cfg["methods"])
        run_results = []
        for dataset_name in table_cfg["datasets"]:
            dataset_cfg = copy.deepcopy(config["datasets"][dataset_name])
            device = _resolve_device(str(dataset_cfg["model"]["device"]).lower())
            dataset_run_results: list[TableRunResult] = []
            for run_seed in run_seeds:
                result = _run_single_seed_for_dataset(
                    run_seed=run_seed,
                    dataset_name=dataset_name,
                    dataset_cfg=dataset_cfg,
                    method_map=copy.deepcopy(table_cfg["methods"]),
                    device=device,
                    progress_mode=progress_mode,
                    heartbeat_seconds=heartbeat_seconds,
                )
                run_results.append(result)
                dataset_run_results.append(result)
            table1_run_results_by_dataset[dataset_name] = dataset_run_results
        comparison = _aggregate_table1(run_results, table_cfg["paper_targets"])
        comparison_tables.append(comparison)
        print("\nTable 1 Comparison")
        print(comparison.to_string(index=False))

    if "table2" in selected_tables:
        table_cfg = config["reproduction"]["tables"]["table2"]
        dataset_name = str(table_cfg["dataset"])
        dataset_cfg = copy.deepcopy(config["datasets"][dataset_name])
        device = _resolve_device(str(dataset_cfg["model"]["device"]).lower())
        run_results = [
            _run_single_seed_for_dataset(
                run_seed=run_seed,
                dataset_name=dataset_name,
                dataset_cfg=dataset_cfg,
                method_map=copy.deepcopy(table_cfg["methods"]),
                device=device,
                progress_mode=progress_mode,
                heartbeat_seconds=heartbeat_seconds,
            )
            for run_seed in run_seeds
        ]
        comparison = _aggregate_table2(run_results, table_cfg["paper_targets"])
        comparison_tables.append(comparison)
        print("\nTable 2 Comparison")
        print(comparison.to_string(index=False))

    if "table3" in selected_tables:
        table_cfg = config["reproduction"]["tables"]["table3"]
        dataset_name = str(table_cfg["dataset"])
        table3_method_map = copy.deepcopy(table_cfg["methods"])
        can_reuse_table1 = (
            dataset_name in table1_run_results_by_dataset
            and table1_method_map is not None
            and _method_subset_matches(table3_method_map, table1_method_map)
        )
        if can_reuse_table1:
            run_results = table1_run_results_by_dataset[dataset_name]
        else:
            dataset_cfg = copy.deepcopy(config["datasets"][dataset_name])
            device = _resolve_device(str(dataset_cfg["model"]["device"]).lower())
            run_results = [
                _run_single_seed_for_dataset(
                    run_seed=run_seed,
                    dataset_name=dataset_name,
                    dataset_cfg=dataset_cfg,
                    method_map=table3_method_map,
                    device=device,
                    progress_mode=progress_mode,
                    heartbeat_seconds=heartbeat_seconds,
                )
                for run_seed in run_seeds
            ]
        fairness_cfg = table_cfg["fairness"]
        comparison = _aggregate_table3(
            run_results,
            subgroup_feature=str(fairness_cfg["subgroup_feature"]),
            subgroup_order=[str(value) for value in fairness_cfg["subgroup_order"]],
            targets=table_cfg["paper_targets"],
        )
        comparison_tables.append(comparison)
        print("\nTable 3 Comparison")
        print(comparison.to_string(index=False))

    report_path = write_reproduction_report(
        output_path=report_output_path,
        paper_id="cols_tables_1_2_3",
        reproduction_metadata={
            "timestamp": datetime.now(timezone.utc),
            "framework_version": "1.0.0",
            "source_script": Path(__file__).name,
            "config_path": str(config_path),
            "tables": selected_tables,
            "run_seeds": run_seeds,
        },
        experiments_data=_build_report_entries(comparison_tables),
    )
    print(f"reproduction_report_path: {report_path}")


@pytest.mark.slow
def test_reproduce() -> None:
    pytest.skip("Run this module as a script for bounded reproduction probes.")


if __name__ == "__main__":
    main()
