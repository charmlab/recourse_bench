from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from time import strftime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

import dataset  # noqa: F401
import evaluation  # noqa: F401
import method  # noqa: F401
import model  # noqa: F401
import preprocess  # noqa: F401
from experiment import Experiment
from utils.caching import set_cache_dir
from utils.logger import setup_logger
from utils.registry import get_registry

DEFAULT_CONFIG_PATH = Path(__file__).with_name("compas_mlp_dice_reproduce.yaml")
DEFAULT_MANUAL_KS = (1, 2, 4, 6, 8, 10)
DEFAULT_MANUAL_RADII = (0.5, 1.0, 2.0)


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _materialize_datasets(experiment: Experiment) -> tuple[object, object]:
    datasets = [experiment._raw_dataset]
    for preprocess_step in experiment._preprocess:
        next_datasets = []
        for current_dataset in datasets:
            transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                next_datasets.append(transformed)
        datasets = next_datasets
    return experiment._resolve_train_test(datasets)


def _build_method(method_config: dict, target_model):
    cfg = deepcopy(method_config)
    method_name = cfg.pop("name")
    return get_registry("method")[method_name](target_model=target_model, **cfg)


def _target_indices(dataset, class_to_index: dict[int | str, int]) -> np.ndarray:
    target = dataset.get(target=True).iloc[:, 0]
    resolved = []
    for value in target.tolist():
        if isinstance(value, float) and float(value).is_integer():
            value = int(value)
        resolved.append(class_to_index[value])
    return np.asarray(resolved, dtype=np.int64)


def _compute_model_accuracy(target_model, testset) -> float:
    prediction = (
        target_model.predict(testset, batch_size=512).argmax(dim=1).cpu().numpy()
    )
    y_true = _target_indices(testset, target_model.get_class_to_index())
    return float(np.mean(prediction == y_true))


def _select_factual_features(
    testset,
    target_model,
    desired_class: int | str,
    num_factuals: int,
    seed: int,
) -> pd.DataFrame:
    class_to_index = target_model.get_class_to_index()
    desired_index = int(class_to_index[desired_class])
    predicted = (
        target_model.predict(testset, batch_size=512).argmax(dim=1).cpu().numpy()
    )
    feature_df = testset.get(target=False)
    factual_pool = feature_df.loc[predicted != desired_index].copy(deep=True)
    if factual_pool.shape[0] < num_factuals:
        raise ValueError(
            f"Requested {num_factuals} factuals but only found {factual_pool.shape[0]}"
        )
    return factual_pool.sample(n=num_factuals, random_state=seed).copy(deep=True)


def _compute_mads(
    trainset,
    continuous_indices: tuple[int, ...],
    feature_names: list[str],
) -> dict[int, float]:
    feature_df = trainset.get(target=False).loc[:, feature_names]
    mads: dict[int, float] = {}
    for feature_index in continuous_indices:
        feature_name = feature_names[feature_index]
        series = feature_df[feature_name].astype("float64")
        median = float(series.median())
        mad = float(np.median(np.abs(series.to_numpy() - median)))
        mads[feature_index] = mad if mad > 0.0 else 1.0
    return mads


def _categorical_feature_difference(
    left: np.ndarray,
    right: np.ndarray,
    categorical_groups,
    binary_feature_indices: tuple[int, ...],
) -> float:
    total_features = len(categorical_groups) + len(binary_feature_indices)
    if total_features == 0:
        return 0.0

    differences = 0
    for group in categorical_groups:
        group_indices = list(group.indices)
        left_index = int(np.argmax(left[group_indices]))
        right_index = int(np.argmax(right[group_indices]))
        differences += int(left_index != right_index)
    for feature_index in binary_feature_indices:
        differences += int(not np.isclose(left[feature_index], right[feature_index]))
    return float(differences / total_features)


def _continuous_feature_distance(
    left: np.ndarray,
    right: np.ndarray,
    continuous_indices: tuple[int, ...],
    mads: dict[int, float],
) -> float:
    if not continuous_indices:
        return 0.0
    distances = []
    for feature_index in continuous_indices:
        mad = mads.get(feature_index, 1.0)
        distances.append(abs(float(left[feature_index] - right[feature_index])) / mad)
    return float(np.mean(distances))


def _count_feature_difference(
    left: np.ndarray,
    right: np.ndarray,
    categorical_groups,
    binary_feature_indices: tuple[int, ...],
    continuous_indices: tuple[int, ...],
) -> float:
    total_features = (
        len(categorical_groups) + len(binary_feature_indices) + len(continuous_indices)
    )
    if total_features == 0:
        return 0.0

    differences = 0
    for group in categorical_groups:
        group_indices = list(group.indices)
        differences += int(
            int(np.argmax(left[group_indices])) != int(np.argmax(right[group_indices]))
        )
    for feature_index in binary_feature_indices:
        differences += int(not np.isclose(left[feature_index], right[feature_index]))
    for feature_index in continuous_indices:
        differences += int(not np.isclose(left[feature_index], right[feature_index]))
    return float(differences / total_features)


def _continuous_sparsity(
    candidate: np.ndarray,
    factual: np.ndarray,
    continuous_indices: tuple[int, ...],
) -> float:
    if not continuous_indices:
        return float("nan")
    changed = 0
    for feature_index in continuous_indices:
        changed += int(not np.isclose(candidate[feature_index], factual[feature_index]))
    return float(1.0 - changed / len(continuous_indices))


def _sample_boundary_points(
    factual: np.ndarray,
    num_samples: int,
    radius_multiplier: float,
    continuous_indices: tuple[int, ...],
    mads: dict[int, float],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    categorical_groups,
    binary_feature_value_map: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    samples = np.repeat(factual.reshape(1, -1), num_samples, axis=0).astype(np.float32)

    for feature_index in continuous_indices:
        radius = radius_multiplier * mads.get(feature_index, 1.0)
        low = max(lower_bounds[feature_index], factual[feature_index] - radius)
        high = min(upper_bounds[feature_index], factual[feature_index] + radius)
        if np.isclose(low, high):
            samples[:, feature_index] = low
        else:
            samples[:, feature_index] = rng.uniform(low, high, size=num_samples)

    for group in categorical_groups:
        group_indices = list(group.indices)
        chosen = rng.integers(0, len(group_indices), size=num_samples)
        samples[:, group_indices] = 0.0
        row_indices = np.arange(num_samples)
        samples[row_indices, np.asarray(group_indices)[chosen]] = 1.0

    for feature_index, allowed_values in binary_feature_value_map.items():
        samples[:, feature_index] = rng.choice(allowed_values, size=num_samples)

    return samples


def _aggregate_boundary_metrics(
    y_true: list[int],
    y_pred: list[int],
    desired_index: int,
) -> dict[str, float]:
    if not y_true:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[desired_index],
        average=None,
        zero_division=0,
    )
    return {
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
    }


def _evaluate_setting(
    factual_features: pd.DataFrame,
    target_model,
    dice_method,
    requested_k: int,
    num_boundary_samples: int,
    radii: tuple[float, ...],
    seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    feature_names = list(dice_method._feature_names)
    continuous_indices = tuple(dice_method._continuous_indices)
    categorical_groups = tuple(dice_method._categorical_groups)
    binary_feature_indices = tuple(sorted(dice_method._binary_feature_value_map))
    lower_bounds = (
        dice_method._search_metadata.lower_bounds.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    upper_bounds = (
        dice_method._search_metadata.upper_bounds.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    desired_class = dice_method._desired_class
    desired_index = int(target_model.get_class_to_index()[desired_class])
    factual_index = 1 - desired_index

    mads = _compute_mads(
        trainset=dice_method._trainset_reference,
        continuous_indices=continuous_indices,
        feature_names=feature_names,
    )

    summary_records: list[dict[str, float]] = []
    boundary_truth = {radius: [] for radius in radii}
    boundary_pred = {radius: [] for radius in radii}
    rng = np.random.default_rng(seed)

    iterator = tqdm(
        factual_features.iterrows(),
        total=factual_features.shape[0],
        desc=f"dice-k-{requested_k}",
        leave=False,
    )
    for _, row in iterator:
        factual = row.to_numpy(dtype=np.float32)
        counterfactuals = dice_method.get_diverse_counterfactuals(
            row.to_frame().T,
            total_cfs=requested_k,
        )
        candidate_array = counterfactuals.to_numpy(dtype=np.float32)
        validity = float(counterfactuals.shape[0] / requested_k)

        if candidate_array.shape[0] == 0:
            summary_records.append(
                {
                    "validity": validity,
                    "categorical_proximity": float("nan"),
                    "continuous_distance": float("nan"),
                    "continuous_proximity": float("nan"),
                    "categorical_diversity": float("nan"),
                    "continuous_diversity": float("nan"),
                    "count_diversity": float("nan"),
                    "continuous_sparsity": float("nan"),
                }
            )
            continue

        categorical_proximities = [
            1.0
            - _categorical_feature_difference(
                candidate,
                factual,
                categorical_groups=categorical_groups,
                binary_feature_indices=binary_feature_indices,
            )
            for candidate in candidate_array
        ]
        continuous_distances = [
            _continuous_feature_distance(
                candidate,
                factual,
                continuous_indices=continuous_indices,
                mads=mads,
            )
            for candidate in candidate_array
        ]
        continuous_sparsities = [
            _continuous_sparsity(
                candidate,
                factual,
                continuous_indices=continuous_indices,
            )
            for candidate in candidate_array
        ]

        if candidate_array.shape[0] < 2:
            categorical_diversity = 0.0
            continuous_diversity = 0.0
            count_diversity = 0.0
        else:
            pair_categorical = []
            pair_continuous = []
            pair_count = []
            for left_index in range(candidate_array.shape[0] - 1):
                for right_index in range(left_index + 1, candidate_array.shape[0]):
                    left = candidate_array[left_index]
                    right = candidate_array[right_index]
                    pair_categorical.append(
                        _categorical_feature_difference(
                            left,
                            right,
                            categorical_groups=categorical_groups,
                            binary_feature_indices=binary_feature_indices,
                        )
                    )
                    pair_continuous.append(
                        _continuous_feature_distance(
                            left,
                            right,
                            continuous_indices=continuous_indices,
                            mads=mads,
                        )
                    )
                    pair_count.append(
                        _count_feature_difference(
                            left,
                            right,
                            categorical_groups=categorical_groups,
                            binary_feature_indices=binary_feature_indices,
                            continuous_indices=continuous_indices,
                        )
                    )
            categorical_diversity = float(np.mean(pair_categorical))
            continuous_diversity = float(np.mean(pair_continuous))
            count_diversity = float(np.mean(pair_count))

        summary_records.append(
            {
                "validity": validity,
                "categorical_proximity": float(np.mean(categorical_proximities)),
                "continuous_distance": float(np.mean(continuous_distances)),
                "continuous_proximity": float(-np.mean(continuous_distances)),
                "categorical_diversity": categorical_diversity,
                "continuous_diversity": continuous_diversity,
                "count_diversity": count_diversity,
                "continuous_sparsity": float(np.mean(continuous_sparsities)),
            }
        )

        knn = KNeighborsClassifier(n_neighbors=1)
        training_X = np.vstack(
            [
                factual.reshape(1, -1),
                counterfactuals.to_numpy(dtype=np.float32),
            ]
        )
        training_y = np.asarray(
            [1 - desired_index] + [desired_index] * counterfactuals.shape[0],
            dtype=np.int64,
        )
        knn.fit(training_X, training_y)

        for radius in radii:
            boundary_points = _sample_boundary_points(
                factual=factual,
                num_samples=num_boundary_samples,
                radius_multiplier=radius,
                continuous_indices=continuous_indices,
                mads=mads,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                categorical_groups=categorical_groups,
                binary_feature_value_map=dice_method._binary_feature_value_map,
                rng=rng,
            )
            target_prediction = target_model.get_prediction(
                pd.DataFrame(boundary_points, columns=feature_names),
                proba=False,
            )
            y_true = target_prediction.argmax(dim=1).detach().cpu().numpy().tolist()
            y_pred = knn.predict(boundary_points).tolist()
            boundary_truth[radius].extend(y_true)
            boundary_pred[radius].extend(y_pred)

    summary_df = pd.DataFrame(summary_records)
    setting_summary = {
        column: float(summary_df[column].mean()) for column in summary_df.columns
    }
    setting_summary["factual_class_index"] = float(factual_index)
    setting_summary["desired_class_index"] = float(desired_index)

    boundary_rows = []
    for radius in radii:
        metrics = _aggregate_boundary_metrics(
            y_true=boundary_truth[radius],
            y_pred=boundary_pred[radius],
            desired_index=desired_index,
        )
        boundary_rows.append(
            {
                "radius_mad": float(radius),
                **metrics,
            }
        )
    return setting_summary, boundary_rows


def _evaluate_lime_proxy(
    factual_features: pd.DataFrame,
    target_model,
    schema_method,
    num_boundary_samples: int,
    radii: tuple[float, ...],
    lime_num_samples: int,
    seed: int,
) -> pd.DataFrame:
    feature_names = list(schema_method._feature_names)
    continuous_indices = tuple(schema_method._continuous_indices)
    categorical_groups = tuple(schema_method._categorical_groups)
    lower_bounds = (
        schema_method._search_metadata.lower_bounds.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    upper_bounds = (
        schema_method._search_metadata.upper_bounds.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    desired_class = schema_method._desired_class
    desired_index = int(target_model.get_class_to_index()[desired_class])
    train_radius = max(2.0, max(radii))
    mads = _compute_mads(
        trainset=schema_method._trainset_reference,
        continuous_indices=continuous_indices,
        feature_names=feature_names,
    )

    boundary_truth = {radius: [] for radius in radii}
    boundary_pred = {radius: [] for radius in radii}
    rng = np.random.default_rng(seed)

    iterator = tqdm(
        factual_features.iterrows(),
        total=factual_features.shape[0],
        desc="lime-proxy",
        leave=False,
    )
    for _, row in iterator:
        factual = row.to_numpy(dtype=np.float32)
        local_train = _sample_boundary_points(
            factual=factual,
            num_samples=lime_num_samples,
            radius_multiplier=train_radius,
            continuous_indices=continuous_indices,
            mads=mads,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            categorical_groups=categorical_groups,
            binary_feature_value_map=schema_method._binary_feature_value_map,
            rng=rng,
        )
        train_prediction = target_model.get_prediction(
            pd.DataFrame(local_train, columns=feature_names),
            proba=False,
        )
        y_local = train_prediction.argmax(dim=1).detach().cpu().numpy()

        lime_model = None
        constant_prediction: int | None = None
        if np.unique(y_local).shape[0] >= 2:
            lime_model = LogisticRegression(
                max_iter=1000,
                random_state=seed,
                solver="lbfgs",
            )
            lime_model.fit(local_train, y_local)
        else:
            constant_prediction = int(y_local[0])

        for radius in radii:
            boundary_points = _sample_boundary_points(
                factual=factual,
                num_samples=num_boundary_samples,
                radius_multiplier=radius,
                continuous_indices=continuous_indices,
                mads=mads,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                categorical_groups=categorical_groups,
                binary_feature_value_map=schema_method._binary_feature_value_map,
                rng=rng,
            )
            target_prediction = target_model.get_prediction(
                pd.DataFrame(boundary_points, columns=feature_names),
                proba=False,
            )
            y_true = target_prediction.argmax(dim=1).detach().cpu().numpy().tolist()
            if lime_model is None:
                y_pred = [constant_prediction] * len(y_true)
            else:
                y_pred = lime_model.predict(boundary_points).tolist()
            boundary_truth[radius].extend(y_true)
            boundary_pred[radius].extend(y_pred)

    rows = []
    for radius in radii:
        rows.append(
            {
                "radius_mad": float(radius),
                **_aggregate_boundary_metrics(
                    y_true=boundary_truth[radius],
                    y_pred=boundary_pred[radius],
                    desired_index=desired_index,
                ),
            }
        )
    return pd.DataFrame(rows)


def _get_summary_value(
    summary_df: pd.DataFrame,
    setting: str,
    k: int,
    column: str,
) -> float:
    match = summary_df.loc[
        (summary_df["setting"] == setting) & (summary_df["k"] == int(k)),
        column,
    ]
    if match.empty:
        return float("nan")
    return float(match.iloc[0])


def _get_boundary_value(
    boundary_df: pd.DataFrame,
    setting: str,
    k: int,
    radius: float,
    column: str,
) -> float:
    match = boundary_df.loc[
        (boundary_df["setting"] == setting)
        & (boundary_df["k"] == int(k))
        & (np.isclose(boundary_df["radius_mad"], float(radius))),
        column,
    ]
    if match.empty:
        return float("nan")
    return float(match.iloc[0])


def _get_lime_value(
    lime_df: pd.DataFrame,
    radius: float,
    column: str,
) -> float:
    match = lime_df.loc[np.isclose(lime_df["radius_mad"], float(radius)), column]
    if match.empty:
        return float("nan")
    return float(match.iloc[0])


def _append_check(
    checks: list[dict[str, object]],
    name: str,
    relation: str,
    actual: float,
    expected: float,
    passed: bool,
    margin: float,
) -> None:
    checks.append(
        {
            "name": name,
            "relation": relation,
            "actual": actual,
            "expected": expected,
            "passed": bool(passed),
            "margin": margin,
        }
    )


def _assert_paper_results(
    summary_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    lime_df: pd.DataFrame,
    config: dict,
) -> dict[str, object]:
    assertion_cfg = deepcopy(config.get("assertions", {}))
    checks: list[dict[str, object]] = []

    accuracy = float(summary_df["model_accuracy"].iloc[0])
    accuracy_target = float(assertion_cfg.get("accuracy_target", 0.67))
    accuracy_tolerance = float(assertion_cfg.get("accuracy_tolerance", 0.05))
    accuracy_margin = accuracy_tolerance - abs(accuracy - accuracy_target)
    _append_check(
        checks=checks,
        name="model_accuracy_close_to_paper",
        relation="abs_diff<=",
        actual=accuracy,
        expected=accuracy_target,
        passed=bool(accuracy_margin >= 0.0),
        margin=float(accuracy_margin),
    )

    requested_compare_ks = tuple(
        int(value) for value in assertion_cfg.get("compare_ks", [2, 4])
    )
    available_ks = tuple(
        sorted(int(value) for value in summary_df["k"].unique().tolist())
    )
    compare_ks = tuple(k for k in requested_compare_ks if k in set(available_ks))
    if not compare_ks:
        raise AssertionError(
            "Paper assertions could not run because none of the configured compare_ks "
            f"{requested_compare_ks} are present in summary_df k values {available_ks}"
        )
    validity_floors = {
        int(key): float(value)
        for key, value in assertion_cfg.get("diverse_validity_floor", {}).items()
    }
    diversity_epsilon = float(assertion_cfg.get("diversity_epsilon", 0.02))
    proximity_ratio_floor = float(
        assertion_cfg.get("categorical_proximity_ratio_floor", 0.7)
    )

    for k in compare_ks:
        diverse_validity = _get_summary_value(summary_df, "DiverseCF", k, "validity")
        validity_floor = float(validity_floors.get(k, 0.8))
        _append_check(
            checks=checks,
            name=f"diverse_validity_floor_k_{k}",
            relation=">=",
            actual=diverse_validity,
            expected=validity_floor,
            passed=bool(
                math.isfinite(diverse_validity) and diverse_validity >= validity_floor
            ),
            margin=(
                float(diverse_validity - validity_floor)
                if math.isfinite(diverse_validity)
                else float("-inf")
            ),
        )

        for metric_name in [
            "categorical_diversity",
            "continuous_diversity",
            "count_diversity",
        ]:
            diverse_metric = _get_summary_value(summary_df, "DiverseCF", k, metric_name)
            baseline_metric = max(
                _get_summary_value(summary_df, "NoDiversityCF", k, metric_name),
                _get_summary_value(summary_df, "RandomInitCF", k, metric_name),
            )
            expected_value = baseline_metric - diversity_epsilon
            passed = bool(
                math.isfinite(diverse_metric)
                and math.isfinite(expected_value)
                and diverse_metric >= expected_value
            )
            _append_check(
                checks=checks,
                name=f"diverse_{metric_name}_not_weaker_k_{k}",
                relation=">=",
                actual=diverse_metric,
                expected=expected_value,
                passed=passed,
                margin=(
                    float(diverse_metric - expected_value)
                    if math.isfinite(diverse_metric) and math.isfinite(expected_value)
                    else float("-inf")
                ),
            )

        diverse_cat_prox = _get_summary_value(
            summary_df,
            "DiverseCF",
            k,
            "categorical_proximity",
        )
        baseline_cat_prox = max(
            _get_summary_value(summary_df, "NoDiversityCF", k, "categorical_proximity"),
            _get_summary_value(summary_df, "RandomInitCF", k, "categorical_proximity"),
        )
        expected_cat_prox = proximity_ratio_floor * baseline_cat_prox
        passed = bool(
            math.isfinite(diverse_cat_prox)
            and math.isfinite(expected_cat_prox)
            and diverse_cat_prox >= expected_cat_prox
        )
        _append_check(
            checks=checks,
            name=f"diverse_categorical_proximity_ratio_k_{k}",
            relation=">=",
            actual=diverse_cat_prox,
            expected=expected_cat_prox,
            passed=passed,
            margin=(
                float(diverse_cat_prox - expected_cat_prox)
                if math.isfinite(diverse_cat_prox) and math.isfinite(expected_cat_prox)
                else float("-inf")
            ),
        )

    compare_k_randominit = max(compare_ks)
    randominit_margin = float(assertion_cfg.get("randominit_validity_margin_k4", 0.1))
    diverse_validity = _get_summary_value(
        summary_df,
        "DiverseCF",
        compare_k_randominit,
        "validity",
    )
    randominit_validity = _get_summary_value(
        summary_df,
        "RandomInitCF",
        compare_k_randominit,
        "validity",
    )
    expected_validity = randominit_validity + randominit_margin
    passed = bool(
        math.isfinite(diverse_validity)
        and math.isfinite(expected_validity)
        and diverse_validity >= expected_validity
    )
    _append_check(
        checks=checks,
        name=f"diverse_validity_margin_vs_randominit_k_{compare_k_randominit}",
        relation=">=",
        actual=diverse_validity,
        expected=expected_validity,
        passed=passed,
        margin=(
            float(diverse_validity - expected_validity)
            if math.isfinite(diverse_validity) and math.isfinite(expected_validity)
            else float("-inf")
        ),
    )

    boundary_radius = float(assertion_cfg.get("boundary_compare_radius", 0.5))
    requested_boundary_compare_ks = tuple(
        int(value) for value in assertion_cfg.get("boundary_compare_ks", [4])
    )
    boundary_compare_ks = tuple(
        k for k in requested_boundary_compare_ks if k in set(available_ks)
    )
    boundary_f1_floor = float(assertion_cfg.get("boundary_f1_floor", 0.35))
    boundary_lime_ratio_floor = float(
        assertion_cfg.get("boundary_lime_ratio_floor", 0.7)
    )
    lime_f1 = _get_lime_value(lime_df, boundary_radius, "f1")

    for k in boundary_compare_ks:
        diverse_f1 = _get_boundary_value(
            boundary_df, "DiverseCF", k, boundary_radius, "f1"
        )
        _append_check(
            checks=checks,
            name=f"diverse_boundary_f1_floor_k_{k}",
            relation=">=",
            actual=diverse_f1,
            expected=boundary_f1_floor,
            passed=bool(math.isfinite(diverse_f1) and diverse_f1 >= boundary_f1_floor),
            margin=(
                float(diverse_f1 - boundary_f1_floor)
                if math.isfinite(diverse_f1)
                else float("-inf")
            ),
        )
        if math.isfinite(lime_f1):
            expected_f1 = lime_f1 * boundary_lime_ratio_floor
            _append_check(
                checks=checks,
                name=f"diverse_boundary_f1_vs_lime_k_{k}",
                relation=">=",
                actual=diverse_f1,
                expected=expected_f1,
                passed=bool(math.isfinite(diverse_f1) and diverse_f1 >= expected_f1),
                margin=(
                    float(diverse_f1 - expected_f1)
                    if math.isfinite(diverse_f1)
                    else float("-inf")
                ),
            )

    passed = all(bool(check["passed"]) for check in checks)
    report = {
        "passed": passed,
        "checks": checks,
        "num_passed": sum(1 for check in checks if bool(check["passed"])),
        "num_checks": len(checks),
    }
    if not passed:
        failed_checks = [
            f"{check['name']}: actual={check['actual']}, expected={check['relation']} {check['expected']}"
            for check in checks
            if not bool(check["passed"])
        ]
        raise AssertionError("Paper assertions failed:\n" + "\n".join(failed_checks))
    return report


def _score_assertion_report(report: dict[str, object] | None) -> tuple[int, float]:
    if report is None:
        return (-1, float("-inf"))
    checks = report.get("checks", [])
    if not isinstance(checks, list):
        return (-1, float("-inf"))
    num_passed = sum(1 for check in checks if bool(check.get("passed", False)))
    total_margin = 0.0
    for check in checks:
        margin = check.get("margin", 0.0)
        if isinstance(margin, (int, float)) and math.isfinite(float(margin)):
            total_margin += float(margin)
    return num_passed, total_margin


def _prepare_run_config(
    base_config: dict,
    run_name: str,
    run_dir: Path | None,
    device: str,
    model_overrides: dict | None = None,
    method_overrides: dict | None = None,
) -> dict:
    config = deepcopy(base_config)
    config["name"] = run_name
    config["model"]["device"] = device
    config["method"]["device"] = device
    if model_overrides:
        config["model"].update(deepcopy(model_overrides))
    if method_overrides:
        config["method"].update(deepcopy(method_overrides))
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        config["logger"]["path"] = str(run_dir / f"{run_name}.log")
    if "save_name" in config["model"]:
        config["model"]["save_name"] = run_name
    return config


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


def _run_candidate(
    config: dict,
    ks: tuple[int, ...],
    radii: tuple[float, ...],
    num_factuals: int,
    num_boundary_samples: int,
    lime_num_samples: int,
    save_results: bool,
    output_dir: Path | None,
    assert_paper: bool,
) -> dict[str, object]:
    device = _resolve_device()
    logger = setup_logger(
        level=config.get("logger", {}).get("level", "INFO"),
        path=config.get("logger", {}).get("path"),
        name=config.get("name", "dice_reproduce"),
    )
    set_cache_dir(config.get("caching", {}).get("path", "./cache/"))

    experiment = Experiment(config)
    trainset, testset = _materialize_datasets(experiment)
    logger.info("Train/test sizes: %d / %d", len(trainset), len(testset))

    target_model = experiment._target_model
    target_model.fit(trainset)
    accuracy = _compute_model_accuracy(target_model, testset)
    logger.info("Target model test accuracy: %.4f", accuracy)

    desired_class = config["method"]["desired_class"]
    factual_features = _select_factual_features(
        testset=testset,
        target_model=target_model,
        desired_class=desired_class,
        num_factuals=num_factuals,
        seed=int(config["method"].get("seed", 42)),
    )
    logger.info("Selected %d factuals for evaluation", factual_features.shape[0])

    schema_method = _build_method(config["method"], target_model)
    schema_method.fit(trainset)
    schema_method._trainset_reference = trainset

    lime_df = _evaluate_lime_proxy(
        factual_features=factual_features,
        target_model=target_model,
        schema_method=schema_method,
        num_boundary_samples=num_boundary_samples,
        radii=radii,
        lime_num_samples=lime_num_samples,
        seed=int(config["method"].get("seed", 42)),
    )

    summary_rows: list[dict[str, float | int | str]] = []
    boundary_rows: list[dict[str, float | int | str]] = []
    method_overrides = {
        "DiverseCF": {
            "algorithm": "DiverseCF",
            "diversity_weight": float(config["method"].get("diversity_weight", 1.0)),
        },
        "NoDiversityCF": {
            "algorithm": "DiverseCF",
            "diversity_weight": 0.0,
        },
        "RandomInitCF": {
            "algorithm": "RandomInitCF",
            "diversity_weight": float(config["method"].get("diversity_weight", 1.0)),
        },
    }

    for setting_name, overrides in method_overrides.items():
        logger.info("Evaluating setting: %s", setting_name)
        for requested_k in ks:
            logger.info("Generating counterfactuals for k=%d", requested_k)
            method_config = deepcopy(config["method"])
            method_config.update(overrides)
            method_config["num"] = requested_k
            dice_method = _build_method(method_config, target_model)
            dice_method.fit(trainset)
            dice_method._trainset_reference = trainset

            setting_summary, setting_boundary = _evaluate_setting(
                factual_features=factual_features,
                target_model=target_model,
                dice_method=dice_method,
                requested_k=requested_k,
                num_boundary_samples=num_boundary_samples,
                radii=radii,
                seed=int(config["method"].get("seed", 42)) + requested_k,
            )
            summary_rows.append(
                {
                    "setting": setting_name,
                    "k": int(requested_k),
                    "model_accuracy": accuracy,
                    **setting_summary,
                }
            )
            for boundary_record in setting_boundary:
                boundary_rows.append(
                    {
                        "setting": setting_name,
                        "k": int(requested_k),
                        **boundary_record,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    boundary_df = pd.DataFrame(boundary_rows)
    report = None
    if assert_paper:
        report = _assert_paper_results(
            summary_df=summary_df,
            boundary_df=boundary_df,
            lime_df=lime_df,
            config=config,
        )

    if save_results and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / "summary.csv", index=False)
        boundary_df.to_csv(output_dir / "boundary.csv", index=False)
        lime_df.to_csv(output_dir / "lime_proxy.csv", index=False)
        metadata = {
            "config_name": config["name"],
            "device": device,
            "num_factuals": num_factuals,
            "num_boundary_samples": num_boundary_samples,
            "lime_num_samples": lime_num_samples,
            "ks": list(ks),
            "radii": list(radii),
            "accuracy": accuracy,
        }
        _write_json(output_dir / "metadata.json", metadata)
        if report is not None:
            _write_json(output_dir / "assertion_report.json", report)

    return {
        "summary_df": summary_df,
        "boundary_df": boundary_df,
        "lime_df": lime_df,
        "assertion_report": report,
        "accuracy": accuracy,
    }


def _candidate_sequence(base_config: dict) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = [
        {
            "label": "base",
            "model_overrides": {},
            "method_overrides": {},
        },
        {
            "label": "model_epochs_200",
            "model_overrides": {"epochs": 200},
            "method_overrides": {},
        },
        {
            "label": "model_lr_0p005",
            "model_overrides": {"learning_rate": 0.005},
            "method_overrides": {},
        },
        {
            "label": "model_batch_25",
            "model_overrides": {"batch_size": 25},
            "method_overrides": {},
        },
        {
            "label": "method_prox_0p4_div_1p2",
            "model_overrides": {},
            "method_overrides": {
                "proximity_weight": 0.4,
                "diversity_weight": 1.2,
            },
        },
        {
            "label": "method_prox_0p6_div_1p0",
            "model_overrides": {},
            "method_overrides": {
                "proximity_weight": 0.6,
                "diversity_weight": 1.0,
            },
        },
        {
            "label": "method_cat_0p2",
            "model_overrides": {},
            "method_overrides": {"categorical_penalty": 0.2},
        },
        {
            "label": "method_no_sparse",
            "model_overrides": {},
            "method_overrides": {"posthoc_sparsity_param": 0.0},
        },
    ]
    max_runs = int(base_config.get("autorepro", {}).get("max_runs", len(candidates)))
    return candidates[:max_runs]


def run_reproduction(
    config_path: Path = DEFAULT_CONFIG_PATH,
    ks: tuple[int, ...] = DEFAULT_MANUAL_KS,
    radii: tuple[float, ...] = DEFAULT_MANUAL_RADII,
    num_factuals: int = 500,
    num_boundary_samples: int = 1000,
    save_results: bool = True,
    assert_paper: bool = False,
    run_dir: Path | None = None,
) -> dict[str, object]:
    config = _load_config(config_path)
    device = _resolve_device()
    config = _prepare_run_config(
        base_config=config,
        run_name=config.get("name", "dice_reproduce"),
        run_dir=run_dir,
        device=device,
    )
    lime_num_samples = int(config.get("assertions", {}).get("lime_num_samples", 1000))
    return _run_candidate(
        config=config,
        ks=ks,
        radii=radii,
        num_factuals=num_factuals,
        num_boundary_samples=num_boundary_samples,
        lime_num_samples=lime_num_samples,
        save_results=save_results,
        output_dir=run_dir,
        assert_paper=assert_paper,
    )


def run_autorepro(
    config_path: Path = DEFAULT_CONFIG_PATH,
    run_tag: str | None = None,
    save_results: bool = True,
) -> dict[str, object]:
    base_config = _load_config(config_path)
    autorepro_cfg = deepcopy(base_config.get("autorepro", {}))
    assertion_cfg = deepcopy(base_config.get("assertions", {}))
    if run_tag is None:
        run_tag = f"dice_{strftime('%Y%m%d_%H%M%S')}"

    root_dir = PROJECT_ROOT / "results" / "dice" / "autorepro" / run_tag
    loop_log_path = root_dir / "loop.jsonl"
    screening_num_factuals = int(autorepro_cfg.get("screening_num_factuals", 4))
    screening_boundary_samples = int(
        autorepro_cfg.get("screening_boundary_samples", 50)
    )
    acceptance_num_factuals = int(autorepro_cfg.get("acceptance_num_factuals", 8))
    acceptance_boundary_samples = int(
        autorepro_cfg.get("acceptance_boundary_samples", 100)
    )
    compare_ks = tuple(int(value) for value in assertion_cfg.get("compare_ks", [2, 4]))
    compare_radius = float(assertion_cfg.get("boundary_compare_radius", 0.5))
    radii = (compare_radius,)
    screening_method_overrides = deepcopy(
        autorepro_cfg.get("screening_method_overrides", {})
    )
    acceptance_method_overrides = deepcopy(
        autorepro_cfg.get("acceptance_method_overrides", {})
    )

    best_result: dict[str, object] | None = None
    best_score = (-1, float("-inf"))
    device = _resolve_device()

    for index, candidate in enumerate(_candidate_sequence(base_config), start=1):
        label = str(candidate["label"])
        run_name = f"{base_config.get('name', 'dice')}_{index:02d}_{label}"
        candidate_dir = root_dir / run_name
        config = _prepare_run_config(
            base_config=base_config,
            run_name=run_name,
            run_dir=candidate_dir,
            device=device,
            model_overrides=candidate.get("model_overrides"),
            method_overrides=candidate.get("method_overrides"),
        )

        entry = {
            "run_name": run_name,
            "phase": "screening",
            "num_factuals": screening_num_factuals,
            "num_boundary_samples": screening_boundary_samples,
            "ks": list(compare_ks),
            "radii": list(radii),
            "status": "started",
        }
        _append_jsonl(loop_log_path, entry)

        screening_exception = None
        screening_result = None
        try:
            screening_config = deepcopy(config)
            screening_config["method"].update(screening_method_overrides)
            screening_result = _run_candidate(
                config=screening_config,
                ks=compare_ks,
                radii=radii,
                num_factuals=screening_num_factuals,
                num_boundary_samples=screening_boundary_samples,
                lime_num_samples=int(
                    screening_config.get("assertions", {}).get("lime_num_samples", 1000)
                ),
                save_results=save_results,
                output_dir=candidate_dir / "screening",
                assert_paper=True,
            )
            screening_passed = True
        except Exception as error:  # noqa: BLE001
            screening_exception = error
            screening_passed = False

        screening_report = (
            None
            if screening_result is None
            else screening_result.get("assertion_report")
        )
        screening_score = _score_assertion_report(screening_report)
        if screening_score > best_score:
            best_score = screening_score
            best_result = {
                "run_name": run_name,
                "phase": "screening",
                "result": screening_result,
                "error": (
                    None if screening_exception is None else str(screening_exception)
                ),
            }

        _append_jsonl(
            loop_log_path,
            {
                "run_name": run_name,
                "phase": "screening",
                "status": "passed" if screening_passed else "failed",
                "score": screening_score,
                "error": (
                    None if screening_exception is None else str(screening_exception)
                ),
            },
        )
        if not screening_passed:
            continue

        _append_jsonl(
            loop_log_path,
            {
                "run_name": run_name,
                "phase": "acceptance",
                "num_factuals": acceptance_num_factuals,
                "num_boundary_samples": acceptance_boundary_samples,
                "ks": list(compare_ks),
                "radii": list(radii),
                "status": "started",
            },
        )

        acceptance_exception = None
        acceptance_result = None
        try:
            acceptance_config = deepcopy(config)
            acceptance_config["method"].update(acceptance_method_overrides)
            acceptance_result = _run_candidate(
                config=acceptance_config,
                ks=compare_ks,
                radii=radii,
                num_factuals=acceptance_num_factuals,
                num_boundary_samples=acceptance_boundary_samples,
                lime_num_samples=int(
                    acceptance_config.get("assertions", {}).get(
                        "lime_num_samples", 1000
                    )
                ),
                save_results=save_results,
                output_dir=candidate_dir / "acceptance",
                assert_paper=True,
            )
            acceptance_passed = True
        except Exception as error:  # noqa: BLE001
            acceptance_exception = error
            acceptance_passed = False

        acceptance_report = (
            None
            if acceptance_result is None
            else acceptance_result.get("assertion_report")
        )
        acceptance_score = _score_assertion_report(acceptance_report)
        if acceptance_score > best_score:
            best_score = acceptance_score
            best_result = {
                "run_name": run_name,
                "phase": "acceptance",
                "result": acceptance_result,
                "error": (
                    None if acceptance_exception is None else str(acceptance_exception)
                ),
            }

        _append_jsonl(
            loop_log_path,
            {
                "run_name": run_name,
                "phase": "acceptance",
                "status": "passed" if acceptance_passed else "failed",
                "score": acceptance_score,
                "error": (
                    None if acceptance_exception is None else str(acceptance_exception)
                ),
            },
        )

        if acceptance_passed:
            best_config_payload = {
                "run_name": run_name,
                "model": config["model"],
                "method": acceptance_config["method"],
                "ks": list(compare_ks),
                "radii": list(radii),
                "screening_num_factuals": screening_num_factuals,
                "acceptance_num_factuals": acceptance_num_factuals,
                "acceptance_num_boundary_samples": acceptance_boundary_samples,
            }
            _write_json(root_dir / "best_config.json", best_config_payload)
            return {
                "run_name": run_name,
                "root_dir": root_dir,
                **acceptance_result,
            }

    failure_payload = {
        "best_run": None if best_result is None else best_result["run_name"],
        "best_phase": None if best_result is None else best_result["phase"],
        "best_score": list(best_score),
        "error": None if best_result is None else best_result["error"],
    }
    _write_json(root_dir / "best_config.json", failure_payload)
    raise AssertionError(
        "Autorepro did not find a passing configuration. "
        f"Best candidate: {failure_payload}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--num-factuals", type=int, default=None)
    parser.add_argument("--num-boundary-samples", type=int, default=None)
    parser.add_argument("--ks", default=None)
    parser.add_argument("--radii", default=None)
    parser.add_argument("--assert-paper", action="store_true")
    parser.add_argument("--autorepro", action="store_true")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if args.autorepro:
        result = run_autorepro(
            config_path=config_path,
            run_tag=args.run_tag,
            save_results=not args.no_save,
        )
        print(result["summary_df"].to_string(index=False))
        print()
        print(result["boundary_df"].to_string(index=False))
        print()
        print(result["lime_df"].to_string(index=False))
        return

    ks = (
        tuple(int(value) for value in args.ks.split(",") if value.strip())
        if args.ks
        else DEFAULT_MANUAL_KS
    )
    radii = (
        tuple(float(value) for value in args.radii.split(",") if value.strip())
        if args.radii
        else DEFAULT_MANUAL_RADII
    )
    num_factuals = 500 if args.num_factuals is None else int(args.num_factuals)
    num_boundary_samples = (
        1000 if args.num_boundary_samples is None else int(args.num_boundary_samples)
    )

    result = run_reproduction(
        config_path=config_path,
        ks=ks,
        radii=radii,
        num_factuals=num_factuals,
        num_boundary_samples=num_boundary_samples,
        save_results=not args.no_save,
        assert_paper=args.assert_paper,
        run_dir=None,
    )
    print(result["summary_df"].to_string(index=False))
    print()
    print(result["boundary_df"].to_string(index=False))
    print()
    print(result["lime_df"].to_string(index=False))


if __name__ == "__main__":
    main()
