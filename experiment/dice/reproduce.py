from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
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
DEFAULT_KS = (1, 2, 4, 6, 8, 10)
DEFAULT_RADII = (0.5, 1.0, 2.0)


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


def _boundary_metrics(
    counterfactuals: pd.DataFrame,
    factual: np.ndarray,
    target_model,
    desired_index: int,
    feature_names: list[str],
    num_samples: int,
    radii: tuple[float, ...],
    continuous_indices: tuple[int, ...],
    mads: dict[int, float],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    categorical_groups,
    binary_feature_value_map: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> dict[float, tuple[list[int], list[int]]]:
    if counterfactuals.empty:
        return {radius: ([], []) for radius in radii}

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

    results: dict[float, tuple[list[int], list[int]]] = {}
    for radius in radii:
        boundary_points = _sample_boundary_points(
            factual=factual,
            num_samples=num_samples,
            radius_multiplier=radius,
            continuous_indices=continuous_indices,
            mads=mads,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            categorical_groups=categorical_groups,
            binary_feature_value_map=binary_feature_value_map,
            rng=rng,
        )
        target_prediction = target_model.get_prediction(
            pd.DataFrame(boundary_points, columns=feature_names),
            proba=False,
        )
        y_true = target_prediction.argmax(dim=1).detach().cpu().numpy().tolist()
        y_pred = knn.predict(boundary_points).tolist()
        results[radius] = (y_true, y_pred)
    return results


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

        boundary_metrics = _boundary_metrics(
            counterfactuals=counterfactuals,
            factual=factual,
            target_model=target_model,
            desired_index=desired_index,
            feature_names=feature_names,
            num_samples=num_boundary_samples,
            radii=radii,
            continuous_indices=continuous_indices,
            mads=mads,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            categorical_groups=categorical_groups,
            binary_feature_value_map=dice_method._binary_feature_value_map,
            rng=rng,
        )
        for radius, (y_true, y_pred) in boundary_metrics.items():
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


def run_reproduction(
    config_path: Path = DEFAULT_CONFIG_PATH,
    ks: tuple[int, ...] = DEFAULT_KS,
    radii: tuple[float, ...] = DEFAULT_RADII,
    num_factuals: int = 500,
    num_boundary_samples: int = 1000,
    save_results: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _load_config(config_path)
    device = _resolve_device()
    config["model"]["device"] = device
    config["method"]["device"] = device

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

    if save_results:
        output_dir = PROJECT_ROOT / "results" / "dice"
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "compas_summary.csv"
        boundary_path = output_dir / "compas_boundary.csv"
        metadata_path = output_dir / "compas_metadata.json"
        summary_df.to_csv(summary_path, index=False)
        boundary_df.to_csv(boundary_path, index=False)
        metadata_path.write_text(
            json.dumps(
                {
                    "config_path": str(config_path),
                    "device": device,
                    "num_factuals": num_factuals,
                    "num_boundary_samples": num_boundary_samples,
                    "ks": list(ks),
                    "radii": list(radii),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Saved summary to %s", summary_path)
        logger.info("Saved boundary metrics to %s", boundary_path)

    return summary_df, boundary_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--num-factuals", type=int, default=500)
    parser.add_argument("--num-boundary-samples", type=int, default=1000)
    parser.add_argument("--ks", default="1,2,4,6,8,10")
    parser.add_argument("--radii", default="0.5,1,2")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    ks = tuple(int(value) for value in args.ks.split(",") if value.strip())
    radii = tuple(float(value) for value in args.radii.split(",") if value.strip())

    summary_df, boundary_df = run_reproduction(
        config_path=Path(args.config),
        ks=ks,
        radii=radii,
        num_factuals=int(args.num_factuals),
        num_boundary_samples=int(args.num_boundary_samples),
        save_results=not args.no_save,
    )
    print(summary_df.to_string(index=False))
    print()
    print(boundary_df.to_string(index=False))


if __name__ == "__main__":
    main()
