from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("TEMP", "/tmp")
os.environ.setdefault("TMP", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import dataset  # noqa: F401
import method  # noqa: F401
import model  # noqa: F401
import preprocess  # noqa: F401
from method.diverse_dist.support import DiverseDistModelAdapter, DiverseDistTrace
from utils.logger import setup_logger
from utils.registry import get_registry

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yml")
REFERENCE_RESULTS_ROOT = (
    PROJECT_ROOT
    / "reference"
    / "robust_counterfactuals_aaai24"
    / "results"
    / "final_results"
)


@dataclass(frozen=True)
class Condition:
    norm: int
    opt: bool


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _build_object(registry_type: str, config: dict, **kwargs):
    registry = get_registry(registry_type)
    object_config = deepcopy(config)
    name = object_config.pop("name")
    cls = registry[name]
    return cls(**object_config, **kwargs)


def _materialize_dataset(config: dict):
    dataset_obj = _build_object("dataset", config["dataset"])
    for preprocess_config in config.get("preprocess", []):
        preprocess_step = _build_object("preprocess", preprocess_config)
        dataset_obj = preprocess_step.transform(dataset_obj)
    finalize = get_registry("preprocess")["finalize"]()
    return finalize.transform(dataset_obj)


def _clone_with_df(dataset_obj, df: pd.DataFrame, flag: str):
    cloned = dataset_obj.clone()
    cloned.update(flag, True, df=df.copy(deep=True))
    cloned.freeze()
    return cloned


def _split_dataset(dataset_obj, split_seed: int, test_size: float):
    feature_df = dataset_obj.get(target=False)
    target_df = dataset_obj.get(target=True)
    target_series = target_df.iloc[:, 0]
    combined = pd.concat([feature_df, target_df], axis=1)

    train_df, test_df = train_test_split(
        combined,
        test_size=float(test_size),
        random_state=int(split_seed),
        shuffle=True,
        stratify=target_series,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return (
        _clone_with_df(dataset_obj, train_df, "trainset"),
        _clone_with_df(dataset_obj, test_df, "testset"),
    )


def _apply_noise(
    x: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    noise_percent: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    variance = float(noise_percent) * (train_max - train_min) / 100.0
    covariance = np.diag(variance.astype(np.float64, copy=False))
    noisy = rng.multivariate_normal(x.astype(np.float64), covariance)
    return noisy.clip(min=train_min, max=train_max).astype(np.float32, copy=False)


def _compute_distance(x: np.ndarray, y: np.ndarray, norm: int) -> float:
    if int(norm) == 1:
        return float(np.sum(np.abs(x - y)))
    if int(norm) == 2:
        return float(np.linalg.norm(x - y, ord=2))
    raise ValueError("norm must be 1 or 2")


def _compute_k_distance(
    factual: np.ndarray,
    counterfactuals: list[np.ndarray],
    norm: int,
) -> float:
    return float(
        sum(_compute_distance(factual, counterfactual, norm) for counterfactual in counterfactuals)
        / len(counterfactuals)
    )


def _compute_k_diversity(counterfactuals: list[np.ndarray], norm: int) -> float:
    if len(counterfactuals) < 2:
        return 0.0

    values: list[float] = []
    for left_index in range(len(counterfactuals)):
        for right_index in range(left_index + 1, len(counterfactuals)):
            values.append(
                _compute_distance(
                    counterfactuals[left_index],
                    counterfactuals[right_index],
                    norm,
                )
            )
    return float(sum(values) / len(values))


def _compute_set_distance(
    counterfactuals: list[np.ndarray],
    noisy_counterfactuals: list[np.ndarray],
    norm: int,
) -> float:
    forward = [
        min(_compute_distance(counterfactual, noisy_counterfactual, norm) for noisy_counterfactual in noisy_counterfactuals)
        for counterfactual in counterfactuals
    ]
    backward = [
        min(_compute_distance(noisy_counterfactual, counterfactual, norm) for counterfactual in counterfactuals)
        for noisy_counterfactual in noisy_counterfactuals
    ]
    return float(
        sum(forward) / (2 * len(counterfactuals))
        + sum(backward) / (2 * len(noisy_counterfactuals))
    )


def _compute_set_distance_max(
    counterfactuals: list[np.ndarray],
    noisy_counterfactuals: list[np.ndarray],
    norm: int,
) -> float:
    forward = [
        min(_compute_distance(counterfactual, noisy_counterfactual, norm) for noisy_counterfactual in noisy_counterfactuals)
        for counterfactual in counterfactuals
    ]
    backward = [
        min(_compute_distance(noisy_counterfactual, counterfactual, norm) for counterfactual in counterfactuals)
        for noisy_counterfactual in noisy_counterfactuals
    ]
    return float(0.5 * (max(forward) + max(backward)))


def _extract_counterfactual_set(method_obj) -> list[np.ndarray]:
    traces = getattr(method_obj, "_last_explanation_sets", [])
    if len(traces) != 1:
        return []

    trace = traces[0]
    if not isinstance(trace, DiverseDistTrace):
        return []

    return [np.asarray(counterfactual, dtype=np.float32).reshape(-1) for counterfactual in trace.counterfactuals]


def _is_valid_counterfactual_set(
    adapter: DiverseDistModelAdapter,
    factual: np.ndarray,
    counterfactuals: list[np.ndarray],
) -> bool:
    if not counterfactuals:
        return False

    factual_label = int(adapter.predict_label_indices(factual.reshape(1, -1))[0])
    counterfactual_labels = adapter.predict_label_indices(
        np.asarray(counterfactuals, dtype=np.float32)
    )
    return bool(np.all(counterfactual_labels != factual_label))


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def _format_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "nan"
    return f"{mean:.2f} ± {std:.2f}"


def _parse_mean_std_line(line: str) -> tuple[float, float]:
    value = line.split(":", maxsplit=1)[1].strip().rstrip(".")
    mean_text, std_text = [item.strip() for item in value.split(",", maxsplit=1)]
    return float(mean_text), float(std_text)


def _parse_reference_result(path: Path) -> dict[str, float]:
    targets: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Valid cfx:"):
            targets["validity"] = float(line.split(":", maxsplit=1)[1].strip().rstrip("."))
        elif line.startswith("Avg k-distance:"):
            mean, std = _parse_mean_std_line(line)
            targets["k_distance_mean"] = mean
            targets["k_distance_std"] = std
        elif line.startswith("Avg k-diversity:"):
            mean, std = _parse_mean_std_line(line)
            targets["k_diversity_mean"] = mean
            targets["k_diversity_std"] = std
        elif line.startswith("Avg match-distance:"):
            mean, std = _parse_mean_std_line(line)
            targets["set_distance_sum_mean"] = mean
            targets["set_distance_sum_std"] = std
        elif line.startswith("Avg match-b-distance:"):
            mean, std = _parse_mean_std_line(line)
            targets["set_distance_max_mean"] = mean
            targets["set_distance_max_std"] = std
        elif line.startswith("Avg time:"):
            mean, std = _parse_mean_std_line(line)
            targets["time_mean"] = mean
            targets["time_std"] = std
    return targets


def _load_paper_targets(config: dict) -> dict[Condition, dict[str, float]]:
    targets: dict[Condition, dict[str, float]] = {}
    for opt in config["reproduction"]["opt_settings"]:
        target_group = "opt_true" if bool(opt) else "opt_false"
        for norm in config["reproduction"]["norms"]:
            targets[Condition(norm=int(norm), opt=bool(opt))] = deepcopy(
                config["reproduction"]["paper_targets"][target_group][f"l{int(norm)}"]
            )
    return targets


def _load_repository_targets() -> dict[Condition, dict[str, float]]:
    targets: dict[Condition, dict[str, float]] = {}
    for condition in (
        Condition(norm=1, opt=True),
        Condition(norm=2, opt=True),
        Condition(norm=1, opt=False),
        Condition(norm=2, opt=False),
    ):
        if condition.opt:
            relative_path = (
                "ours_angle_based_min"
                f"/ours_diabetes_norm{condition.norm}_beta0.5_gamma0.1.txt"
            )
        else:
            relative_path = (
                "ours_angle_based_nomin"
                f"/ours_diabetes_norm{condition.norm}_beta0.5.txt"
            )

        path = REFERENCE_RESULTS_ROOT / relative_path
        if path.exists():
            targets[condition] = _parse_reference_result(path)
    return targets


def _evaluate_condition(
    config: dict,
    trainset,
    testset,
    model_obj,
    feature_names: list[str],
    train_min: np.ndarray,
    train_max: np.ndarray,
    condition: Condition,
    logger,
) -> dict[str, float | int]:
    method_config = deepcopy(config["method"])
    method_config["norm"] = int(condition.norm)
    method_config["opt"] = bool(condition.opt)
    method_obj = _build_object("method", method_config, target_model=model_obj)
    method_obj.fit(trainset)

    adapter = DiverseDistModelAdapter(model_obj, feature_names)
    reproduction_config = config["reproduction"]
    start_index = int(reproduction_config["start_index"])
    num_inputs = int(reproduction_config["num_inputs"])
    repeat_times = int(reproduction_config["repeat_times"])
    noise_percent = float(reproduction_config["noise_percent"])
    noise_seed = int(reproduction_config["noise_seed"])
    show_progress = bool(reproduction_config.get("show_progress", True))

    test_features = testset.get(target=False).reset_index(drop=True)
    selected = test_features.iloc[start_index : start_index + num_inputs].copy(deep=True)
    rng = np.random.RandomState(noise_seed)

    results: list[dict[str, float]] = []
    skipped_noise_label_mismatch = 0
    eligible_runs = 0

    iterator = selected.iterrows()
    if show_progress:
        iterator = tqdm(
            iterator,
            total=selected.shape[0],
            desc=f"norm{condition.norm}-opt{int(condition.opt)}",
        )

    for _, factual_row in iterator:
        factual = factual_row.to_numpy(dtype=np.float32, copy=True)
        factual_label = int(adapter.predict_label_indices(factual.reshape(1, -1))[0])
        factual_df = pd.DataFrame([factual], columns=feature_names)

        for _ in range(repeat_times):
            factual_noisy = _apply_noise(
                x=factual,
                train_min=train_min,
                train_max=train_max,
                noise_percent=noise_percent,
                rng=rng,
            )
            noisy_label = int(adapter.predict_label_indices(factual_noisy.reshape(1, -1))[0])
            if noisy_label != factual_label:
                skipped_noise_label_mismatch += 1
                continue

            eligible_runs += 1
            factual_noisy_df = pd.DataFrame([factual_noisy], columns=feature_names)

            start_time = time.perf_counter()
            method_obj.get_counterfactuals(factual_df)
            runtime = time.perf_counter() - start_time
            counterfactuals = _extract_counterfactual_set(method_obj)

            method_obj.get_counterfactuals(factual_noisy_df)
            noisy_counterfactuals = _extract_counterfactual_set(method_obj)

            if not counterfactuals or not noisy_counterfactuals:
                continue

            if not _is_valid_counterfactual_set(adapter, factual, counterfactuals):
                continue
            if not _is_valid_counterfactual_set(adapter, factual_noisy, noisy_counterfactuals):
                continue

            results.append(
                {
                    "k_distance": _compute_k_distance(factual, counterfactuals, condition.norm),
                    "k_diversity": _compute_k_diversity(counterfactuals, condition.norm),
                    "set_distance_sum": _compute_set_distance(
                        counterfactuals,
                        noisy_counterfactuals,
                        condition.norm,
                    ),
                    "set_distance_max": _compute_set_distance_max(
                        counterfactuals,
                        noisy_counterfactuals,
                        condition.norm,
                    ),
                    "time": float(runtime),
                }
            )

    logger.warning(
        "Completed norm=%s opt=%s with %s successful runs, %s eligible runs, %s skipped noisy-label mismatches",
        condition.norm,
        condition.opt,
        len(results),
        eligible_runs,
        skipped_noise_label_mismatch,
    )

    validity = (
        100.0 * float(len(results)) / float(eligible_runs)
        if eligible_runs > 0
        else float("nan")
    )
    k_distance_mean, k_distance_std = _mean_std([result["k_distance"] for result in results])
    k_diversity_mean, k_diversity_std = _mean_std([result["k_diversity"] for result in results])
    set_distance_sum_mean, set_distance_sum_std = _mean_std(
        [result["set_distance_sum"] for result in results]
    )
    set_distance_max_mean, set_distance_max_std = _mean_std(
        [result["set_distance_max"] for result in results]
    )
    time_mean, time_std = _mean_std([result["time"] for result in results])

    return {
        "validity": validity,
        "k_distance_mean": k_distance_mean,
        "k_distance_std": k_distance_std,
        "k_diversity_mean": k_diversity_mean,
        "k_diversity_std": k_diversity_std,
        "set_distance_sum_mean": set_distance_sum_mean,
        "set_distance_sum_std": set_distance_sum_std,
        "set_distance_max_mean": set_distance_max_mean,
        "set_distance_max_std": set_distance_max_std,
        "time_mean": time_mean,
        "time_std": time_std,
        "successful_runs": len(results),
        "eligible_runs": eligible_runs,
        "skipped_noise_label_mismatch": skipped_noise_label_mismatch,
    }


def _build_comparison_row(
    condition: Condition,
    observed: dict[str, float | int],
    repository_target: dict[str, float] | None,
    paper_target: dict[str, float] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "opt": bool(condition.opt),
        "norm": int(condition.norm),
        "validity": f"{observed['validity']:.1f}%",
        "k_distance": _format_mean_std(
            float(observed["k_distance_mean"]),
            float(observed["k_distance_std"]),
        ),
        "k_diversity": _format_mean_std(
            float(observed["k_diversity_mean"]),
            float(observed["k_diversity_std"]),
        ),
        "set_d_sum": _format_mean_std(
            float(observed["set_distance_sum_mean"]),
            float(observed["set_distance_sum_std"]),
        ),
        "set_d_max": _format_mean_std(
            float(observed["set_distance_max_mean"]),
            float(observed["set_distance_max_std"]),
        ),
        "time": _format_mean_std(
            float(observed["time_mean"]),
            float(observed["time_std"]),
        ),
        "successful": int(observed["successful_runs"]),
        "eligible": int(observed["eligible_runs"]),
        "skipped_noise": int(observed["skipped_noise_label_mismatch"]),
    }

    if repository_target is not None:
        row["repo_validity"] = f"{repository_target['validity']:.1f}%"
        row["repo_k_distance"] = _format_mean_std(
            repository_target["k_distance_mean"],
            repository_target["k_distance_std"],
        )
        row["repo_k_diversity"] = _format_mean_std(
            repository_target["k_diversity_mean"],
            repository_target["k_diversity_std"],
        )
        row["repo_set_d_sum"] = _format_mean_std(
            repository_target["set_distance_sum_mean"],
            repository_target["set_distance_sum_std"],
        )
        row["repo_set_d_max"] = _format_mean_std(
            repository_target["set_distance_max_mean"],
            repository_target["set_distance_max_std"],
        )
        row["repo_time"] = _format_mean_std(
            repository_target["time_mean"],
            repository_target["time_std"],
        )

    if paper_target is not None:
        row["paper_validity"] = f"{paper_target['validity']:.1f}%"
        row["paper_k_distance"] = _format_mean_std(
            paper_target["k_distance_mean"],
            paper_target["k_distance_std"],
        )
        row["paper_k_diversity"] = _format_mean_std(
            paper_target["k_diversity_mean"],
            paper_target["k_diversity_std"],
        )
        row["paper_set_d_sum"] = _format_mean_std(
            paper_target["set_distance_sum_mean"],
            paper_target["set_distance_sum_std"],
        )
        row["paper_set_d_max"] = _format_mean_std(
            paper_target["set_distance_max_mean"],
            paper_target["set_distance_max_std"],
        )
        row["paper_time"] = _format_mean_std(
            paper_target["time_mean"],
            paper_target["time_std"],
        )

    return row


def _print_comparison(
    observed_results: dict[Condition, dict[str, float | int]],
    repository_targets: dict[Condition, dict[str, float]],
    paper_targets: dict[Condition, dict[str, float]],
) -> None:
    rows = []
    for condition, observed in observed_results.items():
        rows.append(
            _build_comparison_row(
                condition=condition,
                observed=observed,
                repository_target=repository_targets.get(condition),
                paper_target=paper_targets.get(condition),
            )
        )

    comparison = pd.DataFrame(rows)
    print("Observed metrics")
    print(comparison.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = _load_config(config_path)

    configured_device = str(config.get("model", {}).get("device", "cpu")).lower()
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested in config but is not available")

    logger = setup_logger(
        level=str(config.get("logger", {}).get("level", "WARNING")),
        path=None,
        name=config.get("name", "diverse_dist_reproduction"),
    )

    dataset_obj = _materialize_dataset(config)
    trainset, testset = _split_dataset(
        dataset_obj,
        split_seed=int(config["reproduction"]["split_seed"]),
        test_size=float(config["reproduction"]["test_size"]),
    )
    model_obj = _build_object("model", config["model"])
    model_obj.fit(trainset)

    train_features = trainset.get(target=False)
    train_array = train_features.to_numpy(dtype=np.float32)
    train_min = train_array.min(axis=0)
    train_max = train_array.max(axis=0)
    feature_names = list(train_features.columns)

    adapter = DiverseDistModelAdapter(model_obj, feature_names)
    test_features = testset.get(target=False)
    test_targets = testset.get(target=True).iloc[:, 0].to_numpy(dtype=np.int64)
    test_predictions = adapter.predict_label_indices(test_features)
    test_accuracy = float(np.mean(test_predictions == test_targets))

    logger.warning(
        "Diabetes reproduction on %s with test accuracy %.4f",
        configured_device,
        test_accuracy,
    )

    paper_targets = _load_paper_targets(config)
    repository_targets = _load_repository_targets()

    observed_results: dict[Condition, dict[str, float | int]] = {}
    for opt in config["reproduction"]["opt_settings"]:
        for norm in config["reproduction"]["norms"]:
            condition = Condition(norm=int(norm), opt=bool(opt))
            logger.warning(
                "Running diabetes reproduction for norm=%s opt=%s",
                condition.norm,
                condition.opt,
            )
            observed_results[condition] = _evaluate_condition(
                config=config,
                trainset=trainset,
                testset=testset,
                model_obj=model_obj,
                feature_names=feature_names,
                train_min=train_min,
                train_max=train_max,
                condition=condition,
                logger=logger,
            )

    print(f"Test accuracy: {test_accuracy:.4f}")
    _print_comparison(
        observed_results=observed_results,
        repository_targets=repository_targets,
        paper_targets=paper_targets,
    )


if __name__ == "__main__":
    main()
