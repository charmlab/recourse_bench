from __future__ import annotations

import argparse
import math
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
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

import dataset  # noqa: F401
import method  # noqa: F401
import model  # noqa: F401
import preprocess  # noqa: F401
from evaluation.evaluation_utils import resolve_evaluation_inputs
from evaluation.validity import ValidityEvaluation
from preprocess.common import FinalizePreProcess
from utils.caching import set_cache_dir
from utils.logger import setup_logger
from utils.registry import get_registry

DEFAULT_CONFIG_PATH = Path(__file__).with_name(
    "credit_cchvae_sklearn_logistic_regression_cchvae_reproduce.yaml"
)
TRAIN_SAMPLE_LIMIT: int | None = None
NCOUNTERFACTUALS: int | None = None
LOF_NEIGHBORS = [5, 10, 20, 50]
CONNECTEDNESS_EPSILONS = [10, 20, 30, 40, 50]
DBSCAN_MIN_SAMPLES = 5
OUTLIER_LIMIT = 0.2
CONNECTEDNESS_EPS = 30
CONNECTEDNESS_LIMIT = 0.5


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_dataset(config: dict):
    cfg = deepcopy(config["dataset"])
    name = cfg.pop("name")
    return get_registry("dataset")[name](**cfg)


def _build_preprocess(config: dict) -> list:
    preprocess = []
    registry = get_registry("preprocess")
    for item in config.get("preprocess", []):
        step_cfg = deepcopy(item)
        name = step_cfg.pop("name")
        preprocess.append(registry[name](**step_cfg))
    if not preprocess or not isinstance(preprocess[-1], FinalizePreProcess):
        preprocess.append(FinalizePreProcess())
    return preprocess


def _build_model(config: dict):
    cfg = deepcopy(config["model"])
    name = cfg.pop("name")
    return get_registry("model")[name](**cfg)


def _build_method(config: dict, target_model):
    cfg = deepcopy(config["method"])
    name = cfg.pop("name")
    return get_registry("method")[name](target_model=target_model, **cfg)


def _resolve_train_test(datasets: list):
    trainsets = [dataset for dataset in datasets if getattr(dataset, "trainset", False)]
    testsets = [dataset for dataset in datasets if getattr(dataset, "testset", False)]
    if len(trainsets) != 1 or len(testsets) != 1:
        raise ValueError("Expected exactly one trainset and one testset")
    return trainsets[0], testsets[0]


def _materialize_datasets(dataset, preprocess_steps: list):
    datasets = [dataset]
    for preprocess_step in preprocess_steps:
        next_datasets = []
        for current_dataset in datasets:
            transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                next_datasets.append(transformed)
        datasets = next_datasets
    return _resolve_train_test(datasets)


def _target_indices(dataset, class_to_index: dict[int | str, int]) -> np.ndarray:
    target = dataset.get(target=True).iloc[:, 0]
    output = []
    for value in target.tolist():
        if isinstance(value, float) and float(value).is_integer():
            value = int(value)
        output.append(class_to_index[value])
    return np.asarray(output, dtype=np.int64)


def _subset_dataset(dataset, indices: pd.Index, flag_name: str) -> object:
    feature_df = dataset.get(target=False)
    target_df = dataset.get(target=True)
    subset_df = pd.concat(
        [feature_df.loc[indices], target_df.loc[indices]],
        axis=1,
    ).reindex(columns=dataset.ordered_features())
    subset = dataset.clone()
    subset.update(flag_name, True, df=subset_df)
    subset.freeze()
    return subset


def _maybe_limit_dataset(dataset, limit: int | None, flag_name: str):
    if limit is None or limit >= len(dataset):
        return dataset
    indices = dataset.get(target=False).index[: int(limit)]
    return _subset_dataset(dataset, indices, flag_name)


def _train_positive_reference(
    trainset, target_model, desired_class: int | str
) -> pd.DataFrame:
    class_to_index = target_model.get_class_to_index()
    desired_index = class_to_index[desired_class]
    predicted = (
        target_model.predict(trainset, batch_size=512).argmax(dim=1).cpu().numpy()
    )
    true_indices = _target_indices(trainset, class_to_index)
    mask = (predicted == desired_index) & (true_indices == desired_index)
    return trainset.get(target=False).loc[mask]


def _outlier_rates(
    train_positive_reference: pd.DataFrame,
    counterfactual_success: pd.DataFrame,
    neighbors: list[int],
) -> dict[str, float]:
    results = {}
    if counterfactual_success.empty:
        for neighbor in neighbors:
            results[f"outlier_rate_k_{neighbor}"] = float("nan")
        return results

    for neighbor in neighbors:
        model = LocalOutlierFactor(
            n_neighbors=neighbor,
            contamination=0.01,
            novelty=True,
        )
        model.fit(train_positive_reference.to_numpy(dtype="float32"))
        pred = model.predict(counterfactual_success.to_numpy(dtype="float32"))
        results[f"outlier_rate_k_{neighbor}"] = float(np.mean(pred == -1))
    return results


def _connectedness_rates(
    train_positive_reference: pd.DataFrame,
    counterfactual_success: pd.DataFrame,
    epsilons: list[int | float],
    min_samples: int,
) -> dict[str, float]:
    results = {}
    if counterfactual_success.empty:
        for epsilon in epsilons:
            results[f"not_connected_rate_eps_{epsilon}"] = float("nan")
        return results

    reference = train_positive_reference.to_numpy(dtype="float32")
    counter = counterfactual_success.to_numpy(dtype="float32")
    for epsilon in epsilons:
        not_connected = []
        for row in counter:
            density_control = np.r_[reference, row.reshape(1, -1)]
            labels = (
                DBSCAN(eps=float(epsilon), min_samples=min_samples)
                .fit(density_control)
                .labels_
            )
            not_connected.append(labels[-1] == -1)
        results[f"not_connected_rate_eps_{epsilon}"] = float(np.mean(not_connected))
    return results


def _assert_results(results: pd.DataFrame, config: dict) -> None:
    row = results.iloc[0]

    if not math.isfinite(float(row["validity"])) or float(row["validity"]) <= 0.0:
        raise AssertionError("validity must be > 0")

    for neighbor in LOF_NEIGHBORS:
        value = float(row[f"outlier_rate_k_{neighbor}"])
        if math.isfinite(value) and value > float(OUTLIER_LIMIT):
            raise AssertionError(f"outlier_rate_k_{neighbor}={value:.4f} exceeds limit")

    connectedness_value = float(row[f"not_connected_rate_eps_{CONNECTEDNESS_EPS}"])
    if math.isfinite(connectedness_value) and connectedness_value > float(
        CONNECTEDNESS_LIMIT
    ):
        raise AssertionError(
            f"not_connected_rate_eps_{CONNECTEDNESS_EPS}={connectedness_value:.4f} exceeds limit"
        )


def run_reproduction(config_path: Path = DEFAULT_CONFIG_PATH) -> pd.DataFrame:
    config = _load_config(config_path)
    device = _resolve_device()
    if str(config["model"]["name"]).lower() == "sklearn_logistic_regression":
        device = "cpu"
    config["model"]["device"] = device
    config["method"]["device"] = device

    logger = setup_logger(
        level=config.get("logger", {}).get("level", "INFO"),
        path=config.get("logger", {}).get("path"),
        name=config.get("name", "cchvae_reproduce"),
    )
    set_cache_dir(config.get("caching", {}).get("path", "./cache/"))

    logger.info("Loaded config from %s", config_path)
    logger.info("Resolved device: %s", device)

    dataset = _build_dataset(config)
    preprocess_steps = _build_preprocess(config)
    trainset, testset = _materialize_datasets(dataset, preprocess_steps)
    logger.info("Train/test sizes: %d / %d", len(trainset), len(testset))

    effective_trainset = _maybe_limit_dataset(
        trainset, TRAIN_SAMPLE_LIMIT, "trainset"
    )
    if TRAIN_SAMPLE_LIMIT is not None:
        logger.info(
            "Using limited train subset for reproduction run: %d rows",
            len(effective_trainset),
        )

    target_model = _build_model(config)
    target_model.fit(effective_trainset)
    logger.info(
        "Target model trained; best_params=%s",
        getattr(getattr(target_model, "_grid_search", None), "best_params_", None),
    )

    method = _build_method(config, target_model)
    method.fit(effective_trainset)

    desired_class = config["method"]["desired_class"]
    class_to_index = target_model.get_class_to_index()
    predicted_test = (
        target_model.predict(testset, batch_size=512).argmax(dim=1).cpu().numpy()
    )
    if desired_class is None:
        search_mask = np.ones_like(predicted_test, dtype=bool)
    else:
        desired_index = class_to_index[desired_class]
        search_mask = predicted_test != desired_index
    candidate_indices = testset.get(target=False).index[search_mask]
    if NCOUNTERFACTUALS is None:
        selected_indices = candidate_indices
    else:
        selected_indices = candidate_indices[:NCOUNTERFACTUALS]
    factual_subset = _subset_dataset(testset, selected_indices, "testset")
    logger.info(
        "Selected %d / %d desired-class-mismatched test samples for search",
        len(factual_subset),
        int(search_mask.sum()),
    )

    counterfactuals = method.predict(
        factual_subset, batch_size=max(len(factual_subset), 1)
    )

    validity = ValidityEvaluation().evaluate(factual_subset, counterfactuals)
    factual_features, counterfactual_features, evaluation_mask, success_mask = (
        resolve_evaluation_inputs(factual_subset, counterfactuals)
    )
    selected_mask = evaluation_mask & success_mask
    factual_success = factual_features.loc[selected_mask.to_numpy()]
    counterfactual_success = counterfactual_features.loc[selected_mask.to_numpy()]
    train_positive_reference = _train_positive_reference(
        trainset=effective_trainset,
        target_model=target_model,
        desired_class=desired_class,
    )

    results = {
        "dataset": str(config["dataset"]["name"]),
        "device": device,
        "n_train": len(trainset),
        "n_test": len(testset),
        "n_search_candidates_total": int(search_mask.sum()),
        "n_counterfactuals_requested": len(selected_indices),
        "n_counterfactuals_evaluated": len(factual_subset),
        "n_counterfactuals_success": int(selected_mask.sum()),
        "validity": float(validity.iloc[0]["validity"]),
    }
    results.update(
        _outlier_rates(
            train_positive_reference=train_positive_reference,
            counterfactual_success=counterfactual_success,
            neighbors=LOF_NEIGHBORS,
        )
    )
    results.update(
        _connectedness_rates(
            train_positive_reference=train_positive_reference,
            counterfactual_success=counterfactual_success,
            epsilons=CONNECTEDNESS_EPSILONS,
            min_samples=DBSCAN_MIN_SAMPLES,
        )
    )

    metrics = pd.DataFrame([results])
    _assert_results(metrics, config)
    logger.info("Reproduction metrics:\n%s", metrics.to_string(index=False))
    print(metrics.to_string(index=False))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the reproduction YAML config",
    )
    args = parser.parse_args()
    run_reproduction(Path(args.path))


if __name__ == "__main__":
    main()
