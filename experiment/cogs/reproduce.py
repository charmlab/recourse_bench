from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tqdm import tqdm

from experiment import Experiment
from method.cogs.support import compute_ranges_numerical_features, gower_distance
from utils.registry import get_registry


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _apply_device(config: dict, device: str) -> dict:
    cfg = deepcopy(config)
    model_name = str(cfg["model"]["name"]).lower()
    resolved_device = "cpu" if model_name == "randomforest" else device
    cfg["model"]["device"] = resolved_device
    cfg["method"]["device"] = resolved_device
    return cfg


def _reference_style_split(dataset, split_preprocess) -> tuple[object, object]:
    df = dataset.snapshot()
    split = split_preprocess._split
    sample = split_preprocess._sample
    seed = split_preprocess._seed

    if isinstance(split, float):
        train_df = df.sample(frac=1.0 - split, random_state=seed)
        test_df = df.drop(train_df.index)
    else:
        test_df = df.sample(n=split, random_state=seed)
        train_df = df.drop(test_df.index)

    if sample is not None:
        test_df = test_df.sample(n=sample, random_state=seed).copy(deep=True)
    else:
        test_df = test_df.copy(deep=True)
    train_df = train_df.copy(deep=True)

    trainset = dataset
    testset = dataset.clone()
    trainset.update("trainset", True, df=train_df)
    testset.update("testset", True, df=test_df)
    return trainset, testset


def _materialize_datasets(experiment: Experiment) -> tuple[object, object]:
    datasets = [experiment._raw_dataset]
    for preprocess_step in experiment._preprocess:
        next_datasets = []
        for current_dataset in datasets:
            if preprocess_step.__class__.__name__ == "SplitPreProcess":
                transformed = _reference_style_split(current_dataset, preprocess_step)
            else:
                transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                next_datasets.append(transformed)
        datasets = next_datasets
    return experiment._resolve_train_test(datasets)


def _compute_model_metrics(model, testset) -> dict[str, float]:
    probabilities = model.predict_proba(testset).detach().cpu()
    predictions = probabilities.argmax(dim=1)

    target = testset.get(target=True).iloc[:, 0]
    class_to_index = model.get_class_to_index()
    encoded_target = torch.tensor(
        [class_to_index[int(value)] for value in target.astype(int).tolist()],
        dtype=torch.long,
    )
    accuracy = float((predictions == encoded_target).to(dtype=torch.float32).mean())
    return {
        "test_accuracy": accuracy,
        "num_test_rows": int(len(testset)),
    }


def _build_subset_dataset(dataset, index: pd.Index) -> object:
    subset = dataset.clone()
    target_column = dataset.target_column
    combined = pd.concat([dataset.get(target=False), dataset.get(target=True)], axis=1)
    filtered = combined.loc[index].copy(deep=True)
    filtered = filtered.loc[:, [*dataset.get(target=False).columns, target_column]]
    subset.update("testset", True, df=filtered)
    subset.freeze()
    return subset


def _select_factuals(
    model,
    testset,
    desired_class: int | str,
    sample_seed: int,
    num_factuals: int | None,
):
    class_to_index = model.get_class_to_index()
    desired_index = int(class_to_index[desired_class])
    predictions = model.predict(testset).argmax(dim=1).detach().cpu().numpy()
    features = testset.get(target=False)

    candidate_mask = predictions != desired_index
    candidate_index = features.index[candidate_mask]
    if len(candidate_index) == 0:
        raise RuntimeError("No candidate factuals found outside the desired class")

    if num_factuals is None:
        sampled_index = candidate_index
        sample_size = len(candidate_index)
    else:
        sample_size = min(int(num_factuals), len(candidate_index))
        rng = np.random.RandomState(int(sample_seed))
        sampled_positions = rng.choice(
            len(candidate_index), size=sample_size, replace=False
        )
        sampled_index = candidate_index[np.sort(sampled_positions)]
    factuals = _build_subset_dataset(testset, sampled_index)
    return factuals, {
        "candidate_pool_size": int(len(candidate_index)),
        "sampled_factuals": int(sample_size),
    }


def _compute_counterfactual_metrics(model, factuals, counterfactual_features: pd.DataFrame, desired_class: int | str) -> dict[str, float]:
    factual_features = factuals.get(target=False)
    valid_mask = ~counterfactual_features.isna().any(axis=1)
    successful = 0
    l1_distances: list[float] = []
    features_changed: list[int] = []

    class_to_index = model.get_class_to_index()
    desired_index = int(class_to_index[desired_class])

    if bool(valid_mask.any()):
        valid_counterfactuals = counterfactual_features.loc[valid_mask].copy(deep=True)
        predictions = model.get_prediction(valid_counterfactuals, proba=False).argmax(dim=1).detach().cpu().numpy()
        for row_index, prediction in zip(valid_counterfactuals.index, predictions, strict=False):
            if int(prediction) != desired_index:
                continue
            successful += 1
            factual = factual_features.loc[row_index].to_numpy(dtype=np.float64)
            counterfactual = valid_counterfactuals.loc[row_index].to_numpy(dtype=np.float64)
            l1_distances.append(float(np.abs(counterfactual - factual).sum()))
            features_changed.append(int(np.sum(counterfactual != factual)))

    denominator = int(counterfactual_features.shape[0])
    success_rate = float(successful / denominator) if denominator > 0 else float("nan")
    return {
        "success_rate": success_rate,
        "successful_counterfactuals": int(successful),
        "avg_l1_distance": float(np.mean(l1_distances)) if l1_distances else float("nan"),
        "avg_num_features_changed": float(np.mean(features_changed)) if features_changed else float("nan"),
    }


def _compute_cogs_loss(method, factual: np.ndarray, candidate: np.ndarray, desired_class: int | str) -> float:
    if np.isnan(candidate).any():
        return float("inf")
    num_feature_ranges = compute_ranges_numerical_features(
        method._feature_intervals,
        method._indices_categorical_features,
    )
    gower_dist = gower_distance(
        candidate,
        factual,
        num_feature_ranges,
        method._indices_categorical_features,
    )
    l0 = float(np.sum(candidate != factual) / len(factual))
    predicted = method._adapter.predict(candidate.reshape(1, -1))[0]
    failed = float(predicted != desired_class)
    return float(0.5 * gower_dist + 0.5 * l0 + failed)


def _search_best_counterfactual(
    method,
    factual: pd.Series,
    desired_class: int | str,
    n_reps: int,
) -> tuple[pd.Series, float]:
    factual_df = factual.to_frame().T
    best_cf: pd.Series | None = None
    best_loss = float("inf")
    best_runtime = float("nan")
    base_seed = 0 if method._seed is None else int(method._seed)
    factual_array = factual.to_numpy(dtype=np.float64)

    for rep_idx in range(int(n_reps)):
        method._seed = base_seed + rep_idx
        started_at = time.perf_counter()
        counterfactual_df = method.get_counterfactuals(factual_df)
        runtime_seconds = float(time.perf_counter() - started_at)
        candidate = counterfactual_df.iloc[0].to_numpy(dtype=np.float64)
        loss = _compute_cogs_loss(method, factual_array, candidate, desired_class)
        if loss < best_loss:
            best_loss = loss
            best_runtime = runtime_seconds
            best_cf = counterfactual_df.iloc[0].copy(deep=True)

    if best_cf is None:
        best_cf = pd.Series(np.nan, index=factual.index, dtype="float64")
    return best_cf, best_runtime


def _fit_reproduction_random_forest(experiment: Experiment, trainset, reproduction_cfg: dict):
    search_cfg = deepcopy(reproduction_cfg.get("rf_search", {}))
    if not bool(search_cfg.get("enabled", False)):
        experiment._target_model.fit(trainset)
        return experiment._target_model, None

    features = trainset.get(target=False).to_numpy(dtype=np.float64)
    target_series = trainset.get(target=True).iloc[:, 0].astype(int)
    labels = target_series.to_numpy(dtype=np.int64)
    seed = int(experiment._cfg["model"].get("seed", 0))
    cv_folds = int(search_cfg.get("cv_folds", 5))

    estimator = RandomForestClassifier(random_state=seed)
    param_grid = {
        "n_estimators": [int(value) for value in search_cfg["n_estimators"]],
        "min_samples_split": [
            int(value) for value in search_cfg["min_samples_split"]
        ],
        "max_features": list(search_cfg["max_features"]),
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    gcv = GridSearchCV(
        estimator,
        param_grid=param_grid,
        refit=True,
        cv=cv,
        n_jobs=1,
    )
    gcv.fit(features, labels)

    model_cfg = deepcopy(experiment._cfg["model"])
    model_name = model_cfg.pop("name")
    model_cfg["n_estimators"] = int(gcv.best_params_["n_estimators"])
    model_cfg["min_samples_split"] = int(gcv.best_params_["min_samples_split"])
    model_cfg["max_features"] = gcv.best_params_["max_features"]
    model_class = get_registry("model")[model_name]
    searched_model = model_class(**model_cfg)
    searched_model.fit(trainset)
    return searched_model, {
        "best_params": dict(gcv.best_params_),
        "best_cv_score": float(gcv.best_score_),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./experiment/cogs/config.yml",
    )
    parser.add_argument("--max-factuals", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = _apply_device(_load_config(config_path), device)

    experiment = Experiment(config)
    trainset, testset = _materialize_datasets(experiment)

    reproduction_cfg = deepcopy(config["reproduction"])
    target_model, rf_search_summary = _fit_reproduction_random_forest(
        experiment,
        trainset,
        reproduction_cfg,
    )
    model_metrics = _compute_model_metrics(target_model, testset)

    num_factuals = int(args.max_factuals) if args.max_factuals is not None else None
    desired_class = reproduction_cfg["desired_class"]
    factuals, factual_summary = _select_factuals(
        target_model,
        testset,
        desired_class=desired_class,
        sample_seed=int(reproduction_cfg["sample_seed"]),
        num_factuals=num_factuals,
    )

    method_cfg = deepcopy(config["method"])
    method_name = method_cfg.pop("name")
    method_class = get_registry("method")[method_name]
    method = method_class(target_model=target_model, **method_cfg)
    method.fit(trainset)

    factual_features = factuals.get(target=False)
    n_reps = int(reproduction_cfg.get("n_reps", 1))
    started_at = time.perf_counter()
    counterfactual_rows = []
    selected_runtimes: list[float] = []
    for _, row in tqdm(
        factual_features.iterrows(),
        total=factual_features.shape[0],
        desc="cogs-reproduce",
        leave=False,
    ):
        best_cf, best_runtime = _search_best_counterfactual(
            method,
            row,
            desired_class=desired_class,
            n_reps=n_reps,
        )
        counterfactual_rows.append(best_cf)
        selected_runtimes.append(best_runtime)
    runtime_seconds = float(time.perf_counter() - started_at)

    counterfactual_features = pd.DataFrame(
        counterfactual_rows,
        index=factual_features.index,
        columns=factual_features.columns,
    )
    cf_metrics = _compute_counterfactual_metrics(
        target_model,
        factuals,
        counterfactual_features,
        desired_class=desired_class,
    )

    print("CoGS Adult Reproduction")
    print(f"device: {device}")
    print(f"train_rows: {len(trainset)}")
    print(f"test_rows: {model_metrics['num_test_rows']}")
    print(f"test_accuracy: {model_metrics['test_accuracy']:.4f}")
    if rf_search_summary is not None:
        print(f"rf_best_params: {rf_search_summary['best_params']}")
        print(f"rf_best_cv_score: {rf_search_summary['best_cv_score']:.4f}")
    print(f"candidate_pool_size: {factual_summary['candidate_pool_size']}")
    print(f"num_factuals_evaluated: {factual_summary['sampled_factuals']}")
    print(f"n_reps: {n_reps}")
    print(f"success_rate: {cf_metrics['success_rate']:.4f}")
    print(f"successful_counterfactuals: {cf_metrics['successful_counterfactuals']}")
    print(f"avg_l1_distance: {cf_metrics['avg_l1_distance']:.4f}")
    print(f"avg_num_features_changed: {cf_metrics['avg_num_features_changed']:.4f}")
    print(f"runtime_seconds: {runtime_seconds:.3f}")
    if factual_summary["sampled_factuals"] > 0:
        print(
            f"avg_runtime_per_factual: {runtime_seconds / factual_summary['sampled_factuals']:.3f}"
        )
    if selected_runtimes:
        print(f"avg_selected_run_time: {float(np.nanmean(selected_runtimes)):.3f}")


if __name__ == "__main__":
    main()
