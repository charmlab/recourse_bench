from __future__ import annotations

import argparse
import logging
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
from tqdm import tqdm

from experiment import Experiment
from method.sns.support import resolve_target_indices, sns_search, validate_counterfactuals
from method.sns.support import min_l2_search
from utils.registry import get_registry


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _apply_device(config: dict, device: str) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["device"] = device
    cfg["method"]["device"] = device
    return cfg


def _materialize_single_dataset(experiment: Experiment):
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
    if len(datasets) != 1:
        raise ValueError("SNS reproduction expects preprocess to yield one dataset")
    return datasets[0]


def _clone_frozen_dataset(template, df: pd.DataFrame, flag: str):
    dataset = template.clone()
    dataset.update(flag, True, df=df)
    dataset.freeze()
    return dataset


def _build_reference_split(encoded_dataset, cfg: dict):
    reproduction_cfg = cfg["reproduction"]
    split_seed = int(reproduction_cfg["split_seed"])
    train_size = int(reproduction_cfg["train_size"])
    test_size = int(reproduction_cfg["test_size"])

    full_df = encoded_dataset.clone().snapshot()
    num_rows = int(full_df.shape[0])
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must both be >= 1")
    if train_size + test_size > num_rows:
        raise ValueError("train_size + test_size exceeds available rows")

    rng = np.random.RandomState(split_seed)
    positions = rng.permutation(num_rows)
    train_positions = positions[:train_size]
    test_positions = positions[train_size : train_size + test_size]
    unused_positions = positions[train_size + test_size :]

    train_df = full_df.iloc[train_positions].copy(deep=True)
    test_df = full_df.iloc[test_positions].copy(deep=True)

    feature_columns = [column for column in full_df.columns if column != encoded_dataset.target_column]
    mean = train_df.loc[:, feature_columns].mean(axis=0)
    std = train_df.loc[:, feature_columns].std(axis=0, ddof=0).replace(0.0, 1.0)

    train_scaled = train_df.copy(deep=True)
    test_scaled = test_df.copy(deep=True)
    float_columns = {column: "float64" for column in feature_columns}
    train_scaled = train_scaled.astype(float_columns)
    test_scaled = test_scaled.astype(float_columns)
    train_scaled.loc[:, feature_columns] = (train_df.loc[:, feature_columns] - mean) / std
    test_scaled.loc[:, feature_columns] = (test_df.loc[:, feature_columns] - mean) / std

    trainset = _clone_frozen_dataset(encoded_dataset, train_scaled, "trainset")
    testset = _clone_frozen_dataset(encoded_dataset, test_scaled, "testset")

    return trainset, testset, int(len(unused_positions))


def _build_model_from_cfg(cfg: dict, seed: int | None = None):
    model_cfg = deepcopy(cfg["model"])
    if seed is not None:
        model_cfg["seed"] = int(seed)
    model_name = model_cfg.pop("name")
    model_class = get_registry("model")[model_name]
    return model_class(**model_cfg)


def _select_factuals(testset, max_factuals: int | None, sample_seed: int) -> pd.DataFrame:
    factuals = testset.get(target=False).copy(deep=True)
    if max_factuals is None or max_factuals >= factuals.shape[0]:
        return factuals
    rng = np.random.RandomState(sample_seed)
    selected_positions = np.sort(rng.choice(factuals.shape[0], size=max_factuals, replace=False))
    return factuals.iloc[selected_positions].copy(deep=True)


def _run_base_and_sns(method, factuals: pd.DataFrame):
    original_prediction = (
        method._target_model.get_prediction(factuals, proba=True)
        .detach()
        .cpu()
        .numpy()
        .argmax(axis=1)
    )
    target_indices = resolve_target_indices(
        method._target_model,
        original_prediction,
        desired_class=None,
    )

    base_rows = []
    sns_rows = []
    for row_position, (_, row) in enumerate(
        tqdm(
            factuals.iterrows(),
            total=factuals.shape[0],
            desc="sns-reproduce",
            leave=False,
        )
    ):
        factual = row.to_numpy(dtype=np.float64)
        original_index = int(original_prediction[row_position])
        target_index = int(target_indices[row_position])
        base_cf = min_l2_search(
            method._target_model,
            factual=factual,
            original_index=original_index,
            target_index=target_index,
            clamp=method._clamp,
            steps=method._base_steps,
            step_size=method._base_step_size,
            confidence=method._base_confidence,
            beta=method._base_beta,
            targeted=False,
            art_classifier=method._art_classifier,
            lambda_start=method._base_lambda_start,
            lambda_growth=method._base_lambda_growth,
            lambda_max=method._base_lambda_max,
        )
        if base_cf is None:
            nan_row = np.full(len(method._feature_names), np.nan, dtype=np.float64)
            base_rows.append(nan_row)
            sns_rows.append(nan_row)
            continue

        base_rows.append(base_cf)
        sns_cf = sns_search(
            method._target_model,
            counterfactual=base_cf,
            target_index=target_index,
            clamp=method._clamp,
            sns_eps=method._sns_eps,
            sns_nb_iters=method._sns_nb_iters,
            sns_eps_iter=method._sns_eps_iter,
            n_interpolations=method._n_interpolations,
        )
        sns_rows.append(sns_cf)

    base_df = pd.DataFrame(base_rows, index=factuals.index, columns=factuals.columns)
    sns_df = pd.DataFrame(sns_rows, index=factuals.index, columns=factuals.columns)
    base_df = validate_counterfactuals(
        target_model=method._target_model,
        factuals=factuals,
        candidates=base_df,
        desired_class=None,
    )
    sns_df = validate_counterfactuals(
        target_model=method._target_model,
        factuals=factuals,
        candidates=sns_df,
        desired_class=None,
    )
    return original_prediction, base_df, sns_df


def _compute_l2_costs(factuals: pd.DataFrame, counterfactuals: pd.DataFrame) -> list[float]:
    costs = []
    for idx in counterfactuals.index:
        cf = counterfactuals.loc[idx]
        if cf.isna().any():
            continue
        factual = factuals.loc[idx]
        costs.append(
            float(
                np.linalg.norm(
                    cf.to_numpy(dtype=np.float64) - factual.to_numpy(dtype=np.float64),
                    ord=2,
                )
            )
        )
    return costs


def _compute_invalidation_rate(
    counterfactuals: pd.DataFrame,
    original_prediction: np.ndarray,
    related_models: list,
) -> float:
    valid_rows = ~counterfactuals.isna().any(axis=1)
    if not bool(valid_rows.any()):
        return float("nan")

    valid_counterfactuals = counterfactuals.loc[valid_rows]
    original_valid = original_prediction[valid_rows.to_numpy()]

    invalidation_rates = []
    for model in related_models:
        prediction = (
            model.get_prediction(valid_counterfactuals, proba=False)
            .argmax(dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        invalidation_rates.append(float(np.mean(prediction == original_valid)))
    return float(np.mean(invalidation_rates))


def _compute_accuracy(model, testset) -> float:
    prediction = model.predict(testset).argmax(dim=1).detach().cpu().numpy()
    target = testset.get(target=True).iloc[:, 0].astype(int).to_numpy()
    return float(np.mean(prediction == target))


def _build_rs_models(config: dict, trainset, max_related_models: int | None) -> list:
    reproduction_cfg = config["reproduction"]
    rs_count = int(reproduction_cfg["rs_count"])
    if max_related_models is not None:
        rs_count = min(rs_count, int(max_related_models))
    rs_seed_start = int(reproduction_cfg["rs_seed_start"])

    models = []
    for offset in range(rs_count):
        model = _build_model_from_cfg(config, seed=rs_seed_start + offset)
        model.fit(trainset)
        models.append(model)
    return models


def _build_loo_models(config: dict, trainset, max_related_models: int | None) -> list:
    reproduction_cfg = config["reproduction"]
    loo_count = int(reproduction_cfg["loo_count"])
    if max_related_models is not None:
        loo_count = min(loo_count, int(max_related_models))

    train_df = trainset.clone().snapshot()
    rng = np.random.RandomState(int(reproduction_cfg["loo_selection_seed"]))
    sampled_positions = rng.choice(train_df.shape[0], size=loo_count, replace=False)
    sampled_indices = train_df.index[sampled_positions]

    models = []
    for removed_index in sampled_indices:
        reduced_df = train_df.drop(index=removed_index).copy(deep=True)
        reduced_trainset = _clone_frozen_dataset(trainset, reduced_df, "trainset")
        model = _build_model_from_cfg(config, seed=int(config["model"]["seed"]))
        model.fit(reduced_trainset)
        models.append(model)
    return models


def _print_comparison(prefix: str, reproduced: float, paper: float) -> None:
    print(f"{prefix}: {reproduced:.4f}")
    print(f"{prefix}_paper: {paper:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./experiment/sns/config.yaml")
    parser.add_argument("--max-factuals", type=int, default=None)
    parser.add_argument("--max-related-models", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = _apply_device(_load_config(config_path), device)
    logging.getLogger("art").setLevel(logging.WARNING)
    logging.getLogger("art.attacks.evasion.elastic_net").setLevel(logging.WARNING)

    experiment = Experiment(config)
    encoded_dataset = _materialize_single_dataset(experiment)
    trainset, testset, unused_rows = _build_reference_split(encoded_dataset, config)

    feature_count = trainset.get(target=False).shape[1]
    expected_feature_count = int(config["reproduction"]["expected_feature_count"])
    if feature_count != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} encoded German features, got {feature_count}"
        )

    base_model = experiment._target_model
    base_model.fit(trainset)
    method = experiment._method
    method.fit(trainset)

    factuals = _select_factuals(
        testset,
        max_factuals=args.max_factuals,
        sample_seed=int(config["reproduction"]["sample_seed"]),
    )

    started_at = time.perf_counter()
    original_prediction, base_cfs, sns_cfs = _run_base_and_sns(method, factuals)
    rs_models = _build_rs_models(config, trainset, args.max_related_models)
    loo_models = _build_loo_models(config, trainset, args.max_related_models)
    runtime = float(time.perf_counter() - started_at)

    base_success_rate = float((~base_cfs.isna().any(axis=1)).mean())
    sns_success_rate = float((~sns_cfs.isna().any(axis=1)).mean())
    base_valid_count = int((~base_cfs.isna().any(axis=1)).sum())
    sns_valid_count = int((~sns_cfs.isna().any(axis=1)).sum())
    base_costs = _compute_l2_costs(factuals, base_cfs)
    sns_costs = _compute_l2_costs(factuals, sns_cfs)

    base_rs_iv = _compute_invalidation_rate(base_cfs, original_prediction, rs_models)
    sns_rs_iv = _compute_invalidation_rate(sns_cfs, original_prediction, rs_models)
    base_loo_iv = _compute_invalidation_rate(base_cfs, original_prediction, loo_models)
    sns_loo_iv = _compute_invalidation_rate(sns_cfs, original_prediction, loo_models)

    paper_cfg = config["reproduction"]["paper"]

    print("SNS German Credit Reproduction")
    print(f"device: {device}")
    print(f"encoded_feature_count: {feature_count}")
    print(f"train_rows: {len(trainset)}")
    print(f"test_rows: {len(testset)}")
    print(f"unused_rows: {unused_rows}")
    print(f"base_model_test_accuracy: {_compute_accuracy(base_model, testset):.4f}")
    print(f"num_factuals_evaluated: {len(factuals)}")
    print(f"rs_models_evaluated: {len(rs_models)}")
    print(f"loo_models_evaluated: {len(loo_models)}")
    print(f"base_valid_counterfactuals: {base_valid_count}")
    print(f"sns_valid_counterfactuals: {sns_valid_count}")
    print(f"base_success_rate: {base_success_rate:.4f}")
    print(f"sns_success_rate: {sns_success_rate:.4f}")
    print(
        f"base_avg_l2_cost: "
        f"{float(np.mean(base_costs)) if base_costs else float('nan'):.4f}"
    )
    print(
        f"sns_avg_l2_cost: "
        f"{float(np.mean(sns_costs)) if sns_costs else float('nan'):.4f}"
    )
    _print_comparison(
        "base_loo_invalidation_rate",
        base_loo_iv,
        float(paper_cfg["min_l2"]["loo_invalidation"]),
    )
    _print_comparison(
        "base_rs_invalidation_rate",
        base_rs_iv,
        float(paper_cfg["min_l2"]["rs_invalidation"]),
    )
    _print_comparison(
        "sns_loo_invalidation_rate",
        sns_loo_iv,
        float(paper_cfg["min_l2_sns"]["loo_invalidation"]),
    )
    _print_comparison(
        "sns_rs_invalidation_rate",
        sns_rs_iv,
        float(paper_cfg["min_l2_sns"]["rs_invalidation"]),
    )
    print(f"base_avg_l2_cost_paper: {float(paper_cfg['min_l2']['l2_cost']):.4f}")
    print(f"sns_avg_l2_cost_paper: {float(paper_cfg['min_l2_sns']['l2_cost']):.4f}")
    print(f"runtime_seconds: {runtime:.3f}")


if __name__ == "__main__":
    main()
