from __future__ import annotations

import argparse
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from dataset.dataset_object import DatasetObject
from method.sns.support import (
    min_l2_search,
    resolve_target_indices,
    sns_search,
)
from utils.registry import get_registry


DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_STDOUT_LOG = PROJECT_ROOT / "experiment" / "sns" / "reproduce_stdout.txt"

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "german_credit": {
        "display_name": "German Credit",
        "folder": "german_sns",
        "train_csv": "german_sns_train.csv",
        "test_csv": "german_sns_test.csv",
        "affinity_set": [[0], [1]],
        "paper": {
            "min_l2": {"loo_invalidation": 0.36, "rs_invalidation": 0.56, "l2_cost": 4.49},
            "min_l2_sns": {"loo_invalidation": 0.00, "rs_invalidation": 0.06, "l2_cost": 6.23},
        },
        "method_overrides": {},
    },
    "seizure": {
        "display_name": "Seizure",
        "folder": "seizure_sns",
        "train_csv": "seizure_sns_train.csv",
        "test_csv": "seizure_sns_test.csv",
        "affinity_set": [[0], [1]],
        "paper": {
            "min_l2": {"loo_invalidation": 0.64, "rs_invalidation": 0.77, "l2_cost": 8.23},
            "min_l2_sns": {"loo_invalidation": 0.02, "rs_invalidation": 0.13, "l2_cost": 9.60},
        },
        "method_overrides": {},
    },
    "ctg": {
        "display_name": "CTG",
        "folder": "ctg_sns",
        "train_csv": "ctg_sns_train.csv",
        "test_csv": "ctg_sns_test.csv",
        "affinity_set": [[0], [1]],
        "paper": {
            "min_l2": {"loo_invalidation": 0.48, "rs_invalidation": 0.49, "l2_cost": 0.06},
            "min_l2_sns": {"loo_invalidation": 0.00, "rs_invalidation": 0.00, "l2_cost": 0.21},
        },
        "method_overrides": {},
    },
    "warfarin": {
        "display_name": "Warfarin",
        "folder": "warfarin_sns",
        "train_csv": "warfarin_sns_train.csv",
        "test_csv": "warfarin_sns_test.csv",
        "affinity_set": [[0], [1, 2]],
        "paper": {
            "min_l2": {"loo_invalidation": 0.35, "rs_invalidation": 0.30, "l2_cost": 0.54},
            "min_l2_sns": {"loo_invalidation": 0.00, "rs_invalidation": 0.00, "l2_cost": 0.90},
        },
        "method_overrides": {},
    },
    "heloc": {
        "display_name": "HELOC",
        "folder": "heloc_sns",
        "train_csv": "heloc_sns_train.csv",
        "test_csv": "heloc_sns_test.csv",
        "affinity_set": [[0], [1]],
        "paper": {
            "min_l2": {"loo_invalidation": 0.55, "rs_invalidation": 0.61, "l2_cost": 0.11},
            "min_l2_sns": {"loo_invalidation": 0.00, "rs_invalidation": 0.00, "l2_cost": 1.71},
        },
        "method_overrides": {},
    },
    "taiwanese_credit": {
        "display_name": "Taiwanese Credit",
        "folder": "taiwanese_credit_sns",
        "train_csv": "taiwanese_credit_sns_train.csv",
        "test_csv": "taiwanese_credit_sns_test.csv",
        "affinity_set": [[0], [1]],
        "paper": {
            "min_l2": {"loo_invalidation": 0.27, "rs_invalidation": 0.72, "l2_cost": 2.65},
            "min_l2_sns": {"loo_invalidation": 0.00, "rs_invalidation": 0.04, "l2_cost": 4.68},
        },
        "method_overrides": {},
    },
}


class FrameDataset(DatasetObject):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        target_column: str,
        name: str,
        feature_names: list[str],
        trainset: bool = False,
        testset: bool = False,
    ):
        self._rawdf = df.copy(deep=True)
        self._freeze = False
        self.name = name
        self.target_column = target_column
        self.trainset = bool(trainset)
        self.testset = bool(testset)
        self.counterfactual = False
        self.evaluation_filter = False
        self.raw_feature_type = {feature: "numerical" for feature in feature_names}
        self.raw_feature_mutability = {feature: True for feature in feature_names}
        self.raw_feature_actionability = {feature: "any" for feature in feature_names}

    def _read_df(self, path: str) -> pd.DataFrame:
        raise NotImplementedError("FrameDataset reads from an in-memory DataFrame")


class TeeStdout:
    def __init__(self, file):
        self._file = file
        self._stdout = sys.__stdout__

    def write(self, data: str) -> int:
        self._stdout.write(data)
        self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._stdout, "isatty", lambda: False)())


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


def _build_model_from_cfg(cfg: dict, seed: int | None = None):
    model_cfg = deepcopy(cfg["model"])
    if seed is not None:
        model_cfg["seed"] = int(seed)
    model_name = model_cfg.pop("name")
    model_class = get_registry("model")[model_name]
    return model_class(**model_cfg)


def _build_method_from_cfg(cfg: dict, target_model):
    method_cfg = deepcopy(cfg["method"])
    method_name = method_cfg.pop("name")
    method_class = get_registry("method")[method_name]
    return method_class(target_model=target_model, **method_cfg)


def _make_dataset(
    features: pd.DataFrame,
    target: np.ndarray | pd.Series,
    *,
    target_column: str,
    name: str,
    trainset: bool = False,
    testset: bool = False,
) -> FrameDataset:
    target_series = pd.Series(np.asarray(target), index=features.index, name=target_column)
    combined = pd.concat([features.copy(deep=True), target_series], axis=1)
    dataset = FrameDataset(
        combined,
        target_column=target_column,
        name=name,
        feature_names=list(features.columns),
        trainset=trainset,
        testset=testset,
    )
    dataset.freeze()
    return dataset


def _load_local_dataset(dataset_key: str) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset_key]
    folder = DATASET_ROOT / spec["folder"]
    train_path = folder / spec["train_csv"]
    test_path = folder / spec["test_csv"]
    if not train_path.exists():
        raise FileNotFoundError(f"Missing local SNS train csv: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing local SNS test csv: {test_path}")

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    target_column = "target"
    feature_names = [column for column in train_raw.columns if column != target_column]
    train_df = train_raw.loc[:, feature_names].astype(np.float32)
    test_df = test_raw.loc[:, feature_names].astype(np.float32)
    y_train = train_raw.loc[:, target_column].astype(np.int64).to_numpy()
    y_test = test_raw.loc[:, target_column].astype(np.int64).to_numpy()

    trainset = _make_dataset(
        train_df,
        y_train,
        target_column=target_column,
        name=f"{dataset_key}_train_reproduction",
        trainset=True,
    )
    testset = _make_dataset(
        test_df,
        y_test,
        target_column=target_column,
        name=f"{dataset_key}_test_reproduction",
        testset=True,
    )
    return {
        "spec": spec,
        "trainset": trainset,
        "testset": testset,
        "train_df": train_df,
        "test_df": test_df,
        "y_train": y_train,
        "y_test": y_test,
        "feature_count": int(train_df.shape[1]),
        "class_count": int(len(np.unique(np.concatenate([y_train, y_test])))),
    }


def _select_factuals(testset, max_factuals: int | None, sample_seed: int) -> pd.DataFrame:
    factuals = testset.get(target=False).copy(deep=True)
    if max_factuals is None or max_factuals >= factuals.shape[0]:
        return factuals
    rng = np.random.RandomState(sample_seed)
    selected_positions = np.sort(rng.choice(factuals.shape[0], size=max_factuals, replace=False))
    return factuals.iloc[selected_positions].copy(deep=True)


def _predict_indices(model, features: pd.DataFrame) -> np.ndarray:
    return (
        model.get_prediction(features, proba=True)
        .detach()
        .cpu()
        .numpy()
        .argmax(axis=1)
        .astype(np.int64, copy=False)
    )


def _validate_candidates(
    target_model,
    factuals: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    desired_class: int | str | None,
) -> pd.DataFrame:
    if list(candidates.columns) != list(factuals.columns):
        candidates = candidates.reindex(columns=factuals.columns)
    candidates = candidates.copy(deep=True)

    valid_rows = ~candidates.isna().any(axis=1)
    if not bool(valid_rows.any()):
        return candidates

    factual_prediction = _predict_indices(target_model, factuals)
    candidate_prediction = _predict_indices(target_model, candidates.loc[valid_rows])

    if desired_class is None:
        success = candidate_prediction != factual_prediction[valid_rows.to_numpy()]
    else:
        class_to_index = target_model.get_class_to_index()
        desired_index = int(class_to_index[desired_class])
        success = candidate_prediction == desired_index

    success_mask = pd.Series(False, index=candidates.index, dtype=bool)
    success_mask.loc[valid_rows] = success
    candidates.loc[~success_mask, :] = np.nan
    return candidates


def _run_base_and_sns(method, factuals: pd.DataFrame):
    original_prediction = _predict_indices(method._target_model, factuals)

    base_rows: list[np.ndarray] = []
    sns_rows: list[np.ndarray] = []
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
        if len(method._target_model.get_class_to_index()) == 2:
            target_index = int(
                resolve_target_indices(
                    method._target_model,
                    np.asarray([original_index], dtype=np.int64),
                    desired_class=None,
                )[0]
            )
        else:
            target_index = original_index

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
        base_cf_df = pd.DataFrame([base_cf], columns=factuals.columns)
        base_prediction = int(_predict_indices(method._target_model, base_cf_df)[0])
        sns_cf = sns_search(
            method._target_model,
            counterfactual=base_cf,
            target_index=base_prediction,
            clamp=method._clamp,
            sns_eps=method._sns_eps,
            sns_nb_iters=method._sns_nb_iters,
            sns_eps_iter=method._sns_eps_iter,
            n_interpolations=method._n_interpolations,
        )
        sns_rows.append(sns_cf)

    base_df = pd.DataFrame(base_rows, index=factuals.index, columns=factuals.columns)
    sns_df = pd.DataFrame(sns_rows, index=factuals.index, columns=factuals.columns)
    base_df = _validate_candidates(
        target_model=method._target_model,
        factuals=factuals,
        candidates=base_df,
        desired_class=method._desired_class,
    )
    sns_df = _validate_candidates(
        target_model=method._target_model,
        factuals=factuals,
        candidates=sns_df,
        desired_class=method._desired_class,
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


def _convert_to_super_labels(preds: np.ndarray, affinity_set: list[list[int]]) -> np.ndarray:
    converted = preds.astype(np.int64, copy=True)
    for subset in affinity_set:
        super_label = int(subset[0])
        for label in subset:
            converted[converted == int(label)] = super_label
    return converted


def _compute_invalidation_rate(
    counterfactuals: pd.DataFrame,
    original_prediction: np.ndarray,
    related_models: list,
    affinity_set: list[list[int]],
) -> float:
    valid_rows = ~counterfactuals.isna().any(axis=1)
    if not bool(valid_rows.any()):
        return float("nan")

    valid_counterfactuals = counterfactuals.loc[valid_rows]
    original_valid = original_prediction[valid_rows.to_numpy()].astype(np.int64, copy=False)
    original_super = _convert_to_super_labels(original_valid, affinity_set)

    invalidation_rates = []
    for model in related_models:
        prediction = _predict_indices(model, valid_counterfactuals)
        prediction_super = _convert_to_super_labels(prediction, affinity_set)
        invalidation_rates.append(float(np.mean(prediction_super == original_super)))
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
        reduced_trainset = _make_dataset(
            reduced_df.loc[:, reduced_df.columns != trainset.target_column],
            reduced_df.loc[:, trainset.target_column].to_numpy(),
            target_column=trainset.target_column,
            name=f"{trainset.name}_loo_{removed_index}",
            trainset=True,
        )
        model = _build_model_from_cfg(config, seed=int(config["model"]["seed"]))
        model.fit(reduced_trainset)
        models.append(model)
    return models


def _print_comparison(prefix: str, reproduced: float, paper: float) -> None:
    print(f"{prefix}: {reproduced:.4f}")
    print(f"{prefix}_paper: {paper:.4f}")


def _resolve_dataset_keys(argument: str) -> list[str]:
    if argument == "all":
        return list(DATASET_SPECS.keys())
    if argument not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset key: {argument}")
    return [argument]


def _prepare_dataset_config(base_config: dict, dataset_key: str) -> dict:
    cfg = deepcopy(base_config)
    dataset_spec = DATASET_SPECS[dataset_key]
    cfg["name"] = f"sns_{dataset_key}_reproduce"
    method_overrides = dataset_spec.get("method_overrides", {})
    for key, value in method_overrides.items():
        cfg["method"][key] = deepcopy(value)
    return cfg


def _run_single_dataset(
    *,
    dataset_key: str,
    config: dict,
    max_factuals: int | None,
    max_related_models: int | None,
) -> dict[str, Any]:
    loaded = _load_local_dataset(dataset_key)
    spec = loaded["spec"]
    trainset = loaded["trainset"]
    testset = loaded["testset"]

    base_model = _build_model_from_cfg(config, seed=int(config["model"]["seed"]))
    base_model.fit(trainset)
    method = _build_method_from_cfg(config, target_model=base_model)
    method.fit(trainset)

    factuals = _select_factuals(
        testset,
        max_factuals=max_factuals,
        sample_seed=int(config["reproduction"]["sample_seed"]),
    )

    started_at = time.perf_counter()
    original_prediction, base_cfs, sns_cfs = _run_base_and_sns(method, factuals)
    rs_models = _build_rs_models(config, trainset, max_related_models)
    loo_models = _build_loo_models(config, trainset, max_related_models)
    runtime = float(time.perf_counter() - started_at)

    base_valid_mask = ~base_cfs.isna().any(axis=1)
    sns_valid_mask = ~sns_cfs.isna().any(axis=1)
    base_success_rate = float(base_valid_mask.mean())
    sns_success_rate = float(sns_valid_mask.mean())
    base_valid_count = int(base_valid_mask.sum())
    sns_valid_count = int(sns_valid_mask.sum())
    base_costs = _compute_l2_costs(factuals, base_cfs)
    sns_costs = _compute_l2_costs(factuals, sns_cfs)

    base_rs_iv = _compute_invalidation_rate(base_cfs, original_prediction, rs_models, spec["affinity_set"])
    sns_rs_iv = _compute_invalidation_rate(sns_cfs, original_prediction, rs_models, spec["affinity_set"])
    base_loo_iv = _compute_invalidation_rate(base_cfs, original_prediction, loo_models, spec["affinity_set"])
    sns_loo_iv = _compute_invalidation_rate(sns_cfs, original_prediction, loo_models, spec["affinity_set"])

    paper_cfg = spec["paper"]
    baseline_accuracy = _compute_accuracy(base_model, testset)

    print()
    print(f"SNS {spec['display_name']} Reproduction")
    print(f"dataset_key: {dataset_key}")
    print(f"device: {config['model']['device']}")
    print(f"feature_count: {loaded['feature_count']}")
    print(f"class_count: {loaded['class_count']}")
    print(f"train_rows: {len(trainset)}")
    print(f"test_rows: {len(testset)}")
    print(f"base_model_test_accuracy: {baseline_accuracy:.4f}")
    print(f"num_factuals_evaluated: {len(factuals)}")
    print(f"rs_models_evaluated: {len(rs_models)}")
    print(f"loo_models_evaluated: {len(loo_models)}")
    print(f"base_valid_counterfactuals: {base_valid_count}")
    print(f"sns_valid_counterfactuals: {sns_valid_count}")
    print(f"base_success_rate: {base_success_rate:.4f}")
    print(f"sns_success_rate: {sns_success_rate:.4f}")
    print(f"base_avg_l2_cost: {float(np.mean(base_costs)) if base_costs else float('nan'):.4f}")
    print(f"sns_avg_l2_cost: {float(np.mean(sns_costs)) if sns_costs else float('nan'):.4f}")
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

    return {
        "dataset_key": dataset_key,
        "display_name": spec["display_name"],
        "feature_count": loaded["feature_count"],
        "class_count": loaded["class_count"],
        "baseline_accuracy": baseline_accuracy,
        "base_loo_iv": base_loo_iv,
        "base_rs_iv": base_rs_iv,
        "base_cost": float(np.mean(base_costs)) if base_costs else float("nan"),
        "sns_loo_iv": sns_loo_iv,
        "sns_rs_iv": sns_rs_iv,
        "sns_cost": float(np.mean(sns_costs)) if sns_costs else float("nan"),
        "paper": paper_cfg,
    }


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    summary = pd.DataFrame(
        [
            {
                "dataset": item["dataset_key"],
                "base_loo_iv": item["base_loo_iv"],
                "base_rs_iv": item["base_rs_iv"],
                "base_cost": item["base_cost"],
                "sns_loo_iv": item["sns_loo_iv"],
                "sns_rs_iv": item["sns_rs_iv"],
                "sns_cost": item["sns_cost"],
            }
            for item in results
        ]
    )
    print()
    print("SNS Reproduction Summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./experiment/sns/config.yaml")
    parser.add_argument("--dataset", default="all", choices=["all", *DATASET_SPECS.keys()])
    parser.add_argument("--max-factuals", type=int, default=None)
    parser.add_argument("--max-related-models", type=int, default=None)
    parser.add_argument("--stdout-log", default=str(DEFAULT_STDOUT_LOG))
    args = parser.parse_args()

    log_path = Path(args.stdout_log).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = TeeStdout(log_file)
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            config_path = (PROJECT_ROOT / args.config).resolve()
            base_config = _apply_device(_load_config(config_path), device)
            logging.getLogger("art").setLevel(logging.WARNING)
            logging.getLogger("art.attacks.evasion.elastic_net").setLevel(logging.WARNING)

            dataset_keys = _resolve_dataset_keys(args.dataset)
            results = []
            for dataset_key in dataset_keys:
                dataset_config = _prepare_dataset_config(base_config, dataset_key)
                results.append(
                    _run_single_dataset(
                        dataset_key=dataset_key,
                        config=dataset_config,
                        max_factuals=args.max_factuals,
                        max_related_models=args.max_related_models,
                    )
                )
            _print_summary_table(results)
            print()
            print(f"stdout_log_path: {log_path}")
        finally:
            sys.stdout.flush()
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
