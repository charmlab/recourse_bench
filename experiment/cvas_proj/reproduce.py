from __future__ import annotations

import argparse
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from evaluation.evaluation_utils import resolve_evaluation_inputs
from experiment import Experiment
from utils.registry import get_registry

PROFILE_DEFAULTS = {
    "smoke": {
        "current_epochs": 100,
        "future_epochs": 100,
        "num_samples": 100,
        "num_future_models": 5,
        "max_factuals": 10,
        "rho_neg_values": [0.0, 5.0, 10.0],
    },
    "german_reference": {
        "current_epochs": None,
        "future_epochs": None,
        "num_samples": None,
        "num_future_models": 100,
        "max_factuals": None,
        "rho_neg_values": [float(value) for value in range(11)],
    },
}


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _apply_device(config: dict, device: str) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["device"] = device
    cfg["method"]["device"] = device
    cfg["model"]["save_name"] = None
    return cfg


def _resolve_profile(profile: str) -> dict:
    if profile not in PROFILE_DEFAULTS:
        raise ValueError(f"Unsupported profile: {profile}")
    return deepcopy(PROFILE_DEFAULTS[profile])


def _apply_profile_to_configs(
    current_cfg: dict,
    future_cfg: dict,
    profile_defaults: dict,
) -> tuple[dict, dict]:
    current = deepcopy(current_cfg)
    future = deepcopy(future_cfg)
    if profile_defaults["current_epochs"] is not None:
        current["model"]["epochs"] = int(profile_defaults["current_epochs"])
    if profile_defaults["future_epochs"] is not None:
        future["model"]["epochs"] = int(profile_defaults["future_epochs"])
    if profile_defaults["num_samples"] is not None:
        current["method"]["num_samples"] = int(profile_defaults["num_samples"])
    return current, future


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


def _materialize_full_dataset(config: dict):
    cfg = deepcopy(config)
    cfg["preprocess"] = [
        deepcopy(step)
        for step in cfg.get("preprocess", [])
        if step.get("name", "").lower() not in {"split", "stratified_split"}
    ]
    experiment = Experiment(cfg)
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
        raise ValueError("Expected exactly one frozen dataset after removing split")
    return datasets[0]


def _build_frozen_dataset(template_dataset, df: pd.DataFrame, marker: str):
    dataset = template_dataset.clone()
    dataset.update(marker, True, df=df.copy(deep=True))
    dataset.freeze()
    return dataset


def _compute_model_metrics(model, testset) -> dict[str, float]:
    probabilities = model.predict_proba(testset).detach().cpu()
    prediction = probabilities.argmax(dim=1)

    y = testset.get(target=True).iloc[:, 0]
    class_to_index = model.get_class_to_index()
    encoded_target = torch.tensor(
        [class_to_index[int(value)] for value in y.astype(int).tolist()],
        dtype=torch.long,
    )

    accuracy = float((prediction == encoded_target).to(dtype=torch.float32).mean())
    positive_index = class_to_index.get(1, max(class_to_index.values()))
    unique_labels = sorted(set(encoded_target.tolist()))
    if len(unique_labels) < 2:
        auc = float("nan")
    else:
        auc = float(
            roc_auc_score(
                encoded_target.numpy(),
                probabilities[:, positive_index].numpy(),
            )
        )
    return {"test_accuracy": accuracy, "test_auc": auc}


def _build_filtered_testset(testset, keep_mask: pd.Series):
    filtered = testset.clone()
    target_column = testset.target_column
    combined = pd.concat([testset.get(target=False), testset.get(target=True)], axis=1)
    filtered_df = combined.loc[keep_mask].copy(deep=True)
    filtered_df = filtered_df.loc[
        :, [*testset.get(target=False).columns, target_column]
    ]
    filtered.update("testset", True, df=filtered_df)
    filtered.freeze()
    return filtered


def _subset_dataset_rows(dataset, max_rows: int):
    combined = pd.concat([dataset.get(target=False), dataset.get(target=True)], axis=1)
    subset_df = combined.iloc[:max_rows].copy(deep=True)
    subset = dataset.clone()
    subset.update("testset", True, df=subset_df)
    subset.freeze()
    return subset


def _select_recourse_factuals(model, testset, desired_class: int | str | None):
    if desired_class is None:
        return testset

    class_to_index = model.get_class_to_index()
    desired_index = class_to_index[desired_class]
    predictions = model.predict(testset).argmax(dim=1).detach().cpu().numpy()
    keep_mask = pd.Series(
        predictions != desired_index,
        index=testset.get(target=False).index,
    )
    return _build_filtered_testset(testset, keep_mask)


def _instantiate_model(model_cfg: dict):
    cfg = deepcopy(model_cfg)
    name = cfg.pop("name")
    model_class = get_registry("TargetModel")[name]
    return model_class(**cfg)


def _instantiate_method(method_cfg: dict, target_model):
    cfg = deepcopy(method_cfg)
    name = cfg.pop("name")
    method_class = get_registry("Method")[name]
    return method_class(target_model=target_model, **cfg)


def _instantiate_evaluations(evaluation_cfg: list[dict]) -> list:
    registry = get_registry("Evaluation")
    instances = []
    for cfg in evaluation_cfg:
        item_cfg = deepcopy(cfg)
        name = item_cfg.pop("name")
        instances.append(registry[name](**item_cfg))
    return instances


def _run_current_experiment(experiment: Experiment) -> dict:
    trainset, testset = _materialize_datasets(experiment)

    experiment._target_model.fit(trainset)
    model_metrics = _compute_model_metrics(experiment._target_model, testset)

    experiment._method.fit(trainset)
    factuals = _select_recourse_factuals(
        experiment._target_model,
        testset,
        getattr(experiment._method, "_desired_class", None),
    )
    counterfactuals = experiment._method.predict(factuals)

    evaluation_results = [
        evaluation_step.evaluate(factuals, counterfactuals)
        for evaluation_step in experiment._evaluation
    ]
    metrics = pd.concat(evaluation_results, axis=1)
    experiment._metrics = metrics

    return {
        "testset": testset,
        "factuals": factuals,
        "counterfactuals": counterfactuals,
        "metrics": metrics,
        "model_metrics": model_metrics,
    }


def _run_future_model_experiment(experiment: Experiment) -> dict:
    trainset, testset = _materialize_datasets(experiment)
    experiment._target_model.fit(trainset)
    model_metrics = _compute_model_metrics(experiment._target_model, testset)
    return {"testset": testset, "model_metrics": model_metrics}


def _compute_future_validity(
    factuals,
    counterfactuals,
    future_model,
    future_testset,
) -> tuple[float, int, int]:
    (
        _factual_features,
        counterfactual_features,
        evaluation_mask,
        success_mask,
    ) = resolve_evaluation_inputs(factuals, counterfactuals)

    denominator = int(evaluation_mask.sum())
    selected_success_mask = evaluation_mask & success_mask
    successful_count = int(selected_success_mask.sum())

    expected_columns = list(future_testset.get(target=False).columns)
    if counterfactual_features.shape[1] != len(expected_columns):
        raise ValueError(
            "Current counterfactual features do not match future model input width"
        )

    if denominator == 0:
        return float("nan"), 0, 0
    if successful_count == 0:
        return 0.0, denominator, successful_count

    future_features = counterfactual_features.loc[
        selected_success_mask.to_numpy()
    ].copy(deep=True)
    future_features.columns = expected_columns
    future_prediction = future_model.get_prediction(future_features, proba=True).detach().cpu()
    positive_index = future_model.get_class_to_index().get(1, 1)
    future_positive = future_prediction[:, positive_index] >= 0.5
    future_validity = float(
        future_positive.to(dtype=torch.float32).sum().item() / denominator
    )
    return future_validity, denominator, successful_count


def _compute_future_validity_against_model(
    factuals,
    counterfactuals,
    future_model,
    expected_columns: list[str],
) -> tuple[float, int, int]:
    (
        _factual_features,
        counterfactual_features,
        evaluation_mask,
        success_mask,
    ) = resolve_evaluation_inputs(factuals, counterfactuals)

    denominator = int(evaluation_mask.sum())
    selected_success_mask = evaluation_mask & success_mask
    successful_count = int(selected_success_mask.sum())

    if counterfactual_features.shape[1] != len(expected_columns):
        raise ValueError(
            "Current counterfactual features do not match future model input width"
        )

    if denominator == 0:
        return float("nan"), 0, 0
    if successful_count == 0:
        return 0.0, denominator, successful_count

    future_features = counterfactual_features.loc[
        selected_success_mask.to_numpy()
    ].copy(deep=True)
    future_features.columns = expected_columns
    future_prediction = future_model.get_prediction(
        future_features,
        proba=True,
    ).detach().cpu()
    positive_index = future_model.get_class_to_index().get(1, 1)
    future_positive = future_prediction[:, positive_index] >= 0.5
    future_validity = float(
        future_positive.to(dtype=torch.float32).sum().item() / denominator
    )
    return future_validity, denominator, successful_count


def _prepare_current_state(current_cfg: dict, profile_defaults: dict) -> dict:
    experiment = Experiment(current_cfg)
    trainset, testset = _materialize_datasets(experiment)

    current_model = experiment._target_model
    current_model.fit(trainset)
    model_metrics = _compute_model_metrics(current_model, testset)
    evaluation_steps = _instantiate_evaluations(current_cfg["evaluation"])

    factuals = _select_recourse_factuals(
        current_model,
        testset,
        current_cfg["method"].get("desired_class"),
    )
    max_factuals = profile_defaults["max_factuals"]
    if max_factuals is not None and len(factuals) > max_factuals:
        factuals = _subset_dataset_rows(factuals, max_factuals)

    return {
        "cfg": current_cfg,
        "trainset": trainset,
        "testset": testset,
        "current_model": current_model,
        "evaluation_steps": evaluation_steps,
        "factuals": factuals,
        "model_metrics": model_metrics,
    }


def _train_future_model_pool(
    future_cfg: dict,
    profile_defaults: dict,
    num_future_models: int,
) -> dict:
    if num_future_models < 1:
        raise ValueError("num_future_models must be >= 1")

    shifted_dataset = _materialize_full_dataset(future_cfg)
    full_df = pd.concat(
        [shifted_dataset.get(target=False), shifted_dataset.get(target=True)],
        axis=1,
    )
    target = full_df.iloc[:, -1]
    expected_columns = list(shifted_dataset.get(target=False).columns)

    future_models = []
    future_metrics = []
    for future_index in range(num_future_models):
        train_df, test_df = train_test_split(
            full_df,
            train_size=0.8,
            random_state=future_index,
            stratify=target,
        )
        trainset = _build_frozen_dataset(shifted_dataset, train_df, "trainset")
        testset = _build_frozen_dataset(shifted_dataset, test_df, "testset")

        model_cfg = deepcopy(future_cfg["model"])
        future_model = _instantiate_model(model_cfg)
        future_model.fit(trainset)
        future_metrics.append(_compute_model_metrics(future_model, testset))
        future_models.append(future_model)

    return {
        "future_models": future_models,
        "expected_columns": expected_columns,
        "future_metrics": future_metrics,
    }


def _compute_future_validity_across_pool(
    factuals,
    counterfactuals,
    future_models: list,
    expected_columns: list[str],
) -> tuple[float, int, int]:
    future_validities = []
    denominator = 0
    successful_count = 0

    for future_model in future_models:
        future_validity, denominator, successful_count = (
            _compute_future_validity_against_model(
                factuals,
                counterfactuals,
                future_model,
                expected_columns,
            )
        )
        future_validities.append(float(future_validity))

    if not future_validities:
        return float("nan"), denominator, successful_count
    return float(np.mean(future_validities)), denominator, successful_count


def _run_single_surrogate_setting(
    current_state: dict,
    future_state: dict,
    surrogate_method: str,
    rho_neg: float,
) -> dict:
    method_cfg = deepcopy(current_state["cfg"]["method"])
    method_cfg["surrogate_method"] = surrogate_method
    method_cfg["rho_neg"] = float(0.0 if surrogate_method == "mpm" else rho_neg)
    method = _instantiate_method(method_cfg, current_state["current_model"])
    method.fit(current_state["trainset"])

    counterfactuals = method.predict(current_state["factuals"])
    evaluation_results = [
        evaluation_step.evaluate(current_state["factuals"], counterfactuals)
        for evaluation_step in current_state["evaluation_steps"]
    ]
    current_metrics = pd.concat(evaluation_results, axis=1).iloc[0].to_dict()
    future_validity, num_factuals, num_successful = _compute_future_validity_across_pool(
        current_state["factuals"],
        counterfactuals,
        future_state["future_models"],
        future_state["expected_columns"],
    )

    future_metric_frame = pd.DataFrame(future_state["future_metrics"])
    return {
        "surrogate_method": surrogate_method,
        "rho_neg": float(0.0 if surrogate_method == "mpm" else rho_neg),
        "num_factuals": int(num_factuals),
        "num_successful": int(num_successful),
        "current_validity": float(current_metrics.get("validity", float("nan"))),
        "future_validity": float(future_validity),
        "distance_l0": float(current_metrics.get("distance_l0", float("nan"))),
        "distance_l1": float(current_metrics.get("distance_l1", float("nan"))),
        "distance_l2": float(current_metrics.get("distance_l2", float("nan"))),
        "distance_linf": float(current_metrics.get("distance_linf", float("nan"))),
        "current_model_test_accuracy": float(
            current_state["model_metrics"]["test_accuracy"]
        ),
        "current_model_test_auc": float(current_state["model_metrics"]["test_auc"]),
        "future_model_test_accuracy_mean": float(
            future_metric_frame["test_accuracy"].mean()
        ),
        "future_model_test_auc_mean": float(future_metric_frame["test_auc"].mean()),
    }


def _is_dominated(row: pd.Series, candidates: pd.DataFrame) -> bool:
    for _, other in candidates.iterrows():
        if (
            float(other["distance_l1"]) <= float(row["distance_l1"])
            and float(other["future_validity"]) >= float(row["future_validity"])
            and (
                float(other["distance_l1"]) < float(row["distance_l1"])
                or float(other["future_validity"]) > float(row["future_validity"])
            )
        ):
            return True
    return False


def _format_value(value) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "nan"
        return f"{float(value):.6f}"
    return str(value)


def _print_single_output(output: dict) -> None:
    for key, value in output.items():
        print(f"{key}: {_format_value(value)}")


def _print_compare_output(
    results: pd.DataFrame,
    profile: str,
    device: str,
    num_future_models: int,
    max_factuals: int | None,
) -> None:
    print(f"profile: {profile}")
    print(f"device: {device}")
    print(f"num_future_models: {num_future_models}")
    print(
        "max_factuals: "
        + ("all" if max_factuals is None else str(int(max_factuals)))
    )
    print()
    print(results.to_string(index=False))

    print()
    print("pareto_frontier_by_surrogate:")
    for surrogate_method, group in results.groupby("surrogate_method", sort=False):
        ordered = group.sort_values(["distance_l1", "future_validity"]).reset_index(
            drop=True
        )
        pareto_rows = ordered[
            ~ordered.apply(lambda row: _is_dominated(row, ordered), axis=1)
        ]
        print(f"[{surrogate_method}]")
        print(
            pareto_rows[
                [
                    "rho_neg",
                    "current_validity",
                    "future_validity",
                    "distance_l1",
                    "num_successful",
                    "num_factuals",
                ]
            ].to_string(index=False)
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-config",
        default="./experiment/cvas_proj/german_mlp_cvas_proj_reproduce_current.yaml",
    )
    parser.add_argument(
        "--future-config",
        default="./experiment/cvas_proj/german_mlp_cvas_proj_reproduce_future.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "compare"],
        default="compare",
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "german_reference"],
        default="german_reference",
    )
    parser.add_argument("--surrogate-methods", nargs="*", default=["fr_rmpm"])
    parser.add_argument("--rho-neg-values", nargs="*", type=float, default=[10.0])
    parser.add_argument("--num-future-models", type=int, default=None)
    parser.add_argument("--max-factuals", type=int, default=None)
    args = parser.parse_args()

    device = "cpu"
    profile_defaults = _resolve_profile(args.profile)

    current_cfg_base = _apply_device(
        _load_config((PROJECT_ROOT / args.current_config).resolve()),
        device,
    )
    future_cfg_base = _apply_device(
        _load_config((PROJECT_ROOT / args.future_config).resolve()),
        device,
    )
    current_cfg, future_cfg = _apply_profile_to_configs(
        current_cfg_base,
        future_cfg_base,
        profile_defaults,
    )

    if args.mode == "single":
        current_experiment = Experiment(current_cfg)
        current_results = _run_current_experiment(current_experiment)

        future_experiment = Experiment(future_cfg)
        future_results = _run_future_model_experiment(future_experiment)

        future_validity, num_factuals, num_successful = _compute_future_validity(
            current_results["factuals"],
            current_results["counterfactuals"],
            future_experiment._target_model,
            future_results["testset"],
        )

        current_metrics = current_results["metrics"].iloc[0].to_dict()
        output = {
            "mode": args.mode,
            "profile": args.profile,
            "device": device,
            "current_model_test_accuracy": float(
                current_results["model_metrics"]["test_accuracy"]
            ),
            "current_model_test_auc": float(
                current_results["model_metrics"]["test_auc"]
            ),
            "current_validity": float(current_metrics.get("validity", float("nan"))),
            "distance_l0": float(current_metrics.get("distance_l0", float("nan"))),
            "distance_l1": float(current_metrics.get("distance_l1", float("nan"))),
            "distance_l2": float(current_metrics.get("distance_l2", float("nan"))),
            "distance_linf": float(
                current_metrics.get("distance_linf", float("nan"))
            ),
            "future_model_test_accuracy": float(
                future_results["model_metrics"]["test_accuracy"]
            ),
            "future_model_test_auc": float(
                future_results["model_metrics"]["test_auc"]
            ),
            "future_validity": future_validity,
            "num_factuals": num_factuals,
            "num_successful": num_successful,
        }
        _print_single_output(output)
        return

    if args.num_future_models is not None:
        profile_defaults["num_future_models"] = int(args.num_future_models)
    if args.max_factuals is not None:
        profile_defaults["max_factuals"] = int(args.max_factuals)

    surrogate_methods = (
        list(args.surrogate_methods)
        if args.surrogate_methods
        else ["mpm", "quad_rmpm", "bw_rmpm", "fr_rmpm"]
    )
    rho_neg_values = (
        [float(value) for value in args.rho_neg_values]
        if args.rho_neg_values
        else list(profile_defaults["rho_neg_values"])
    )

    current_state = _prepare_current_state(current_cfg, profile_defaults)
    future_state = _train_future_model_pool(
        future_cfg,
        profile_defaults,
        num_future_models=int(profile_defaults["num_future_models"]),
    )

    rows = []
    for surrogate_method in surrogate_methods:
        if surrogate_method == "mpm":
            rows.append(
                _run_single_surrogate_setting(
                    current_state=current_state,
                    future_state=future_state,
                    surrogate_method=surrogate_method,
                    rho_neg=0.0,
                )
            )
            continue

        for rho_neg in rho_neg_values:
            rows.append(
                _run_single_surrogate_setting(
                    current_state=current_state,
                    future_state=future_state,
                    surrogate_method=surrogate_method,
                    rho_neg=float(rho_neg),
                )
            )

    results = pd.DataFrame(rows).sort_values(
        ["surrogate_method", "rho_neg"],
        ignore_index=True,
    )
    _print_compare_output(
        results=results,
        profile=args.profile,
        device=device,
        num_future_models=int(profile_defaults["num_future_models"]),
        max_factuals=profile_defaults["max_factuals"],
    )


if __name__ == "__main__":
    main()
