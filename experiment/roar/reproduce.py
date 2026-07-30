from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from recourse_bench.evaluation.evaluation_utils import resolve_evaluation_inputs
from recourse_bench.experiments import Experiment


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _apply_device(config: dict, device: str) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["device"] = device
    cfg["method"]["device"] = device
    return cfg


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


def _reference_style_split(dataset, split_preprocess) -> tuple[object, object]:
    df = dataset.snapshot()
    split = split_preprocess._split
    sample = split_preprocess._sample
    seed = split_preprocess._seed

    if isinstance(split, float):
        train_df, test_df = train_test_split(
            df,
            train_size=1.0 - split,
            random_state=seed,
            shuffle=True,
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=split,
            random_state=seed,
            shuffle=True,
        )

    if sample is not None:
        test_df = test_df.sample(n=sample, random_state=seed).copy(deep=True)
    else:
        test_df = test_df.copy(deep=True)
    train_df = train_df.copy(deep=True)

    train_set = dataset
    test_set = dataset.clone()
    train_set.update("train_set", True, df=train_df)
    test_set.update("test_set", True, df=test_df)
    return train_set, test_set


def _compute_model_metrics(model, test_set) -> dict[str, float]:
    probabilities = model.predict_proba(test_set).detach().cpu()
    prediction = probabilities.argmax(dim=1)

    y = test_set.get(target=True).iloc[:, 0]
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


def _build_filtered_test_set(test_set, keep_mask: pd.Series):
    filtered = test_set.clone()
    target_column = test_set.target_column
    combined = pd.concat([test_set.get(target=False), test_set.get(target=True)], axis=1)
    filtered_df = combined.loc[keep_mask].copy(deep=True)
    filtered_df = filtered_df.loc[
        :, [*test_set.get(target=False).columns, target_column]
    ]
    filtered.update("test_set", True, df=filtered_df)
    filtered.freeze()
    return filtered


def _select_recourse_factuals(model, test_set, desired_class: int | str | None):
    if desired_class is None:
        return test_set

    class_to_index = model.get_class_to_index()
    desired_index = class_to_index[desired_class]
    predictions = model.predict(test_set).argmax(dim=1).detach().cpu().numpy()
    keep_mask = pd.Series(
        predictions != desired_index, index=test_set.get(target=False).index
    )
    return _build_filtered_test_set(test_set, keep_mask)


def _run_current_experiment(experiment: Experiment) -> dict:
    train_set, test_set = _materialize_datasets(experiment)

    experiment._target_model.fit(train_set)
    model_metrics = _compute_model_metrics(experiment._target_model, test_set)

    experiment._method.fit(train_set)
    factuals = _select_recourse_factuals(
        experiment._target_model,
        test_set,
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
        "train_set": train_set,
        "test_set": test_set,
        "factuals": factuals,
        "counterfactuals": counterfactuals,
        "metrics": metrics,
        "model_metrics": model_metrics,
    }


def _run_future_model_experiment(experiment: Experiment) -> dict:
    train_set, test_set = _materialize_datasets(experiment)
    experiment._target_model.fit(train_set)
    model_metrics = _compute_model_metrics(experiment._target_model, test_set)
    return {
        "train_set": train_set,
        "test_set": test_set,
        "model_metrics": model_metrics,
    }


def _compute_future_validity(
    factuals, counterfactuals, future_model, future_test_set
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

    expected_columns = list(future_test_set.get(target=False).columns)
    actual_columns = list(counterfactual_features.columns)
    if len(actual_columns) != len(expected_columns):
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
    future_prediction = (
        future_model.get_prediction(future_features, proba=True).detach().cpu()
    )
    positive_index = future_model.get_class_to_index().get(1, 1)
    future_positive = future_prediction[:, positive_index] >= 0.5
    future_validity = float(
        future_positive.to(dtype=torch.float32).sum().item() / denominator
    )
    return future_validity, denominator, successful_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-config",
        default="./experiment/roar/german_mlp_roar_reproduce_current.yaml",
    )
    parser.add_argument(
        "--future-config",
        default="./experiment/roar/german_mlp_roar_reproduce_future.yaml",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    current_cfg = _apply_device(
        _load_config((PROJECT_ROOT / args.current_config).resolve()),
        device,
    )
    future_cfg = _apply_device(
        _load_config((PROJECT_ROOT / args.future_config).resolve()),
        device,
    )

    current_experiment = Experiment(current_cfg)
    current_results = _run_current_experiment(current_experiment)

    future_experiment = Experiment(future_cfg)
    future_results = _run_future_model_experiment(future_experiment)

    future_validity, num_factuals, num_successful = _compute_future_validity(
        current_results["factuals"],
        current_results["counterfactuals"],
        future_experiment._target_model,
        future_results["test_set"],
    )

    current_metrics = current_results["metrics"].iloc[0].to_dict()
    output = {
        "device": device,
        "num_factuals": num_factuals,
        "num_successful": num_successful,
        "current_validity": float(current_metrics.get("validity", float("nan"))),
        "future_validity": future_validity,
        "distance_l0": float(current_metrics.get("distance_l0", float("nan"))),
        "distance_l1": float(current_metrics.get("distance_l1", float("nan"))),
        "distance_l2": float(current_metrics.get("distance_l2", float("nan"))),
        "distance_linf": float(current_metrics.get("distance_linf", float("nan"))),
        "current_model_test_accuracy": float(
            current_results["model_metrics"]["test_accuracy"]
        ),
        "current_model_test_auc": float(current_results["model_metrics"]["test_auc"]),
        "future_model_test_accuracy": float(
            future_results["model_metrics"]["test_accuracy"]
        ),
        "future_model_test_auc": float(future_results["model_metrics"]["test_auc"]),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
