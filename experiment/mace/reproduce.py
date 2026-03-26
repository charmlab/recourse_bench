from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from evaluation.evaluation_utils import resolve_evaluation_inputs
from experiment import Experiment
from method.mace.library import normalizedDistance

DEFAULT_CONFIG_PATHS = {
    "adult": Path(__file__).with_name("adult_randomforest_mace_reproduce.yaml"),
    "credit": Path(__file__).with_name("credit_randomforest_mace_reproduce.yaml"),
    "compas": Path(__file__).with_name("compas_randomforest_mace_reproduce.yaml"),
}
EPSILON_SEQUENCE = [1.0e-1, 1.0e-3, 1.0e-5]
NORM_TO_METRIC = {
    "zero_norm": "distance_l0",
    "one_norm": "distance_l1",
    "infty_norm": "distance_linf",
}
MACE_DISTANCE_NORMS = {
    "distance_l0": "zero_norm",
    "distance_l1": "one_norm",
    "distance_linf": "infty_norm",
}


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _apply_overrides(
    config: dict,
    norm_type: str,
    epsilon: float,
    desired_class: int | str | None,
) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["device"] = "cpu"
    cfg["method"]["device"] = "cpu"
    cfg["method"]["norm_type"] = str(norm_type)
    cfg["method"]["epsilon"] = float(epsilon)
    cfg["method"]["desired_class"] = desired_class
    cfg["name"] = (
        f"{cfg['dataset']['name']}_randomforest_mace_"
        f"{str(norm_type).lower()}_{epsilon:.0e}".replace("+", "")
    )
    cfg["logger"]["path"] = f"./logs/{cfg['name']}.log"
    return cfg


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


def _subset_dataset(dataset, indices: pd.Index, flag_name: str):
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


def _select_factuals(dataset, target_model, desired_class, num_factuals: int):
    predictions = target_model.predict(dataset).argmax(dim=1).detach().cpu().numpy()
    class_to_index = target_model.get_class_to_index()

    if desired_class is None:
        if len(class_to_index) != 2:
            raise ValueError(
                "desired_class=None selection requires binary classification"
            )
        target_index = 1
    else:
        target_index = int(class_to_index[desired_class])

    keep_mask = predictions != target_index
    keep_index = dataset.get(target=False).index[keep_mask]
    if keep_index.shape[0] < num_factuals:
        raise ValueError(
            f"Requested {num_factuals} factuals but only found {keep_index.shape[0]}"
        )
    selected_index = keep_index[:num_factuals]
    return _subset_dataset(dataset, selected_index, "testset")


def _evaluate(experiment: Experiment, factuals, counterfactuals) -> pd.DataFrame:
    evaluation_results = [
        evaluation_step.evaluate(factuals, counterfactuals)
        for evaluation_step in experiment._evaluation
    ]
    return pd.concat(evaluation_results, axis=1)


def _to_builtin_scalar(value):
    if pd.isna(value):
        return float("nan")
    if hasattr(value, "item"):
        return value.item()
    return value


def _compute_mace_distances(experiment: Experiment, factuals, counterfactuals) -> dict[str, float]:
    (
        factual_features,
        counterfactual_features,
        evaluation_mask,
        success_mask,
    ) = resolve_evaluation_inputs(factuals, counterfactuals)
    selected_mask = evaluation_mask & success_mask

    if int(selected_mask.sum()) == 0:
        return {metric_name: float("nan") for metric_name in MACE_DISTANCE_NORMS}

    factual_success = factual_features.loc[selected_mask.to_numpy()]
    counterfactual_success = counterfactual_features.loc[selected_mask.to_numpy()]

    wrapper = experiment._method._dataset_wrapper
    factual_labels = experiment._method._predict_label(factual_success)
    counterfactual_labels = experiment._method._predict_label(counterfactual_success)

    distances_by_metric = {metric_name: [] for metric_name in MACE_DISTANCE_NORMS}
    for row_position, row_index in enumerate(factual_success.index):
        factual_sample = wrapper.factual_to_short_dict(
            factual_success.loc[row_index],
            int(factual_labels[row_position]),
        )
        counterfactual_sample = wrapper.factual_to_short_dict(
            counterfactual_success.loc[row_index],
            int(counterfactual_labels[row_position]),
        )
        for metric_name, norm_type in MACE_DISTANCE_NORMS.items():
            distances_by_metric[metric_name].append(
                float(
                    normalizedDistance.getDistanceBetweenSamples(
                        factual_sample,
                        counterfactual_sample,
                        norm_type,
                        wrapper,
                    )
                )
            )

    return {
        metric_name: float(sum(values) / len(values))
        for metric_name, values in distances_by_metric.items()
    }


def _run_single(
    config: dict,
    num_factuals: int,
) -> dict:
    experiment = Experiment(config)
    trainset, testset = _materialize_datasets(experiment)

    experiment._target_model.fit(trainset)
    factuals = _select_factuals(
        testset,
        experiment._target_model,
        getattr(experiment._method, "_desired_class", None),
        num_factuals,
    )

    experiment._method.fit(trainset)
    start_time = time.perf_counter()
    counterfactuals = experiment._method.predict(factuals)
    generation_seconds = time.perf_counter() - start_time
    metrics = _evaluate(experiment, factuals, counterfactuals)
    mace_distances = _compute_mace_distances(experiment, factuals, counterfactuals)

    validity = float(metrics.iloc[0]["validity"])
    if num_factuals != len(factuals):
        raise AssertionError("Selected factual count does not match requested count")
    if validity != 1.0:
        raise AssertionError(f"Expected validity == 1.0, received {validity}")

    metric_row = {
        key: _to_builtin_scalar(value)
        for key, value in metrics.iloc[0].to_dict().items()
    }
    metric_row.update(mace_distances)
    for key, value in metric_row.items():
        if key.startswith("distance_") and pd.isna(value):
            raise AssertionError(f"Distance metric {key} is NaN")

    return {
        "config_name": config["name"],
        "dataset": config["dataset"]["name"],
        "norm_type": config["method"]["norm_type"],
        "epsilon": float(config["method"]["epsilon"]),
        "num_factuals": int(len(factuals)),
        "generation_seconds": float(generation_seconds),
        "metrics": metric_row,
    }


def _assert_trend(results: list[dict]) -> None:
    if len(results) != len(EPSILON_SEQUENCE):
        raise AssertionError("Trend assertion requires exactly one result per epsilon")
    metric_name = NORM_TO_METRIC[results[0]["norm_type"]]
    by_epsilon = {float(result["epsilon"]): result for result in results}

    for epsilon in EPSILON_SEQUENCE:
        if epsilon not in by_epsilon:
            raise AssertionError(
                f"Missing epsilon={epsilon} result for trend assertion"
            )

    previous_distance: float | None = None
    previous_runtime: float | None = None
    for epsilon in EPSILON_SEQUENCE:
        current_distance = float(by_epsilon[epsilon]["metrics"][metric_name])
        current_runtime = float(by_epsilon[epsilon]["generation_seconds"])
        if (
            previous_distance is not None
            and current_distance > previous_distance + 1e-8
        ):
            raise AssertionError(
                f"{metric_name} must be non-increasing as epsilon decreases"
            )
        if previous_runtime is not None and current_runtime < previous_runtime - 1e-8:
            raise AssertionError(
                "generation_seconds must be non-decreasing as epsilon decreases"
            )
        previous_distance = current_distance
        previous_runtime = current_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DEFAULT_CONFIG_PATHS),
        default="adult",
    )
    parser.add_argument(
        "--norm",
        choices=sorted(NORM_TO_METRIC),
        default="zero_norm",
    )
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--desired-class", type=int, default=1)
    parser.add_argument("--num-factuals", type=int, default=20)
    parser.add_argument("--assert-trend", action="store_true")
    args = parser.parse_args()

    base_config = _load_config(DEFAULT_CONFIG_PATHS[args.dataset])

    if args.assert_trend:
        all_results = []
        for epsilon in EPSILON_SEQUENCE:
            config = _apply_overrides(
                base_config,
                norm_type=args.norm,
                epsilon=epsilon,
                desired_class=args.desired_class,
            )
            all_results.append(_run_single(config, args.num_factuals))
        _assert_trend(all_results)
        print(json.dumps(all_results, indent=2, sort_keys=True))
        return

    config = _apply_overrides(
        base_config,
        norm_type=args.norm,
        epsilon=args.epsilon,
        desired_class=args.desired_class,
    )
    result = _run_single(config, args.num_factuals)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
