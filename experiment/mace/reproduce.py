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

import numpy as np
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
    predictions = (
        target_model.predict(dataset, batch_size=max(1, len(dataset)))
        .argmax(dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    target_index = _resolve_target_index(target_model, desired_class)

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


def _resolve_target_index(target_model, desired_class) -> int:
    class_to_index = target_model.get_class_to_index()
    if desired_class is None:
        if len(class_to_index) != 2:
            raise ValueError(
                "desired_class=None selection requires binary classification"
            )
        return 1
    return int(class_to_index[desired_class])


def _to_builtin_scalar(value):
    if pd.isna(value):
        return float("nan")
    if hasattr(value, "item"):
        return value.item()
    return value


def _build_counterfactual_dataset(factuals, counterfactual_features, desired_class):
    target_column = factuals.target_column
    counterfactual_target = pd.DataFrame(
        -1.0,
        index=counterfactual_features.index,
        columns=[target_column],
    )
    counterfactual_df = pd.concat(
        [counterfactual_features, counterfactual_target], axis=1
    ).reindex(columns=factuals.ordered_features())

    output = factuals.clone()
    output.update("counterfactual", True, df=counterfactual_df)
    if desired_class is not None:
        evaluation_filter = pd.DataFrame(
            True,
            index=counterfactual_df.index,
            columns=["evaluation_filter"],
            dtype=bool,
        )
        output.update("evaluation_filter", evaluation_filter)
    output.freeze()
    return output


def _select_observable_pool(dataset, target_model, desired_class):
    predictions = target_model.predict(dataset).argmax(dim=1).detach().cpu().numpy()
    target_index = _resolve_target_index(target_model, desired_class)
    pool_index = dataset.get(target=False).index[predictions == target_index]
    observable_pool = dataset.get(target=False).loc[pool_index].copy(deep=True)
    if observable_pool.shape[0] == 0:
        raise ValueError("Observable pool is empty for MO baseline")
    return observable_pool


def _violates_actionability(wrapper, factual_sample, observable_sample) -> bool:
    for attr_name_kurz in wrapper.getInputAttributeNames("kurz"):
        attr_obj = wrapper.attributes_kurz[attr_name_kurz]
        factual_value = factual_sample[attr_name_kurz]
        observable_value = observable_sample[attr_name_kurz]
        if attr_obj.actionability == "none" and factual_value != observable_value:
            return True
        if (
            attr_obj.actionability == "same-or-increase"
            and factual_value > observable_value
        ):
            return True
        if (
            attr_obj.actionability == "same-or-decrease"
            and factual_value < observable_value
        ):
            return True
    return False


def _compute_mo_counterfactuals(
    experiment: Experiment, factuals, observable_pool: pd.DataFrame
):
    wrapper = experiment._method._dataset_wrapper
    factual_features = factuals.get(target=False)
    factual_labels = experiment._method._predict_label(factual_features)
    observable_labels = experiment._method._predict_label(observable_pool)

    observable_samples = []
    for row_position, row_index in enumerate(observable_pool.index):
        observable_samples.append(
            wrapper.factual_to_short_dict(
                observable_pool.loc[row_index],
                int(observable_labels[row_position]),
            )
        )

    rows = []
    for row_position, row_index in enumerate(factual_features.index):
        factual_sample = wrapper.factual_to_short_dict(
            factual_features.loc[row_index],
            int(factual_labels[row_position]),
        )
        best_sample = None
        best_distance = float("inf")
        norm_type = str(experiment._method._norm_type[0])

        for observable_sample in observable_samples:
            if observable_sample["y"] == factual_sample["y"]:
                continue
            if _violates_actionability(wrapper, factual_sample, observable_sample):
                continue

            candidate_distance = float(
                normalizedDistance.getDistanceBetweenSamples(
                    factual_sample,
                    observable_sample,
                    norm_type,
                    wrapper,
                )
            )
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_sample = observable_sample

        if best_sample is None:
            rows.append(
                pd.Series(np.nan, index=wrapper._feature_names, dtype="float64")
            )
        else:
            rows.append(wrapper.short_dict_to_feature_row(best_sample))

    counterfactual_features = pd.DataFrame(
        rows,
        index=factual_features.index,
        columns=factual_features.columns,
    )
    return _build_counterfactual_dataset(
        factuals,
        counterfactual_features,
        getattr(experiment._method, "_desired_class", None),
    )


def _compute_distance_summary(experiment: Experiment, factuals, counterfactuals):
    (
        factual_features,
        counterfactual_features,
        evaluation_mask,
        success_mask,
    ) = resolve_evaluation_inputs(factuals, counterfactuals)
    selected_mask = evaluation_mask & success_mask

    if int(selected_mask.sum()) == 0:
        empty_metrics = {
            metric_name: float("nan") for metric_name in MACE_DISTANCE_NORMS
        }
        empty_rows = {metric_name: [] for metric_name in MACE_DISTANCE_NORMS}
        return empty_metrics, empty_rows

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

    summary = {
        metric_name: float(sum(values) / len(values))
        for metric_name, values in distances_by_metric.items()
    }
    return summary, distances_by_metric


def _compute_mace_vs_mo_comparison(
    norm_type: str,
    epsilon: float,
    mace_distances: dict[str, float],
    mo_distances: dict[str, float],
    mace_pointwise: dict[str, list[float]],
    mo_pointwise: dict[str, list[float]],
) -> dict[str, float | str]:
    optimized_metric = NORM_TO_METRIC[str(norm_type)]
    mace_distance = float(mace_distances[optimized_metric])
    mo_distance = float(mo_distances[optimized_metric])

    improvement_terms = []
    for mace_value, mo_value in zip(
        mace_pointwise[optimized_metric], mo_pointwise[optimized_metric]
    ):
        if mo_value <= 0:
            continue
        improvement_terms.append(1.0 - float(mace_value) / float(mo_value))

    if improvement_terms:
        improvement = 100.0 * float(sum(improvement_terms) / len(improvement_terms))
    else:
        improvement = float("nan")

    return {
        "optimized_metric": optimized_metric,
        "mace_distance": mace_distance,
        "mo_distance": mo_distance,
        "mace_vs_mo_improvement": improvement,
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
    counterfactuals = experiment._method.predict(
        factuals,
        batch_size=max(1, len(factuals)),
    )
    generation_seconds = time.perf_counter() - start_time
    metrics = _evaluate(experiment, factuals, counterfactuals)
    mace_distances, mace_pointwise = _compute_distance_summary(
        experiment,
        factuals,
        counterfactuals,
    )

    observable_pool = _select_observable_pool(
        testset,
        experiment._target_model,
        getattr(experiment._method, "_desired_class", None),
    )
    mo_start_time = time.perf_counter()
    mo_counterfactuals = _compute_mo_counterfactuals(
        experiment, factuals, observable_pool
    )
    mo_generation_seconds = time.perf_counter() - mo_start_time
    mo_metrics_df = _evaluate(experiment, factuals, mo_counterfactuals)
    mo_distances, mo_pointwise = _compute_distance_summary(
        experiment,
        factuals,
        mo_counterfactuals,
    )
    comparison = _compute_mace_vs_mo_comparison(
        norm_type=str(config["method"]["norm_type"]),
        epsilon=float(config["method"]["epsilon"]),
        mace_distances=mace_distances,
        mo_distances=mo_distances,
        mace_pointwise=mace_pointwise,
        mo_pointwise=mo_pointwise,
    )

    metric_row = {
        key: _to_builtin_scalar(value)
        for key, value in metrics.iloc[0].to_dict().items()
    }
    metric_row.update(mace_distances)

    mo_metric_row = {
        key: _to_builtin_scalar(value)
        for key, value in mo_metrics_df.iloc[0].to_dict().items()
    }
    mo_metric_row.update(mo_distances)

    return {
        "config_name": config["name"],
        "dataset": config["dataset"]["name"],
        "norm_type": config["method"]["norm_type"],
        "epsilon": float(config["method"]["epsilon"]),
        "num_factuals": int(len(factuals)),
        "generation_seconds": float(generation_seconds),
        "metrics": metric_row,
        "mo_generation_seconds": float(mo_generation_seconds),
        "mo_metrics": mo_metric_row,
        "comparison": comparison,
    }


TABLE3_FOREST_MO_REFERENCES = {
    1.0e-5: {
        "adult": {"zero_norm": 51.0, "one_norm": 82.0, "infty_norm": 71.0},
        "credit": {"zero_norm": 68.0, "one_norm": 97.0, "infty_norm": 96.0},
        "compas": {"zero_norm": 1.0, "one_norm": 6.0, "infty_norm": 6.0},
    },
    1.0e-3: {
        "adult": {"zero_norm": 51.0, "one_norm": 81.0, "infty_norm": 69.0},
        "credit": {"zero_norm": 68.0, "one_norm": 61.0, "infty_norm": 38.0},
        "compas": {"zero_norm": 1.0, "one_norm": 6.0, "infty_norm": 6.0},
    },
}


def _paper_reference_improvement(
    dataset: str,
    norm_type: str,
    epsilon: float,
) -> float | None:
    epsilon_key = float(epsilon)
    if epsilon_key not in TABLE3_FOREST_MO_REFERENCES:
        return None
    dataset_refs = TABLE3_FOREST_MO_REFERENCES[epsilon_key]
    if dataset not in dataset_refs:
        return None
    return dataset_refs[dataset].get(norm_type)


def _assert_result(result: dict, expected_num_factuals: int, epsilon: float) -> None:
    if int(result["num_factuals"]) != int(expected_num_factuals):
        raise AssertionError(
            "Selected factual count does not match requested count: "
            f"{result['num_factuals']} vs {expected_num_factuals}"
        )

    metrics = result["metrics"]
    mo_metrics = result["mo_metrics"]
    comparison = result["comparison"]

    if float(metrics["validity"]) != 1.0:
        raise AssertionError(f"Expected MACE validity == 1.0, got {metrics['validity']}")
    if float(mo_metrics["validity"]) != 1.0:
        raise AssertionError(f"Expected MO validity == 1.0, got {mo_metrics['validity']}")

    for name, value in metrics.items():
        if name.startswith("distance_") and pd.isna(value):
            raise AssertionError(f"MACE metric {name} is NaN")
    for name, value in mo_metrics.items():
        if name.startswith("distance_") and pd.isna(value):
            raise AssertionError(f"MO metric {name} is NaN")

    if float(comparison["mace_distance"]) > float(comparison["mo_distance"]) + float(
        epsilon
    ) + 1e-8:
        raise AssertionError(
            f"{comparison['optimized_metric']} expected MACE <= MO + epsilon, got "
            f"{comparison['mace_distance']} vs {comparison['mo_distance']}"
        )

    improvement = comparison["mace_vs_mo_improvement"]
    if not pd.isna(improvement) and float(improvement) < -1e-8:
        raise AssertionError(
            "MACE vs MO improvement must be >= 0, "
            f"got {comparison['mace_vs_mo_improvement']}"
        )


def _iter_jobs(
    datasets: list[str],
    norms: list[str],
    epsilon: float,
    desired_class: int | str | None,
    num_factuals: int,
):
    for dataset in datasets:
        base_config = _load_config(DEFAULT_CONFIG_PATHS[dataset])
        for norm_type in norms:
            yield {
                "dataset": dataset,
                "norm_type": norm_type,
                "epsilon": float(epsilon),
                "desired_class": desired_class,
                "num_factuals": int(num_factuals),
                "config": _apply_overrides(
                    base_config,
                    norm_type=norm_type,
                    epsilon=epsilon,
                    desired_class=desired_class,
                ),
            }


def _build_summary(results: list[dict], errors: list[dict]) -> dict:
    summary_table = []
    successful_results = []
    for result in results:
        assert_passed = bool(result.get("assert_passed", False))
        paper_ref = _paper_reference_improvement(
            dataset=str(result["dataset"]),
            norm_type=str(result["norm_type"]),
            epsilon=float(result["epsilon"]),
        )
        summary_table.append(
            {
                "dataset": result["dataset"],
                "norm_type": result["norm_type"],
                "epsilon": result["epsilon"],
                "num_factuals": result["num_factuals"],
                "mace_validity": result["metrics"]["validity"],
                "mo_validity": result["mo_metrics"]["validity"],
                "optimized_metric": result["comparison"]["optimized_metric"],
                "mace_distance": result["comparison"]["mace_distance"],
                "mo_distance": result["comparison"]["mo_distance"],
                "mace_vs_mo_improvement": result["comparison"][
                    "mace_vs_mo_improvement"
                ],
                "paper_improvement_reference": paper_ref,
                "assert_passed": assert_passed,
                "assert_error": result.get("assert_error"),
            }
        )
        if assert_passed:
            successful_results.append(result)

    aggregate: dict[str, object] = {
        "num_jobs": len(results) + len(errors),
        "num_results": len(results),
        "num_errors": len(errors),
        "num_assert_passed": sum(1 for result in results if result.get("assert_passed")),
    }

    if successful_results:
        aggregate["overall"] = {
            "mean_mace_distance": float(
                sum(
                    float(result["comparison"]["mace_distance"])
                    for result in successful_results
                )
                / len(successful_results)
            ),
            "mean_mo_distance": float(
                sum(
                    float(result["comparison"]["mo_distance"])
                    for result in successful_results
                )
                / len(successful_results)
            ),
            "mean_mace_vs_mo_improvement": float(
                sum(
                    float(result["comparison"]["mace_vs_mo_improvement"])
                    for result in successful_results
                    if not pd.isna(result["comparison"]["mace_vs_mo_improvement"])
                )
                / max(
                    1,
                    sum(
                        1
                        for result in successful_results
                        if not pd.isna(result["comparison"]["mace_vs_mo_improvement"])
                    ),
                )
            ),
        }

        per_dataset = {}
        for dataset in sorted({result["dataset"] for result in successful_results}):
            dataset_results = [
                result
                for result in successful_results
                if result["dataset"] == dataset
            ]
            per_dataset[dataset] = {
                "mean_mace_distance": float(
                    sum(float(item["comparison"]["mace_distance"]) for item in dataset_results)
                    / len(dataset_results)
                ),
                "mean_mo_distance": float(
                    sum(float(item["comparison"]["mo_distance"]) for item in dataset_results)
                    / len(dataset_results)
                ),
            }
        aggregate["per_dataset"] = per_dataset

    return {
        "summary_table": summary_table,
        "aggregate": aggregate,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DEFAULT_CONFIG_PATHS), action="append")
    parser.add_argument("--datasets", choices=sorted(DEFAULT_CONFIG_PATHS), nargs="+")
    parser.add_argument("--norm", choices=sorted(NORM_TO_METRIC), action="append")
    parser.add_argument("--norms", choices=sorted(NORM_TO_METRIC), nargs="+")
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--desired-class", type=int, default=1)
    parser.add_argument("--num-factuals", type=int, default=500)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--continue-on-error", dest="continue_on_error", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.set_defaults(continue_on_error=True)
    args = parser.parse_args()

    datasets = args.datasets or args.dataset or list(DEFAULT_CONFIG_PATHS)
    norms = args.norms or args.norm or list(NORM_TO_METRIC)

    results: list[dict] = []
    errors: list[dict] = []

    for job in _iter_jobs(
        datasets=datasets,
        norms=norms,
        epsilon=args.epsilon,
        desired_class=args.desired_class,
        num_factuals=args.num_factuals,
    ):
        try:
            result = _run_single(job["config"], job["num_factuals"])
            try:
                _assert_result(result, job["num_factuals"], job["epsilon"])
                result["assert_passed"] = True
                result["assert_error"] = None
            except Exception as error:
                result["assert_passed"] = False
                result["assert_error"] = str(error)
                if args.strict:
                    raise
            results.append(result)
        except Exception as error:
            error_record = {
                "dataset": job["dataset"],
                "norm_type": job["norm_type"],
                "epsilon": job["epsilon"],
                "num_factuals": job["num_factuals"],
                "error_message": str(error),
            }
            errors.append(error_record)
            if args.strict:
                payload = {
                    "results": results,
                    **_build_summary(results, errors),
                }
                rendered = json.dumps(payload, indent=2, sort_keys=True)
                if args.output_json:
                    Path(args.output_json).write_text(rendered, encoding="utf-8")
                print(rendered)
                raise SystemExit(1)
            if not args.continue_on_error:
                payload = {
                    "results": results,
                    **_build_summary(results, errors),
                }
                rendered = json.dumps(payload, indent=2, sort_keys=True)
                if args.output_json:
                    Path(args.output_json).write_text(rendered, encoding="utf-8")
                print(rendered)
                raise SystemExit(1)

    payload = {
        "results": results,
        **_build_summary(results, errors),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json:
        Path(args.output_json).write_text(rendered, encoding="utf-8")
    print(rendered)

    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
