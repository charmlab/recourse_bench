from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment import Experiment

CONFIG_ROOT = PROJECT_ROOT / "benchmark" / "configs"
DEFAULT_SUITE_PATH = CONFIG_ROOT / "suites" / "default.yml"
CPU_ONLY_MODELS = {"randomforest", "sklearn_logistic_regression"}
RESERVED_SUITE_KEYS = {"base_config", "datasets", "methods", "models", "name", "output_csv"}
DEFAULT_PREDICTION_BATCH_SIZE = 512
DEFAULT_METHOD_BATCH_SIZE = 20
DEFAULT_WRITE_MODE = "replace-overlaps"
VALID_WRITE_MODES = {"append", "replace-overlaps", "rewrite-all"}
SUPPORTED_MODEL_CACHE_EXTENSIONS = {
    "linear": "pt",
    "mlp": "pt",
    "mlp_bayesian": "pkl",
}


def _load_yaml_dict(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config at {path} must parse to a dictionary")
    return loaded


def _deep_merge(base: object, override: object) -> object:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = deepcopy(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged
    return deepcopy(override)


def _resolve_component_path(component_type: str, name: str) -> Path:
    path = CONFIG_ROOT / component_type / f"{name}.yml"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {component_type.rstrip('s')} config for '{name}': {path}"
        )
    return path


def _resolve_device(model_name: str, requested_device: str | None) -> str:
    requested = str(requested_device or "auto").lower()
    if model_name in CPU_ONLY_MODELS:
        if requested not in {"auto", "cpu"}:
            raise ValueError(f"Model '{model_name}' only supports cpu, received {requested}")
        return "cpu"

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    return requested


def _subset_dataset(dataset, indices: pd.Index) -> object:
    index = pd.Index(indices)
    subset_df = pd.concat(
        [dataset.get(target=False).loc[index], dataset.get(target=True).loc[index]],
        axis=1,
    ).reindex(columns=dataset.ordered_features())

    subset = dataset.clone()
    subset.update("benchmark_subset", True, df=subset_df)

    if hasattr(dataset, "evaluation_filter"):
        evaluation_filter = dataset.attr("evaluation_filter")
        if isinstance(evaluation_filter, pd.Series):
            subset.update("evaluation_filter", evaluation_filter.loc[index].copy(deep=True))
        elif isinstance(evaluation_filter, pd.DataFrame):
            subset.update("evaluation_filter", evaluation_filter.loc[index].copy(deep=True))

    subset.freeze()
    return subset


def _materialize_train_test(experiment: Experiment):
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


def _normalize_names(values: list[str], label: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"Suite field '{label}' must be a non-empty list[str]")
    resolved = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"Suite field '{label}' must contain str values only")
        resolved.append(value)
    return resolved


def _parse_name_filter(raw_values: list[str] | None) -> list[str] | None:
    if raw_values is None:
        return None

    resolved: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for token in str(raw_value).split(","):
            name = token.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            resolved.append(name)
    return resolved or None


def _resolve_selected_names(
    available_names: list[str],
    selected_names: list[str] | None,
    label: str,
) -> list[str]:
    if selected_names is None:
        return available_names

    available_set = set(available_names)
    invalid = [name for name in selected_names if name not in available_set]
    if invalid:
        raise ValueError(
            f"Unknown {label} filter values {invalid}; suite allows {available_names}"
        )

    resolved = [name for name in available_names if name in set(selected_names)]
    if not resolved:
        raise ValueError(f"{label} filter removed all planned runs")
    return resolved


def _suite_overrides(suite_cfg: dict) -> dict:
    return {
        key: deepcopy(value)
        for key, value in suite_cfg.items()
        if key not in RESERVED_SUITE_KEYS
    }


def _resolve_benchmark_cfg(config: dict) -> dict:
    benchmark_cfg = deepcopy(config.get("benchmark", {}))
    benchmark_cfg.setdefault("desired_class", 1)
    benchmark_cfg.setdefault("sample_target", 50)
    benchmark_cfg.setdefault("min_factuals", 25)
    benchmark_cfg.setdefault("allow_partial_factuals", True)
    benchmark_cfg.setdefault("sample_seed", 0)
    benchmark_cfg.setdefault("factual_filter", "prediction_not_equal_desired_class")
    benchmark_cfg.setdefault("prediction_batch_size", DEFAULT_PREDICTION_BATCH_SIZE)
    benchmark_cfg.setdefault("method_batch_size", DEFAULT_METHOD_BATCH_SIZE)
    benchmark_cfg.setdefault("continue_on_error", True)
    model_cache_cfg = deepcopy(benchmark_cfg.get("model_cache", {}))
    model_cache_cfg.setdefault("enabled", True)
    model_cache_cfg.setdefault("namespace", "benchmark")
    benchmark_cfg["model_cache"] = model_cache_cfg
    return benchmark_cfg


def _resolve_method_benchmark_cfg(config: dict) -> dict:
    method_benchmark_cfg = deepcopy(config.get("method_benchmark", {}))
    method_benchmark_cfg.setdefault("use_dataset_desired_class", True)
    method_benchmark_cfg.setdefault("factual_filter", None)
    return method_benchmark_cfg


def _remove_nested_keys(payload: object, drop_keys: set[str]) -> object:
    if isinstance(payload, dict):
        return {
            key: _remove_nested_keys(value, drop_keys)
            for key, value in payload.items()
            if key not in drop_keys
        }
    if isinstance(payload, list):
        return [_remove_nested_keys(item, drop_keys) for item in payload]
    return payload


def _benchmark_model_cache_payload(config: dict) -> dict:
    model_payload = _remove_nested_keys(
        deepcopy(config["model"]),
        {"device", "pretrained_path", "save_name"},
    )
    return {
        "dataset": deepcopy(config["dataset"]),
        "preprocess": deepcopy(config.get("preprocess", [])),
        "model": model_payload,
    }


def _benchmark_model_cache_save_name(config: dict) -> str | None:
    benchmark_cfg = config.get("benchmark", {})
    model_cache_cfg = benchmark_cfg.get("model_cache", {})
    if not bool(model_cache_cfg.get("enabled", True)):
        return None

    model_name = str(config["model"]["name"])
    if model_name not in SUPPORTED_MODEL_CACHE_EXTENSIONS:
        return None

    explicit_pretrained = config["model"].get("pretrained_path")
    if explicit_pretrained:
        return None

    explicit_save_name = config["model"].get("save_name")
    if explicit_save_name:
        return str(explicit_save_name)

    payload = _benchmark_model_cache_payload(config)
    payload_bytes = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload_bytes).hexdigest()[:16]
    namespace = str(model_cache_cfg.get("namespace", "benchmark"))
    return f"{namespace}__{model_name}__{digest}"


def _apply_model_cache_config(run_cfg: dict) -> None:
    save_name = _benchmark_model_cache_save_name(run_cfg)
    if save_name is None:
        return

    model_name = str(run_cfg["model"]["name"])
    extension = SUPPORTED_MODEL_CACHE_EXTENSIONS[model_name]
    cache_root = Path(run_cfg.get("caching", {}).get("path", "./cache/")) / "models"

    run_cfg["model"]["save_name"] = save_name
    run_cfg["model"]["pretrained_path"] = (
        cache_root / f"{save_name}.{extension}"
    ).as_posix()


def _is_compatible(dataset_name: str, model_name: str, method_cfg: dict) -> bool:
    compatibility = method_cfg.get("compatibility", {})
    allowed_models = compatibility.get("allowed_models")
    allowed_datasets = compatibility.get("allowed_datasets")

    if allowed_models is not None and model_name not in set(allowed_models):
        return False
    if allowed_datasets is not None and dataset_name not in set(allowed_datasets):
        return False
    return True


def _build_run_config(
    *,
    base_cfg: dict,
    dataset_cfg: dict,
    model_cfg: dict,
    method_cfg: dict,
    suite_cfg: dict,
    dataset_name: str,
    model_name: str,
    method_name: str,
) -> dict:
    run_cfg = deepcopy(base_cfg)
    for component_cfg in (dataset_cfg, model_cfg, method_cfg, suite_cfg):
        run_cfg = _deep_merge(run_cfg, component_cfg)

    run_cfg["name"] = f"benchmark_{dataset_name}_{model_name}_{method_name}"
    run_cfg["benchmark"] = _resolve_benchmark_cfg(run_cfg)
    run_cfg["method_benchmark"] = _resolve_method_benchmark_cfg(run_cfg)

    if run_cfg["method_benchmark"]["factual_filter"] is not None:
        run_cfg["benchmark"]["factual_filter"] = run_cfg["method_benchmark"]["factual_filter"]

    if run_cfg["method_benchmark"]["use_dataset_desired_class"]:
        run_cfg.setdefault("method", {})
        run_cfg["method"]["desired_class"] = run_cfg["benchmark"]["desired_class"]

    resolved_device = _resolve_device(
        run_cfg["model"]["name"],
        run_cfg["model"].get("device", "auto"),
    )
    run_cfg["model"]["device"] = resolved_device
    run_cfg["method"]["device"] = resolved_device
    _apply_model_cache_config(run_cfg)
    return run_cfg


def _plan_runs(
    suite_path: Path,
    *,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    methods: list[str] | None = None,
) -> tuple[list[tuple[str, dict]], Path]:
    suite_cfg = _load_yaml_dict(suite_path)
    base_config_name = str(suite_cfg.get("base_config", "base.yml"))
    base_cfg = _load_yaml_dict(CONFIG_ROOT / base_config_name)
    suite_cfg_overrides = _suite_overrides(suite_cfg)

    dataset_names = _resolve_selected_names(
        _normalize_names(suite_cfg.get("datasets", []), "datasets"),
        datasets,
        "dataset",
    )
    model_names = _resolve_selected_names(
        _normalize_names(suite_cfg.get("models", []), "models"),
        models,
        "model",
    )
    method_names = _resolve_selected_names(
        _normalize_names(suite_cfg.get("methods", []), "methods"),
        methods,
        "method",
    )

    dataset_cfgs = {
        name: _load_yaml_dict(_resolve_component_path("datasets", name))
        for name in dataset_names
    }
    model_cfgs = {
        name: _load_yaml_dict(_resolve_component_path("models", name))
        for name in model_names
    }
    method_cfgs = {
        name: _load_yaml_dict(_resolve_component_path("methods", name))
        for name in method_names
    }

    planned_runs: list[tuple[str, dict]] = []
    for dataset_name in dataset_names:
        for model_name in model_names:
            for method_name in method_names:
                method_cfg = method_cfgs[method_name]
                if not _is_compatible(dataset_name, model_name, method_cfg):
                    continue
                run_id = f"{dataset_name}__{model_name}__{method_name}"
                run_cfg = _build_run_config(
                    base_cfg=base_cfg,
                    dataset_cfg=dataset_cfgs[dataset_name],
                    model_cfg=model_cfgs[model_name],
                    method_cfg=method_cfg,
                    suite_cfg=suite_cfg_overrides,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    method_name=method_name,
                )
                planned_runs.append((run_id, run_cfg))

    output_path = Path(str(suite_cfg.get("output_csv", base_cfg.get("output_csv", "./benchmark/results/results.csv"))))
    return planned_runs, output_path


def _resolve_eligible_indices(testset, target_model, benchmark_cfg: dict) -> pd.Index:
    feature_df = testset.get(target=False)
    factual_filter = str(benchmark_cfg["factual_filter"]).lower()
    if factual_filter == "all_test":
        return pd.Index(feature_df.index)

    if factual_filter != "prediction_not_equal_desired_class":
        raise ValueError(f"Unsupported factual_filter: {benchmark_cfg['factual_filter']}")

    class_to_index = target_model.get_class_to_index()
    desired_class = benchmark_cfg["desired_class"]
    if desired_class not in class_to_index:
        raise ValueError(f"desired_class '{desired_class}' is invalid for the trained model")

    prediction_batch_size = int(benchmark_cfg["prediction_batch_size"])
    predictions = (
        target_model.predict(testset, batch_size=prediction_batch_size)
        .argmax(dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    desired_index = int(class_to_index[desired_class])
    return pd.Index(feature_df.index[predictions != desired_index])


def _sample_factuals(testset, target_model, benchmark_cfg: dict) -> tuple[object | None, dict]:
    eligible_indices = _resolve_eligible_indices(testset, target_model, benchmark_cfg)
    eligible_count = int(eligible_indices.shape[0])
    sample_target = int(benchmark_cfg["sample_target"])
    min_factuals = int(benchmark_cfg["min_factuals"])
    allow_partial_factuals = bool(benchmark_cfg.get("allow_partial_factuals", True))
    sampled_count = min(sample_target, eligible_count)

    metadata = {
        "requested_factual_count": sample_target,
        "eligible_factual_count": eligible_count,
        "actual_factual_count": sampled_count,
    }

    if eligible_count == 0:
        metadata["actual_factual_count"] = 0
        return None, metadata

    if eligible_count < min_factuals and not allow_partial_factuals:
        metadata["actual_factual_count"] = 0
        return None, metadata

    feature_df = testset.get(target=False).loc[eligible_indices]
    if sampled_count < eligible_count:
        sampled_features = feature_df.sample(
            n=sampled_count,
            random_state=int(benchmark_cfg["sample_seed"]),
        )
        sampled_indices = pd.Index(sampled_features.index)
    else:
        sampled_indices = pd.Index(feature_df.index)

    sampled_dataset = _subset_dataset(testset, sampled_indices)
    return sampled_dataset, metadata


def _metric_means(metric_rows: list[dict[str, object]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    metrics_df = pd.DataFrame(metric_rows).apply(pd.to_numeric, errors="coerce")
    mean_series = metrics_df.mean(axis=0, skipna=True)
    return {
        column: (float(value) if pd.notna(value) else float("nan"))
        for column, value in mean_series.items()
    }


def _detail_rows(
    *,
    run_id: str,
    config: dict,
    factuals,
    counterfactuals,
    evaluation_steps: list,
    sampling_metadata: dict,
) -> tuple[list[dict[str, object]], dict[str, float], int]:
    detail_records: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    counterfactual_features = counterfactuals.get(target=False)
    success_mask = ~counterfactual_features.isna().any(axis=1)

    for rank, factual_index in enumerate(factuals.get(target=False).index, start=1):
        single_index = pd.Index([factual_index])
        single_factual = _subset_dataset(factuals, single_index)
        single_counterfactual = _subset_dataset(counterfactuals, single_index)
        metric_row = pd.concat(
            [
                evaluation_step.evaluate(single_factual, single_counterfactual)
                for evaluation_step in evaluation_steps
            ],
            axis=1,
        ).iloc[0].to_dict()
        metric_rows.append(metric_row)
        detail_records.append(
            {
                "run_id": run_id,
                "row_type": "detail",
                "status": "completed",
                "dataset_name": config["dataset"]["name"],
                "model_name": config["model"]["name"],
                "method_name": config["method"]["name"],
                "desired_class": config["benchmark"]["desired_class"],
                "factual_filter": config["benchmark"]["factual_filter"],
                "requested_factual_count": sampling_metadata["requested_factual_count"],
                "eligible_factual_count": sampling_metadata["eligible_factual_count"],
                "actual_factual_count": sampling_metadata["actual_factual_count"],
                "factual_rank": rank,
                "factual_index": factual_index,
                "counterfactual_found": bool(success_mask.loc[factual_index]),
                **metric_row,
            }
        )

    return detail_records, _metric_means(metric_rows), int(success_mask.sum())


def _summary_record(
    *,
    run_id: str,
    config: dict,
    status: str,
    sampling_metadata: dict,
    successful_counterfactual_count: int = 0,
    factual_index: object = None,
    error_message: str | None = None,
    metrics: dict[str, float] | None = None,
    run_duration_seconds: float | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "run_id": run_id,
        "row_type": "summary",
        "status": status,
        "dataset_name": config["dataset"]["name"],
        "model_name": config["model"]["name"],
        "method_name": config["method"]["name"],
        "desired_class": config["benchmark"]["desired_class"],
        "factual_filter": config["benchmark"]["factual_filter"],
        "requested_factual_count": sampling_metadata.get("requested_factual_count", 0),
        "eligible_factual_count": sampling_metadata.get("eligible_factual_count", 0),
        "actual_factual_count": sampling_metadata.get("actual_factual_count", 0),
        "successful_counterfactual_count": successful_counterfactual_count,
        "factual_index": factual_index,
        "error_message": error_message,
        "run_duration_seconds": run_duration_seconds,
    }
    if metrics:
        record.update(metrics)
    return record


def _execute_run(run_id: str, config: dict) -> list[dict[str, object]]:
    sampling_metadata = {
        "requested_factual_count": int(config["benchmark"]["sample_target"]),
        "eligible_factual_count": 0,
        "actual_factual_count": 0,
    }
    start_time = time.monotonic()
    logger = logging.getLogger(__name__)

    try:
        experiment = Experiment(config)
        logger = experiment._logger
        logger.info("Starting benchmark run: %s", run_id)

        trainset, testset = _materialize_train_test(experiment)
        logger.info("Resolved train/test sizes: %d / %d", len(trainset), len(testset))

        model_pretrained_path = experiment.get_config()["model"].get("pretrained_path")
        if model_pretrained_path:
            model_cache_path = Path(str(model_pretrained_path))
            if model_cache_path.exists():
                logger.info("Using cached target model from %s", model_cache_path)
            else:
                logger.info("Target model cache miss; will train and save to %s", model_cache_path)

        experiment._target_model.fit(trainset)
        experiment._method.fit(trainset)

        factuals, sampling_metadata = _sample_factuals(
            testset,
            experiment._target_model,
            config["benchmark"],
        )
        if factuals is None:
            logger.warning(
                "Skipping run %s due to zero eligible factuals",
                run_id,
            )
            duration = round(time.monotonic() - start_time, 6)
            return [
                _summary_record(
                    run_id=run_id,
                    config=config,
                    status="skipped_insufficient_factuals",
                    sampling_metadata=sampling_metadata,
                    run_duration_seconds=duration,
                )
            ]
        if sampling_metadata["eligible_factual_count"] < int(config["benchmark"]["min_factuals"]):
            logger.warning(
                "Proceeding with partial factual set for run %s (%d < %d)",
                run_id,
                sampling_metadata["eligible_factual_count"],
                int(config["benchmark"]["min_factuals"]),
            )

        method_batch_size = int(config["benchmark"]["method_batch_size"])
        counterfactuals = experiment._method.predict(factuals, batch_size=method_batch_size)
        detail_records, summary_metrics, success_count = _detail_rows(
            run_id=run_id,
            config=config,
            factuals=factuals,
            counterfactuals=counterfactuals,
            evaluation_steps=experiment._evaluation,
            sampling_metadata=sampling_metadata,
        )
        duration = round(time.monotonic() - start_time, 6)
        detail_records.append(
            _summary_record(
                run_id=run_id,
                config=config,
                status="completed",
                sampling_metadata=sampling_metadata,
                successful_counterfactual_count=success_count,
                metrics=summary_metrics,
                run_duration_seconds=duration,
            )
        )
        logger.info(
            "Completed benchmark run %s with %d factuals and %d successful counterfactuals",
            run_id,
            sampling_metadata["actual_factual_count"],
            success_count,
        )
        return detail_records
    except Exception as error:
        duration = round(time.monotonic() - start_time, 6)
        logger.exception("Benchmark run failed: %s", run_id)
        should_raise = not bool(config.get("benchmark", {}).get("continue_on_error", True))
        failed_record = _summary_record(
            run_id=run_id,
            config=config,
            status="failed",
            sampling_metadata=sampling_metadata,
            error_message=f"{error.__class__.__name__}: {error}",
            run_duration_seconds=duration,
        )
        if should_raise:
            raise
        return [failed_record]


def _load_existing_results(output_path: Path) -> pd.DataFrame:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(output_path)


def _merge_results(
    *,
    existing_results: pd.DataFrame,
    new_results: pd.DataFrame,
    run_ids: list[str],
    write_mode: str,
) -> pd.DataFrame:
    if write_mode not in VALID_WRITE_MODES:
        raise ValueError(
            f"Unsupported write_mode '{write_mode}', expected one of {sorted(VALID_WRITE_MODES)}"
        )

    if write_mode == "rewrite-all" or existing_results.empty:
        return new_results.copy(deep=True)

    if write_mode == "append":
        return pd.concat([existing_results, new_results], ignore_index=True, sort=False)

    if "run_id" not in existing_results.columns:
        raise ValueError(
            "Existing output CSV must contain a 'run_id' column for replace-overlaps mode"
        )

    preserved = existing_results.loc[
        ~existing_results["run_id"].astype(str).isin(set(run_ids))
    ].copy(deep=True)
    return pd.concat([preserved, new_results], ignore_index=True, sort=False)


def run_benchmarks(
    suite_path: str | Path = DEFAULT_SUITE_PATH,
    output_csv: str | Path | None = None,
    limit_runs: int | None = None,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    methods: list[str] | None = None,
    write_mode: str = DEFAULT_WRITE_MODE,
) -> pd.DataFrame:
    planned_runs, suite_output_path = _plan_runs(
        Path(suite_path),
        datasets=datasets,
        models=models,
        methods=methods,
    )
    if limit_runs is not None:
        planned_runs = planned_runs[: int(limit_runs)]

    output_path = Path(output_csv) if output_csv is not None else suite_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    new_records: list[dict[str, object]] = []
    logger = logging.getLogger(__name__)
    logger.info(
        "Planned %d benchmark runs with write_mode=%s",
        len(planned_runs),
        write_mode,
    )

    if not planned_runs:
        logger.warning("No benchmark runs matched the current suite and filters")
        print("No benchmark runs executed.")
        return pd.DataFrame()

    persisted_results = (
        pd.DataFrame()
        if write_mode == "rewrite-all"
        else _load_existing_results(output_path)
    )

    for run_id, run_cfg in planned_runs:
        run_results = pd.DataFrame(_execute_run(run_id, run_cfg))
        new_records.extend(run_results.to_dict(orient="records"))
        persisted_results = _merge_results(
            existing_results=persisted_results,
            new_results=run_results,
            run_ids=[run_id],
            write_mode=write_mode,
        )
        persisted_results.to_csv(output_path, index=False)

    results = pd.DataFrame(new_records)
    logging.getLogger(__name__).info("Wrote benchmark results to %s", output_path)
    print(results.to_string(index=False) if not results.empty else "No benchmark runs executed.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--suite",
        default=str(DEFAULT_SUITE_PATH),
        help="Path to a benchmark suite YAML file",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        default=None,
        help="Optional output CSV override",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help="Optional cap on the number of planned runs to execute",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset filters; accepts space-separated and/or comma-separated names",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model filters; accepts space-separated and/or comma-separated names",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional method filters; accepts space-separated and/or comma-separated names",
    )
    parser.add_argument(
        "--write-mode",
        choices=sorted(VALID_WRITE_MODES),
        default=DEFAULT_WRITE_MODE,
        help="How to update an existing output CSV",
    )
    args = parser.parse_args()
    run_benchmarks(
        suite_path=args.suite,
        output_csv=args.output_csv,
        limit_runs=args.limit_runs,
        datasets=_parse_name_filter(args.datasets),
        models=_parse_name_filter(args.models),
        methods=_parse_name_filter(args.methods),
        write_mode=args.write_mode,
    )


if __name__ == "__main__":
    main()
