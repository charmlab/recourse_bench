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

from evaluation.evaluation_utils import resolve_evaluation_inputs
from experiment import Experiment

DEFAULT_CONFIG_PATH = Path(__file__).with_name("reproduce_probe.yml")
REFERENCE_FEATURE_ORDER = [
    "age",
    "two_year_recid",
    "priors_count",
    "length_of_stay",
    "c_charge_degree_cat_F",
    "c_charge_degree_cat_M",
    "race_cat_African-American",
    "race_cat_Other",
    "sex_cat_Female",
    "sex_cat_Male",
]


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_layers(model_overrides: dict) -> list[int] | None:
    if "layers" in model_overrides:
        return [int(layer) for layer in model_overrides["layers"]]

    hidden_layers = model_overrides.get("hidden_layers")
    if hidden_layers is None:
        return None
    if not isinstance(hidden_layers, list):
        raise TypeError("hidden_layers must be a list")
    if hidden_layers and isinstance(hidden_layers[0], list):
        if len(hidden_layers) != 1:
            raise ValueError(
                "Reference MLP hidden_layers must contain exactly one layer block"
            )
        return [int(layer) for layer in hidden_layers[0]]
    return [int(layer) for layer in hidden_layers]


def _translate_reference_config(reference_config: dict, device: str) -> dict:
    experiment_cfg = deepcopy(reference_config.get("experiment", {}))
    seed = int(experiment_cfg.get("seed", 42))
    experiment_name = str(
        experiment_cfg.get("name", "compas_probe_mlp_experiment")
    )

    data_cfg = reference_config.get("data")
    if not isinstance(data_cfg, list) or len(data_cfg) != 1:
        raise ValueError("Reference reproduction expects exactly one data section")
    dataset_cfg = deepcopy(data_cfg[0])
    dataset_name = str(dataset_cfg.get("name", "")).lower()
    if dataset_name != "compas_carla":
        raise ValueError("This reproduction script supports compas_carla only")

    data_overrides = deepcopy(dataset_cfg.get("overrides", {}))
    if bool(data_overrides.get("balance_classes", False)):
        raise ValueError("balance_classes=true is not supported in this reproduction")

    preprocessing_strategy = str(
        data_overrides.get("preprocessing_strategy", "normalize")
    ).lower()
    train_split = float(data_overrides.get("train_split", 0.75))
    if not 0.0 < train_split < 1.0:
        raise ValueError("data.overrides.train_split must satisfy 0 < train_split < 1")

    model_cfg = deepcopy(reference_config.get("model", {}))
    model_overrides = deepcopy(model_cfg.get("overrides", {}))
    layers = _resolve_layers(model_overrides)
    native_model_cfg = {
        "name": str(model_cfg.get("name", "mlp")).lower(),
        "seed": seed,
        "device": device,
        "epochs": int(model_overrides.get("epochs", 25)),
        "learning_rate": float(model_overrides.get("learning_rate", 0.002)),
        "batch_size": int(model_overrides.get("batch_size", 25)),
        "optimizer": str(model_overrides.get("optimizer", "rms")).lower(),
        "criterion": str(model_overrides.get("criterion", "cross_entropy")).lower(),
        "output_activation": str(
            model_overrides.get("output_activation", "softmax")
        ).lower(),
        "save_name": f"{experiment_name}_mlp",
    }
    if layers is not None:
        native_model_cfg["layers"] = layers

    method_cfg = deepcopy(reference_config.get("method", {}))
    method_overrides = deepcopy(method_cfg.get("overrides", {}))
    native_method_cfg = {
        "name": str(method_cfg.get("name", "probe")).lower(),
        "seed": seed,
        "device": device,
    }
    passthrough_keys = [
        "feature_cost",
        "feature_costs",
        "lr",
        "lambda_",
        "y_target",
        "n_iter",
        "t_max_min",
        "max_minutes",
        "norm",
        "clamp",
        "loss_type",
        "binary_cat_features",
        "noise_variance",
        "invalidation_target",
        "inval_target_eps",
        "desired_class",
    ]
    for key in passthrough_keys:
        if key in method_overrides:
            native_method_cfg[key] = deepcopy(method_overrides[key])

    evaluation_cfg = deepcopy(reference_config.get("evaluation", {}))
    metrics_cfg = evaluation_cfg.get("metrics", [])
    native_evaluation_cfg = []
    for metric_cfg in metrics_cfg:
        metric_name = str(metric_cfg.get("name", "")).lower()
        native_metric_cfg = {"name": metric_name}
        if "hyperparameters" in metric_cfg:
            native_metric_cfg.update(deepcopy(metric_cfg["hyperparameters"]))
        native_evaluation_cfg.append(native_metric_cfg)

    return {
        "name": experiment_name,
        "logger": {
            "level": str(experiment_cfg.get("logger", "INFO")).upper(),
            "path": f"./logs/{experiment_name}.log",
        },
        "caching": {"path": "./cache/"},
        "dataset": {"name": dataset_name},
        "preprocess": [
            {
                "name": "scale",
                "seed": seed,
                "scaling": preprocessing_strategy,
                "range": True,
            },
            {
                "name": "encode",
                "seed": seed,
                "encoding": "onehot",
            },
            {
                "name": "reorder",
                "seed": seed,
                "order": REFERENCE_FEATURE_ORDER,
            },
            {
                "name": "split",
                "seed": seed,
                "split": 1.0 - train_split,
            },
        ],
        "model": native_model_cfg,
        "method": native_method_cfg,
        "evaluation": native_evaluation_cfg,
    }


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
    prediction = probabilities.argmax(dim=1)

    y = testset.get(target=True).iloc[:, 0]
    class_to_index = model.get_class_to_index()
    encoded_target = torch.tensor(
        [
            class_to_index[int(value)]
            if isinstance(value, float) and float(value).is_integer()
            else class_to_index[value]
            for value in y.tolist()
        ],
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


def _build_frozen_dataset(template, df: pd.DataFrame, marker: str):
    dataset = template.clone()
    dataset.update(marker, True, df=df.copy(deep=True))
    dataset.freeze()
    return dataset


def _select_factual_indices(
    model,
    feature_pool: pd.DataFrame,
    experiment_cfg: dict,
) -> pd.Index:
    selection_mode = str(
        experiment_cfg.get("factual_selection", "negative_class")
    ).lower()
    sample_seed = int(experiment_cfg.get("seed", 42))
    num_factuals = int(experiment_cfg.get("num_factuals", 20))

    predictions = model.get_prediction(feature_pool, proba=False).argmax(dim=1).cpu()
    class_to_index = model.get_class_to_index()
    negative_index = class_to_index.get(0, min(class_to_index.values()))
    negative_mask = predictions.numpy() == negative_index
    negative_index_pool = feature_pool.index[negative_mask]

    if selection_mode == "negative_class":
        if negative_index_pool.shape[0] < num_factuals:
            raise ValueError(
                "Not enough predicted negative-class instances for factual selection"
            )
        selected = pd.Series(negative_index_pool).sample(
            n=num_factuals,
            random_state=sample_seed,
        )
        return pd.Index(selected.tolist())
    if selection_mode == "all":
        return pd.Index(negative_index_pool)

    raise ValueError(f"Unsupported factual_selection mode: {selection_mode}")


def _save_results(
    output_dir: Path,
    experiment_name: str,
    summary: dict,
    metrics: pd.DataFrame,
    output_format: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([summary])

    resolved_format = output_format.lower()
    if resolved_format not in {"csv", "json", "both"}:
        raise ValueError("output_format must be one of {'csv', 'json', 'both'}")

    if resolved_format in {"csv", "both"}:
        summary_df.to_csv(
            output_dir / f"{experiment_name}_summary.csv",
            index=False,
        )
        metrics.to_csv(
            output_dir / f"{experiment_name}_metrics.csv",
            index=False,
        )

    if resolved_format in {"json", "both"}:
        with (output_dir / f"{experiment_name}_summary.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(summary, file, indent=2)
        with (output_dir / f"{experiment_name}_metrics.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metrics.to_dict(orient="records"), file, indent=2)


def run_reproduction(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    resolved_config_path = Path(config_path).resolve()
    reference_config = _load_config(resolved_config_path)
    experiment_cfg = deepcopy(reference_config.get("experiment", {}))

    native_config = _translate_reference_config(reference_config, _resolve_device())
    experiment = Experiment(native_config)
    logger = experiment._logger

    trainset, testset = _materialize_datasets(experiment)

    logger.info("Training target model")
    experiment._target_model.fit(trainset)
    model_metrics = _compute_model_metrics(experiment._target_model, testset)
    logger.info("Test accuracy M1: %.4f", model_metrics["test_accuracy"])
    logger.info("Test AUC M1: %.4f", model_metrics["test_auc"])

    logger.info("Training PROBE")
    experiment._method.fit(trainset)

    train_df = pd.concat([trainset.get(target=False), trainset.get(target=True)], axis=1)
    test_df = pd.concat([testset.get(target=False), testset.get(target=True)], axis=1)
    combined_df = pd.concat([train_df, test_df], axis=0)
    combined = _build_frozen_dataset(trainset, combined_df, "combined_source")

    selected_indices = _select_factual_indices(
        experiment._target_model,
        combined.get(target=False),
        experiment_cfg,
    )
    factual_df = combined_df.loc[selected_indices].copy(deep=True)
    factuals = _build_frozen_dataset(combined, factual_df, "selected_for_probe")
    logger.info("Selected %d factual instances", len(factuals))

    logger.info("Generating counterfactuals")
    counterfactuals = experiment._method.predict(factuals)

    evaluation_results = [
        evaluation_step.evaluate(factuals, counterfactuals)
        for evaluation_step in experiment._evaluation
    ]
    metrics = pd.concat(evaluation_results, axis=1)
    experiment._metrics = metrics

    _, _, evaluation_mask, success_mask = resolve_evaluation_inputs(
        factuals,
        counterfactuals,
    )
    successful_count = int((evaluation_mask & success_mask).sum())
    summary = {
        "device": native_config["model"]["device"],
        "num_factuals_selected": int(len(factuals)),
        "num_counterfactuals_evaluated": int(evaluation_mask.sum()),
        "num_counterfactuals_successful": successful_count,
        **model_metrics,
    }

    if bool(experiment_cfg.get("save_results", True)):
        _save_results(
            output_dir=Path(str(experiment_cfg.get("output_dir", "./results/probe"))),
            experiment_name=native_config["name"],
            summary=summary,
            metrics=metrics,
            output_format=str(experiment_cfg.get("output_format", "both")),
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))

    return {
        "summary": summary,
        "metrics": metrics,
        "factuals": factuals,
        "counterfactuals": counterfactuals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the reference-style PROBE reproduction YAML",
    )
    args = parser.parse_args()
    run_reproduction(args.path)


if __name__ == "__main__":
    main()
