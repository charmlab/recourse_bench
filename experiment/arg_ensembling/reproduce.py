from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from evaluation.distance import DistanceEvaluation
from experiment import Experiment
from method.arg_ensembling.support import (
    build_baf_program,
    build_model_adapters,
    nearest_neighbor_counterfactual,
    predict_label_indices,
    solve_largest_extension_partitioned,
)
from model.model_object import ModelObject
from utils.seed import seed_context
from utils.registry import get_registry


@dataclass
class StrategyRow:
    dataset: str
    ensemble_size: int
    repeat: int
    strategy: str
    avg_model_accuracy: float
    aggregated_accuracy: float
    non_emptiness_rate: float
    model_agreement_rate: float
    counterfactual_validity_rate: float
    counterfactual_coherence_rate: float
    distance_l0: float
    distance_l1: float
    distance_l2: float
    distance_linf: float
    avg_runtime_sec: float
    factual_count: int


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _resolve_device(preferred_device: str | None = None) -> str:
    if preferred_device is not None:
        preferred_device = str(preferred_device).lower()
        if preferred_device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return preferred_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_base_experiment_config(dataset_name: str, settings: dict, device: str) -> dict:
    seed = int(settings.get("seed", 7))
    sample = settings.get("sample")
    preprocess_cfg: list[dict[str, object]] = [
        {
            "name": "scale",
            "seed": seed,
            "scaling": str(settings.get("scaling", "standardize")),
            "range": True,
        },
        {
            "name": "encode",
            "seed": seed,
            "encoding": str(settings.get("encoding", "onehot")),
        },
        {
            "name": "split",
            "seed": seed,
            "split": float(settings.get("test_split", 0.2)),
        },
    ]
    if sample is not None:
        preprocess_cfg[-1]["sample"] = int(sample)

    return {
        "name": f"{dataset_name}_arg_ensembling_reproduce",
        "logger": {"level": "ERROR"},
        "caching": {"path": "./cache/"},
        "dataset": {"name": dataset_name},
        "preprocess": preprocess_cfg,
        "model": {
            "name": str(settings.get("model_name", "mlp")).lower(),
            "seed": seed,
            "device": device,
            "epochs": int(settings["epochs"]),
            "learning_rate": float(settings["learning_rate"]),
            "batch_size": int(settings["batch_size"]),
            "optimizer": str(settings.get("optimizer", "adam")).lower(),
            "criterion": str(settings.get("criterion", "cross_entropy")).lower(),
            "output_activation": str(
                settings.get("output_activation", "softmax")
            ).lower(),
            "layers": [int(value) for value in settings["layers"]],
        },
        "method": {
            "name": "arg_ensembling",
            "seed": seed,
            "device": device,
            "desired_class": settings.get("desired_class"),
        },
        "evaluation": [{"name": "validity"}, {"name": "distance"}],
    }


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


def _build_model_pool(
    trainset,
    testset,
    model_cfg: dict,
    pool_size: int,
    model_seeds: list[int],
) -> tuple[list[ModelObject], list[float]]:
    if len(model_seeds) < pool_size:
        raise ValueError("model_seeds must contain at least pool_size entries")

    model_pool: list[ModelObject] = []
    accuracies: list[float] = []

    model_registry = get_registry("TargetModel")
    model_name = str(model_cfg["name"])
    model_class = model_registry[model_name]

    for index in range(pool_size):
        current_cfg = deepcopy(model_cfg)
        current_cfg["seed"] = int(model_seeds[index])
        current_cfg["save_name"] = None
        model = model_class(**current_cfg)
        model.fit(trainset)
        probabilities = model.predict_proba(testset).detach().cpu()
        prediction = probabilities.argmax(dim=1).numpy()
        target = testset.get(target=True).iloc[:, 0]
        class_to_index = model.get_class_to_index()
        encoded_target = np.asarray(
            [
                class_to_index[int(value)]
                if isinstance(value, (float, np.floating)) and float(value).is_integer()
                else class_to_index[value]
                for value in target.tolist()
            ],
            dtype=np.int64,
        )
        accuracy = float((prediction == encoded_target).mean())
        model_pool.append(model)
        accuracies.append(accuracy)

    return model_pool, accuracies


def _build_counterfactual_dataset(
    factuals,
    feature_frame: pd.DataFrame,
    evaluation_mask: pd.Series | None = None,
):
    target_column = factuals.target_column
    counterfactual_target = pd.DataFrame(
        -1.0,
        index=feature_frame.index,
        columns=[target_column],
    )
    counterfactual_df = pd.concat([feature_frame, counterfactual_target], axis=1)
    counterfactual_df = counterfactual_df.reindex(columns=factuals.ordered_features())

    output = factuals.clone()
    output.update("counterfactual", True, df=counterfactual_df)
    if evaluation_mask is not None:
        output.update(
            "evaluation_filter",
            pd.DataFrame(
                evaluation_mask.astype(bool), index=feature_frame.index, columns=["evaluation_filter"]
            ),
        )
    output.freeze()
    return output


def _select_recourse_factuals(
    primary_model: ModelObject,
    testset,
    desired_class: int | None,
):
    if desired_class is None:
        return testset

    class_to_index = primary_model.get_class_to_index()
    desired_index = int(class_to_index[desired_class])
    prediction = primary_model.predict(testset).argmax(dim=1).detach().cpu().numpy()
    feature_frame = testset.get(target=False)
    target_frame = testset.get(target=True)
    keep_mask = pd.Series(prediction != desired_index, index=feature_frame.index, dtype=bool)

    filtered = testset.clone()
    filtered_df = pd.concat([feature_frame, target_frame], axis=1).loc[keep_mask].copy(
        deep=True
    )
    filtered.update("testset", True, df=filtered_df)
    filtered.freeze()
    return filtered


def _resolve_target_index(model: ModelObject, desired_class: int) -> int:
    return int(model.get_class_to_index()[desired_class])


def _compute_candidate_frame(
    factual: pd.Series,
    train_features: pd.DataFrame,
    train_prediction_indices: list[np.ndarray],
    factual_predictions: np.ndarray,
    feature_names: list[str],
    desired_index: int | None,
) -> pd.DataFrame:
    candidate_rows: list[pd.Series] = []
    for model_index, train_predictions in enumerate(train_prediction_indices):
        candidate = nearest_neighbor_counterfactual(
            factual=factual,
            train_features=train_features,
            train_predictions=train_predictions,
            original_prediction=int(factual_predictions[model_index]),
            desired_prediction=desired_index,
        )
        if candidate is None:
            candidate_rows.append(pd.Series(np.nan, index=feature_names))
        else:
            candidate_rows.append(candidate.reindex(feature_names))
    return pd.DataFrame(candidate_rows, columns=feature_names)


def _naive_majority_indices(factual_predictions: np.ndarray) -> list[int]:
    if factual_predictions.size == 0:
        return []
    values, counts = np.unique(factual_predictions, return_counts=True)
    majority_value = int(values[np.argmax(counts)])
    return [
        index
        for index, prediction in enumerate(factual_predictions.tolist())
        if int(prediction) == majority_value
    ]


def _all_selected_valid(
    selected_model_indices: list[int],
    selected_ce_indices: list[int],
    factual_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> bool:
    for model_index in selected_model_indices:
        for ce_index in selected_ce_indices:
            if int(candidate_predictions[model_index, ce_index]) == int(
                factual_predictions[model_index]
            ):
                return False
    return True


def _coherent_pairing(
    selected_model_indices: list[int], selected_ce_indices: list[int], num_models: int
) -> bool:
    selected_models = set(int(index) for index in selected_model_indices)
    selected_ces = set(int(index) for index in selected_ce_indices)
    return all((index in selected_models) == (index in selected_ces) for index in range(num_models))


def _agreement(selected_model_indices: list[int], factual_predictions: np.ndarray) -> bool:
    if not selected_model_indices:
        return False
    labels = {int(factual_predictions[index]) for index in selected_model_indices}
    return len(labels) == 1


def _build_strategy_counterfactuals(
    strategy: str,
    candidate_frame: pd.DataFrame,
    selected_ce_indices: list[int],
    factual_feature_frame: pd.DataFrame,
) -> pd.Series:
    feature_names = list(factual_feature_frame.columns)
    if not selected_ce_indices:
        return pd.Series(np.nan, index=feature_names)

    valid_indices = [
        int(index)
        for index in selected_ce_indices
        if 0 <= int(index) < candidate_frame.shape[0]
        and not candidate_frame.iloc[int(index)].isna().any()
    ]
    if not valid_indices:
        return pd.Series(np.nan, index=feature_names)

    factual_values = factual_feature_frame.iloc[0].to_numpy(dtype=np.float64, copy=False)
    best_index = min(
        valid_indices,
        key=lambda index: (
            float(
                np.linalg.norm(
                    candidate_frame.iloc[index].to_numpy(dtype=np.float64, copy=False)
                    - factual_values
                )
            ),
            index,
        ),
    )
    return candidate_frame.iloc[best_index].reindex(feature_names).copy(deep=True)


def _evaluate_strategy(
    strategy: str,
    factuals,
    chosen_counterfactual_rows: list[pd.Series],
    desired_class: int | None,
) -> tuple[float, float, float, float, float]:
    feature_names = list(factuals.get(target=False).columns)
    candidate_frame = pd.DataFrame(
        chosen_counterfactual_rows,
        index=factuals.get(target=False).index,
        columns=feature_names,
    )
    evaluation_mask = pd.Series(True, index=candidate_frame.index, dtype=bool)
    counterfactuals = _build_counterfactual_dataset(
        factuals=factuals,
        feature_frame=candidate_frame,
        evaluation_mask=evaluation_mask,
    )
    distance_metrics = DistanceEvaluation().evaluate(factuals, counterfactuals)
    return (
        float(distance_metrics["distance_l0"].iloc[0]),
        float(distance_metrics["distance_l1"].iloc[0]),
        float(distance_metrics["distance_l2"].iloc[0]),
        float(distance_metrics["distance_linf"].iloc[0]),
        float((~candidate_frame.isna().any(axis=1)).mean()),
    )


def _run_repeat(
    dataset_name: str,
    ensemble_size: int,
    repeat: int,
    model_pool: list[ModelObject],
    model_accuracies: list[float],
    trainset,
    factuals,
    desired_class: int | None,
    seed: int,
) -> list[StrategyRow]:
    with seed_context(seed + repeat * 1000 + ensemble_size):
        selected_indices = np.random.choice(
            len(model_pool), size=ensemble_size, replace=False
        ).tolist()

    selected_models = [model_pool[index] for index in selected_indices]
    selected_accuracies = [model_accuracies[index] for index in selected_indices]
    method_cfg = {
        "target_model": selected_models[0],
        "seed": seed + repeat,
        "device": selected_models[0]._device,
        "desired_class": desired_class,
        "ensemble_models": selected_models[1:],
    }
    from method.arg_ensembling.arg_ensembling import ArgEnsemblingMethod

    method = ArgEnsemblingMethod(**method_cfg)
    method.fit(trainset)

    feature_names = list(method._feature_names)
    factual_features = factuals.get(target=False).loc[:, feature_names].copy(deep=True)
    target_values = factuals.get(target=True).iloc[:, 0].astype(int)
    adapters = method._model_adapters
    desired_index = None if method._desired_index is None else int(method._desired_index)

    strategy_names = ["sn", "sv", "sas"]
    property_flags = {
        name: {
            "non_emptiness": [],
            "model_agreement": [],
            "counterfactual_validity": [],
            "counterfactual_coherence": [],
            "aggregated_accuracy": [],
            "runtime_sec": [],
        }
        for name in strategy_names
    }
    chosen_rows = {name: [] for name in strategy_names}

    factual_iterator = tqdm(
        factual_features.iterrows(),
        total=factual_features.shape[0],
        desc=f"{dataset_name}|M|={ensemble_size}|rep={repeat}",
        leave=False,
    )
    for row_index, factual in factual_iterator:
        factual_df = factual.to_frame().T.reindex(columns=feature_names)
        factual_predictions = np.array(
            [int(predict_label_indices(adapter, factual_df)[0]) for adapter in adapters],
            dtype=np.int64,
        )

        candidate_frame = _compute_candidate_frame(
            factual=factual,
            train_features=method._train_features,
            train_prediction_indices=method._train_prediction_indices,
            factual_predictions=factual_predictions,
            feature_names=feature_names,
            desired_index=desired_index,
        )
        candidate_predictions = np.vstack(
            [predict_label_indices(adapter, candidate_frame) for adapter in adapters]
        )

        start_time = time.perf_counter()
        sas_models, sas_ces = solve_largest_extension_partitioned(
            build_baf_program(
                factual_predictions=factual_predictions,
                counterfactual_predictions=candidate_predictions,
            )
        )
        sas_runtime = time.perf_counter() - start_time

        majority_indices = _naive_majority_indices(factual_predictions)
        sv_ce_indices = [
            index
            for index in majority_indices
            if not candidate_frame.iloc[index].isna().any()
            and all(
                int(candidate_predictions[model_index, index]) != int(factual_predictions[model_index])
                for model_index in majority_indices
            )
        ]

        selected = {
            "sn": (majority_indices, list(majority_indices), 0.0),
            "sv": (majority_indices, sv_ce_indices, 0.0),
            "sas": (list(sas_models), list(sas_ces), sas_runtime),
        }

        for strategy_name, (selected_model_indices, selected_ce_indices, runtime_sec) in selected.items():
            non_empty = bool(selected_model_indices) and bool(selected_ce_indices)
            agreement = _agreement(selected_model_indices, factual_predictions)
            validity = _all_selected_valid(
                selected_model_indices,
                selected_ce_indices,
                factual_predictions,
                candidate_predictions,
            )
            coherence = _coherent_pairing(
                selected_model_indices, selected_ce_indices, len(selected_models)
            )

            if agreement and selected_model_indices:
                aggregated_label = int(factual_predictions[selected_model_indices[0]])
                aggregated_accuracy = float(aggregated_label == int(target_values.loc[row_index]))
            else:
                aggregated_accuracy = 0.0

            property_flags[strategy_name]["non_emptiness"].append(float(non_empty))
            property_flags[strategy_name]["model_agreement"].append(float(agreement))
            property_flags[strategy_name]["counterfactual_validity"].append(float(validity))
            property_flags[strategy_name]["counterfactual_coherence"].append(float(coherence))
            property_flags[strategy_name]["aggregated_accuracy"].append(float(aggregated_accuracy))
            property_flags[strategy_name]["runtime_sec"].append(float(runtime_sec))
            chosen_rows[strategy_name].append(
                _build_strategy_counterfactuals(
                    strategy_name,
                    candidate_frame,
                    selected_ce_indices,
                    factual_df,
                )
            )

    rows: list[StrategyRow] = []
    avg_model_accuracy = float(np.mean(selected_accuracies))
    for strategy_name in strategy_names:
        distance_l0, distance_l1, distance_l2, distance_linf, _ = _evaluate_strategy(
            strategy_name,
            factuals,
            chosen_rows[strategy_name],
            desired_class,
        )
        rows.append(
            StrategyRow(
                dataset=dataset_name,
                ensemble_size=ensemble_size,
                repeat=repeat,
                strategy=strategy_name,
                avg_model_accuracy=avg_model_accuracy,
                aggregated_accuracy=float(
                    np.mean(property_flags[strategy_name]["aggregated_accuracy"])
                ),
                non_emptiness_rate=float(
                    np.mean(property_flags[strategy_name]["non_emptiness"])
                ),
                model_agreement_rate=float(
                    np.mean(property_flags[strategy_name]["model_agreement"])
                ),
                counterfactual_validity_rate=float(
                    np.mean(property_flags[strategy_name]["counterfactual_validity"])
                ),
                counterfactual_coherence_rate=float(
                    np.mean(property_flags[strategy_name]["counterfactual_coherence"])
                ),
                distance_l0=distance_l0,
                distance_l1=distance_l1,
                distance_l2=distance_l2,
                distance_linf=distance_linf,
                avg_runtime_sec=float(np.mean(property_flags[strategy_name]["runtime_sec"])),
                factual_count=int(factual_features.shape[0]),
            )
        )
    return rows


def _summarize_results(results: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    summary: dict[str, list[dict[str, object]]] = {}
    for dataset_name in sorted(results["dataset"].unique()):
        dataset_df = results.loc[results["dataset"] == dataset_name].copy(deep=True)

        accuracy_table = (
            dataset_df.pivot_table(
                index="ensemble_size",
                columns="strategy",
                values="aggregated_accuracy",
                aggfunc="mean",
            )
            .reset_index()
            .rename_axis(columns=None)
        )
        model_accuracy = (
            dataset_df.groupby("ensemble_size", as_index=False)["avg_model_accuracy"]
            .mean()
            .rename(columns={"avg_model_accuracy": "avg_model_accuracy"})
        )
        accuracy_table = accuracy_table.merge(model_accuracy, on="ensemble_size", how="left")
        accuracy_table = accuracy_table.rename(
            columns={
                "sn": "Sn_accuracy",
                "sv": "Sv_accuracy",
                "sas": "Sa,s_accuracy",
                "ensemble_size": "|M|",
            }
        )
        accuracy_table = accuracy_table[
            ["|M|", "avg_model_accuracy", "Sn_accuracy", "Sv_accuracy", "Sa,s_accuracy"]
        ]

        property_rows = []
        for ensemble_size, size_df in dataset_df.groupby("ensemble_size"):
            for strategy_name, label in [("sn", "Sn"), ("sv", "Sv"), ("sas", "Sa,s")]:
                strategy_df = size_df.loc[size_df["strategy"] == strategy_name]
                property_rows.append(
                    {
                        "|M|": int(ensemble_size),
                        "strategy": label,
                        "non_emptiness_rate": float(
                            strategy_df["non_emptiness_rate"].mean()
                        ),
                        "model_agreement_rate": float(
                            strategy_df["model_agreement_rate"].mean()
                        ),
                        "counterfactual_validity_rate": float(
                            strategy_df["counterfactual_validity_rate"].mean()
                        ),
                        "counterfactual_coherence_rate": float(
                            strategy_df["counterfactual_coherence_rate"].mean()
                        ),
                    }
                )

        distance_rows = []
        for ensemble_size, size_df in dataset_df.loc[
            dataset_df["strategy"] == "sas"
        ].groupby("ensemble_size"):
            distance_rows.append(
                {
                    "|M|": int(ensemble_size),
                    "Sa,s_distance_l0": float(size_df["distance_l0"].mean()),
                    "Sa,s_distance_l1": float(size_df["distance_l1"].mean()),
                    "Sa,s_distance_l2": float(size_df["distance_l2"].mean()),
                    "Sa,s_distance_linf": float(size_df["distance_linf"].mean()),
                    "Sa,s_avg_runtime_sec": float(size_df["avg_runtime_sec"].mean()),
                }
            )

        summary[dataset_name] = [
            {
                "accuracy_table": accuracy_table.to_dict(orient="records"),
                "property_table": property_rows,
                "distance_table": distance_rows,
            }
        ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("reproduce_config.yml")),
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    device = _resolve_device(config.get("device"))
    global_seed = int(config.get("seed", 7))
    desired_class = config.get("desired_class")

    all_rows: list[StrategyRow] = []
    for dataset_name, dataset_settings in config["datasets"].items():
        effective_settings = deepcopy(dataset_settings)
        effective_settings.setdefault("seed", global_seed)
        effective_settings.setdefault("desired_class", desired_class)
        base_cfg = _build_base_experiment_config(dataset_name, effective_settings, device)
        experiment = Experiment(base_cfg)
        trainset, testset = _materialize_datasets(experiment)

        model_pool, model_accuracies = _build_model_pool(
            trainset=trainset,
            testset=testset,
            model_cfg=base_cfg["model"],
            pool_size=int(effective_settings["pool_size"]),
            model_seeds=[int(value) for value in effective_settings["model_seeds"]],
        )

        factuals = _select_recourse_factuals(
            primary_model=model_pool[0],
            testset=testset,
            desired_class=desired_class,
        )
        if "max_factuals" in effective_settings and effective_settings["max_factuals"] is not None:
            max_factuals = int(effective_settings["max_factuals"])
            truncated = factuals.clone()
            truncated_df = pd.concat(
                [factuals.get(target=False), factuals.get(target=True)], axis=1
            ).iloc[:max_factuals].copy(deep=True)
            truncated.update("testset", True, df=truncated_df)
            truncated.freeze()
            factuals = truncated

        repeat_tasks = [
            (int(ensemble_size), int(repeat))
            for ensemble_size in effective_settings["ensemble_sizes"]
            for repeat in range(int(effective_settings["repeats"]))
        ]
        for ensemble_size, repeat in tqdm(
            repeat_tasks,
            desc=f"{dataset_name}-repeats",
            leave=True,
        ):
            repeat_rows = _run_repeat(
                dataset_name=dataset_name,
                ensemble_size=ensemble_size,
                repeat=repeat,
                model_pool=model_pool,
                model_accuracies=model_accuracies,
                trainset=trainset,
                factuals=factuals,
                desired_class=desired_class,
                seed=global_seed,
            )
            all_rows.extend(repeat_rows)

    results = pd.DataFrame([row.__dict__ for row in all_rows])
    summary = _summarize_results(results)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
