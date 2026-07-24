from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.utils import write_reproduction_report
from method.arg_ensembling.support import (
    build_baf_program,
    nearest_neighbor_counterfactual,
    solve_argumentative_extension,
)
from model.model_object import ModelObject


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
REPORT_PATH = Path(__file__).with_name("reproduction_report.json")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    feature_path: Path
    target_path: Path
    target_column: str


@dataclass(frozen=True)
class PoolEntry:
    model: "SklearnMlpModel"
    accuracy: float
    simplicity: float
    train_predictions: np.ndarray


@dataclass(frozen=True)
class RunOverrides:
    datasets: list[str] | None
    model_sizes: list[int] | None
    max_test_points: int | None
    pool_target_size: int | None
    repeats: int | None


class SklearnMlpModel(ModelObject):
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...],
        random_state: int,
        learning_rate: str = "adaptive",
        batch_size: int = 64,
        learning_rate_init: float = 0.02,
        max_iter: int = 200,
        device: str = "cpu",
        **kwargs,
    ):
        self._model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            learning_rate=learning_rate,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
        )
        self._seed = random_state
        self._device = device.lower()
        self._need_grad = False
        self._is_trained = False
        self._class_to_index = None

    def fit(self, trainset):
        raise TypeError("SklearnMlpModel.fit() is not used in this reproduction script")

    def fit_frames(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        labels = pd.Series(np.asarray(y).reshape(-1))
        self._model.fit(X.to_numpy(dtype=np.float64), labels.to_numpy())
        classes = list(self._model.classes_)
        normalized_classes: list[int | str] = []
        for value in classes:
            if isinstance(value, (int, np.integer)):
                normalized_classes.append(int(value))
            elif isinstance(value, (float, np.floating)) and float(value).is_integer():
                normalized_classes.append(int(value))
            else:
                normalized_classes.append(str(value))
        self._class_to_index = {
            class_value: index for index, class_value in enumerate(normalized_classes)
        }
        self._is_trained = True

    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Target model is not trained")
        probabilities = torch.tensor(
            self._model.predict_proba(X.to_numpy(dtype=np.float64)),
            dtype=torch.float32,
        )
        if proba:
            return probabilities
        indices = probabilities.argmax(dim=1)
        return torch.nn.functional.one_hot(
            indices, num_classes=probabilities.shape[1]
        ).to(dtype=torch.float32)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise TypeError("SklearnMlpModel.forward() is unavailable")

    def predict_label_indices(self, X: pd.DataFrame) -> np.ndarray:
        return self.get_prediction(X, proba=True).argmax(dim=1).detach().cpu().numpy()


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("Reproduction config must parse to a dictionary")
    return config


def _normalize_hidden_layers(raw_sizes: list[list[int]]) -> list[tuple[int, ...]]:
    return [tuple(int(value) for value in sizes) for sizes in raw_sizes]


def _normalize_table_targets(
    raw_targets: dict[str, dict[str, dict[int | str, dict[str, float]]]],
) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    normalized: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for dataset_name, method_map in raw_targets.items():
        normalized[dataset_name] = {}
        for method_name, size_map in method_map.items():
            normalized[dataset_name][method_name] = {}
            for model_size, values in size_map.items():
                normalized[dataset_name][method_name][int(model_size)] = {
                    "acc": float(values["acc"]),
                    "simp": float(values["simp"]),
                }
    return normalized


def _resolve_dataset_specs(
    config: dict[str, Any],
    selected_datasets: Sequence[str] | None = None,
) -> list[DatasetSpec]:
    dataset_map = config["reproduction"]["datasets"]
    requested = list(selected_datasets) if selected_datasets is not None else list(dataset_map.keys())
    specs: list[DatasetSpec] = []
    for dataset_name in requested:
        if dataset_name not in dataset_map:
            raise KeyError(f"Unknown reproduction dataset: {dataset_name}")
        dataset_cfg = dataset_map[dataset_name]
        specs.append(
            DatasetSpec(
                name=str(dataset_name),
                feature_path=(PROJECT_ROOT / dataset_cfg["feature_path"]).resolve(),
                target_path=(PROJECT_ROOT / dataset_cfg["target_path"]).resolve(),
                target_column=str(dataset_cfg["target_column"]),
            )
        )
    return specs


def _load_dataset(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    feature_frame = pd.read_csv(spec.feature_path)
    target_frame = pd.read_csv(spec.target_path)
    if spec.target_column not in target_frame.columns:
        raise KeyError(
            f"Target column '{spec.target_column}' is missing from {spec.target_path}"
        )
    target_series = target_frame[spec.target_column].astype(int).reset_index(drop=True)
    feature_frame = feature_frame.reset_index(drop=True)
    if len(feature_frame) != len(target_series):
        raise ValueError(
            f"Feature and target row counts differ for dataset '{spec.name}'"
        )
    return feature_frame, target_series


def _select_eval_subset(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    max_test_points: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X_test) <= max_test_points:
        return X_test.reset_index(drop=True), y_test.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X_test), size=max_test_points, replace=False)
    return (
        X_test.iloc[indices].reset_index(drop=True),
        y_test.iloc[indices].reset_index(drop=True),
    )


def _train_model_pool(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hidden_layer_sizes: list[tuple[int, ...]],
    simplicity_scores: list[float],
    pool_target_size: int,
    inner_split: float,
    sklearn_cfg: dict[str, Any],
    seed: int,
) -> list[PoolEntry]:
    pool: list[PoolEntry] = []
    candidate_seed = 0
    rng = np.random.default_rng(seed)

    progress = tqdm(total=pool_target_size, desc="train-model-pool", leave=False)
    while len(pool) < pool_target_size:
        X_inner_train, _, y_inner_train, _ = train_test_split(
            X_train,
            y_train,
            test_size=inner_split,
            random_state=2 * candidate_seed + 15,
            shuffle=True,
        )

        size_index = int(rng.integers(0, len(hidden_layer_sizes)))
        model = SklearnMlpModel(
            hidden_layer_sizes=hidden_layer_sizes[size_index],
            random_state=3 * candidate_seed + 21,
            learning_rate=str(sklearn_cfg["learning_rate"]),
            batch_size=int(sklearn_cfg["batch_size"]),
            learning_rate_init=float(sklearn_cfg["learning_rate_init"]),
            max_iter=int(sklearn_cfg["max_iter"]),
        )
        model.fit_frames(X_inner_train, y_inner_train)
        outer_predictions = model.predict_label_indices(X_test)
        candidate_seed += 12

        if len(np.unique(outer_predictions)) == 1:
            continue

        accuracy = float(accuracy_score(y_test.to_numpy(), outer_predictions))
        train_predictions = model.predict_label_indices(X_train)
        pool.append(
            PoolEntry(
                model=model,
                accuracy=accuracy,
                simplicity=float(simplicity_scores[size_index]),
                train_predictions=train_predictions,
            )
        )
        progress.update(1)
    progress.close()
    return pool


def _majority_vote(predictions: np.ndarray) -> tuple[int, list[int]]:
    labels, counts = np.unique(predictions, return_counts=True)
    selected_label = int(labels[np.argpartition(counts, -1)[-1]])
    selected_indices = np.where(predictions == selected_label)[0].tolist()
    return selected_label, selected_indices


def _build_candidate_frame(
    factual: pd.Series,
    train_features: pd.DataFrame,
    pool_entries: Sequence[PoolEntry],
    factual_predictions: np.ndarray,
) -> pd.DataFrame | None:
    candidate_rows: list[pd.Series] = []
    for model_index, entry in enumerate(pool_entries):
        candidate = nearest_neighbor_counterfactual(
            factual=factual,
            train_features=train_features,
            train_predictions=entry.train_predictions,
            original_prediction=int(factual_predictions[model_index]),
            desired_prediction=None,
        )
        if candidate is None:
            return None
        candidate_rows.append(candidate.reindex(train_features.columns))
    return pd.DataFrame(candidate_rows, columns=train_features.columns)


def _predict_candidate_matrix(
    pool_entries: Sequence[PoolEntry],
    candidate_frame: pd.DataFrame,
) -> np.ndarray:
    return np.vstack(
        [entry.model.predict_label_indices(candidate_frame) for entry in pool_entries]
    )


def _evaluate_avg_row(pool_entries: Sequence[PoolEntry]) -> dict[str, float]:
    return {
        "acc": float(np.mean([entry.accuracy for entry in pool_entries])),
        "simp": float(np.mean([entry.simplicity for entry in pool_entries])),
    }


def _evaluate_majority_baseline(
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    pool_entries: Sequence[PoolEntry],
) -> dict[str, float]:
    predictions = np.zeros(len(X_eval), dtype=np.int64)
    simplicity_total = 0.0

    for row_index, (_, row) in enumerate(X_eval.iterrows()):
        factual_frame = row.to_frame().T.reindex(columns=X_eval.columns)
        factual_predictions = np.asarray(
            [
                entry.model.predict_label_indices(factual_frame)[0]
                for entry in pool_entries
            ],
            dtype=np.int64,
        )
        label, model_indices = _majority_vote(factual_predictions)
        predictions[row_index] = label
        simplicity_total += float(
            np.mean([pool_entries[index].simplicity for index in model_indices])
        )

    return {
        "acc": float(accuracy_score(y_eval.to_numpy(dtype=np.int64), predictions)),
        "simp": float(simplicity_total / len(X_eval)),
    }


def _evaluate_argumentative_method(
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    train_features: pd.DataFrame,
    pool_entries: Sequence[PoolEntry],
    semantics: str,
    preference_mode: str,
) -> dict[str, float]:
    predictions = np.zeros(len(X_eval), dtype=np.int64)
    simplicity_total = 0.0
    success_count = 0

    accuracy_scores = np.asarray(
        [entry.accuracy for entry in pool_entries], dtype=np.float64
    )
    simplicity_scores = np.asarray(
        [entry.simplicity for entry in pool_entries], dtype=np.float64
    )

    for row_index, (_, row) in enumerate(X_eval.iterrows()):
        factual_frame = row.to_frame().T.reindex(columns=X_eval.columns)
        factual_predictions = np.asarray(
            [
                entry.model.predict_label_indices(factual_frame)[0]
                for entry in pool_entries
            ],
            dtype=np.int64,
        )
        candidate_frame = _build_candidate_frame(
            factual=row,
            train_features=train_features,
            pool_entries=pool_entries,
            factual_predictions=factual_predictions,
        )
        if candidate_frame is None:
            continue

        candidate_predictions = _predict_candidate_matrix(pool_entries, candidate_frame)
        extension = solve_argumentative_extension(
            build_baf_program(
                factual_predictions=factual_predictions,
                counterfactual_predictions=candidate_predictions,
                accuracy_scores=accuracy_scores,
                simplicity_scores=simplicity_scores,
                semantics=semantics,
                preference_mode=preference_mode,
            ),
            factual_predictions=factual_predictions,
        )
        if extension is None or not extension.model_indices:
            continue

        selected_predictions = factual_predictions[extension.model_indices]
        label, _ = _majority_vote(selected_predictions)
        predictions[row_index] = label
        simplicity_total += float(np.mean(simplicity_scores[extension.model_indices]))
        success_count += 1

    return {
        "acc": float(accuracy_score(y_eval.to_numpy(dtype=np.int64), predictions)),
        "simp": (
            float(simplicity_total / success_count)
            if success_count > 0
            else float("nan")
        ),
    }


def _method_specs() -> list[tuple[str, str | None, str | None]]:
    return [
        ("avg", None, None),
        ("Sn", None, None),
        ("Sv", None, None),
        ("Sa,d", "d", "none"),
        ("Sa,d-A", "d", "accuracy"),
        ("Sa,d-S", "d", "simplicity"),
        ("Sa,d-AS", "d", "accuracy_simplicity"),
        ("Sa,s", "s", "none"),
        ("Sa,s-A", "s", "accuracy"),
        ("Sa,s-S", "s", "simplicity"),
        ("Sa,s-AS", "s", "accuracy_simplicity"),
    ]


def _initialize_results(
    method_targets: dict[str, dict[int, dict[str, float]]],
    model_sizes: Sequence[int],
) -> dict[str, dict[int, list[dict[str, float]]]]:
    results: dict[str, dict[int, list[dict[str, float]]]] = {}
    for method_name in method_targets:
        results[method_name] = {int(size): [] for size in model_sizes}
    return results


def _aggregate_results(
    results: dict[str, dict[int, list[dict[str, float]]]],
) -> dict[str, dict[int, dict[str, float]]]:
    aggregated: dict[str, dict[int, dict[str, float]]] = {}
    for method_name, size_map in results.items():
        aggregated[method_name] = {}
        for model_size, values in size_map.items():
            aggregated[method_name][model_size] = {
                "acc": float(np.mean([item["acc"] for item in values])),
                "simp": float(np.mean([item["simp"] for item in values])),
            }
    return aggregated


def _build_dataset_table(
    dataset_name: str,
    aggregated: dict[str, dict[int, dict[str, float]]],
    model_sizes: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for method_name, size_map in aggregated.items():
        row: dict[str, object] = {"Dataset": dataset_name, "Method": method_name}
        for model_size in model_sizes:
            row[f"|M|={model_size} Acc"] = size_map[model_size]["acc"]
            row[f"|M|={model_size} Simp"] = size_map[model_size]["simp"]
        rows.append(row)
    return pd.DataFrame(rows)


def _build_dataset_diff_table(
    dataset_name: str,
    aggregated: dict[str, dict[int, dict[str, float]]],
    targets: dict[str, dict[int, dict[str, float]]],
    model_sizes: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for method_name, size_map in aggregated.items():
        row: dict[str, object] = {"Dataset": dataset_name, "Method": method_name}
        for model_size in model_sizes:
            row[f"|M|={model_size} Acc"] = abs(
                size_map[model_size]["acc"] - targets[method_name][model_size]["acc"]
            )
            row[f"|M|={model_size} Simp"] = abs(
                size_map[model_size]["simp"] - targets[method_name][model_size]["simp"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve_model_sizes(
    config: dict[str, Any],
    overrides: RunOverrides,
) -> list[int]:
    return (
        [int(value) for value in overrides.model_sizes]
        if overrides.model_sizes is not None
        else [int(value) for value in config["reproduction"]["model_sizes"]]
    )


def run_reproduction(
    config: dict[str, Any],
    overrides: RunOverrides,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reproduction_cfg = config["reproduction"]
    table_targets = _normalize_table_targets(reproduction_cfg["table4_targets"])
    dataset_specs = _resolve_dataset_specs(config, selected_datasets=overrides.datasets)
    hidden_layer_sizes = _normalize_hidden_layers(
        reproduction_cfg["hidden_layer_sizes"]
    )
    model_sizes = _resolve_model_sizes(config, overrides)
    max_test_points = (
        int(overrides.max_test_points)
        if overrides.max_test_points is not None
        else int(reproduction_cfg["max_test_points"])
    )
    pool_target_size = (
        int(overrides.pool_target_size)
        if overrides.pool_target_size is not None
        else int(reproduction_cfg["pool_target_size"])
    )
    repeats = (
        int(overrides.repeats)
        if overrides.repeats is not None
        else int(reproduction_cfg["repeats"])
    )
    sample_with_replacement = bool(reproduction_cfg.get("sample_with_replacement", True))

    dataset_tables: list[pd.DataFrame] = []
    diff_tables: list[pd.DataFrame] = []

    for dataset_spec in dataset_specs:
        features, target = _load_dataset(dataset_spec)
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=float(reproduction_cfg["outer_test_size"]),
            random_state=int(reproduction_cfg["outer_split_random_state"]),
        )
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        X_eval, y_eval = _select_eval_subset(
            X_test,
            y_test,
            max_test_points=max_test_points,
            seed=int(reproduction_cfg["seed"]),
        )

        pool = _train_model_pool(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hidden_layer_sizes=hidden_layer_sizes,
            simplicity_scores=[
                float(value) for value in reproduction_cfg["simplicity_scores"]
            ],
            pool_target_size=pool_target_size,
            inner_split=float(reproduction_cfg["inner_split"]),
            sklearn_cfg=dict(reproduction_cfg["sklearn"]),
            seed=int(reproduction_cfg["seed"]),
        )

        method_targets = table_targets[dataset_spec.name]
        results = _initialize_results(method_targets, model_sizes)
        for model_size in model_sizes:
            repeat_progress = tqdm(
                range(repeats),
                desc=f"{dataset_spec.name}-|M|={model_size}",
                leave=False,
            )
            for repeat_index in repeat_progress:
                repeat_seed = int(reproduction_cfg["seed"]) * (repeat_index + 3) + 4321
                rng = np.random.default_rng(repeat_seed)
                selected_indices = (
                    rng.integers(0, len(pool), size=model_size)
                    if sample_with_replacement
                    else rng.choice(len(pool), size=model_size, replace=False)
                )
                selected_entries = [pool[int(index)] for index in selected_indices]

                results["avg"][model_size].append(_evaluate_avg_row(selected_entries))
                majority_metrics = _evaluate_majority_baseline(
                    X_eval=X_eval,
                    y_eval=y_eval,
                    pool_entries=selected_entries,
                )
                results["Sn"][model_size].append(dict(majority_metrics))
                results["Sv"][model_size].append(dict(majority_metrics))

                for method_name, semantics, preference_mode in _method_specs():
                    if method_name in {"avg", "Sn", "Sv"}:
                        continue
                    results[method_name][model_size].append(
                        _evaluate_argumentative_method(
                            X_eval=X_eval,
                            y_eval=y_eval,
                            train_features=X_train,
                            pool_entries=selected_entries,
                            semantics=str(semantics),
                            preference_mode=str(preference_mode),
                        )
                    )

        aggregated = _aggregate_results(results)
        dataset_tables.append(
            _build_dataset_table(dataset_spec.name, aggregated, model_sizes)
        )
        diff_tables.append(
            _build_dataset_diff_table(
                dataset_spec.name,
                aggregated,
                method_targets,
                model_sizes,
            )
        )

    return (
        pd.concat(dataset_tables, ignore_index=True),
        pd.concat(diff_tables, ignore_index=True),
    )


def _build_report_experiments(
    reproduced_table: pd.DataFrame,
    targets: dict[str, dict[str, dict[int, dict[str, float]]]],
    model_sizes: Sequence[int],
) -> dict[str, dict[str, Any]]:
    experiments: dict[str, dict[str, Any]] = {}
    for row in reproduced_table.to_dict(orient="records"):
        dataset_name = str(row["Dataset"])
        method_name = str(row["Method"])
        for model_size in model_sizes:
            experiments[f"{dataset_name}_{method_name}_M_{model_size}"] = {
                "configuration": {
                    "dataset": dataset_name,
                    "method": method_name,
                    "model_pool_size": int(model_size),
                },
                "metrics": {
                    "accuracy": {
                        "original": targets[dataset_name][method_name][int(model_size)]["acc"],
                        "reproduced": row[f"|M|={model_size} Acc"],
                    },
                    "simplicity": {
                        "original": targets[dataset_name][method_name][int(model_size)]["simp"],
                        "reproduced": row[f"|M|={model_size} Simp"],
                    },
                },
            }
    return experiments


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH.as_posix(),
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--model-sizes", nargs="+", type=int)
    parser.add_argument("--max-test-points", type=int)
    parser.add_argument("--pool-target-size", type=int)
    parser.add_argument("--repeats", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = _load_config(config_path)
    overrides = RunOverrides(
        datasets=args.datasets,
        model_sizes=args.model_sizes,
        max_test_points=args.max_test_points,
        pool_target_size=args.pool_target_size,
        repeats=args.repeats,
    )

    reproduced_table, diff_table = run_reproduction(config, overrides)
    targets = _normalize_table_targets(config["reproduction"]["table4_targets"])
    model_sizes = _resolve_model_sizes(config, overrides)
    report_path = write_reproduction_report(
        output_path=REPORT_PATH,
        paper_id="arg_ensembling_table4",
        reproduction_metadata={
            "timestamp": datetime.now(timezone.utc),
            "framework_version": "1.0.0",
            "source_script": Path(__file__).name,
            "config_path": str(config_path),
            "experiment_name": config["name"],
            "datasets": (
                list(args.datasets)
                if args.datasets is not None
                else list(config["reproduction"]["datasets"].keys())
            ),
        },
        experiments_data=_build_report_experiments(
            reproduced_table=reproduced_table,
            targets=targets,
            model_sizes=model_sizes,
        ),
    )

    print(f"Experiment: {config['name']}")
    print("Paper target: Table 4")
    print()
    print("Reproduced Table 4:")
    print(reproduced_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()
    print("Absolute Differences vs Paper Table 4:")
    print(diff_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"reproduction_report_path: {report_path}")


@pytest.mark.slow
def test_reproduce() -> None:
    pytest.skip(
        "Run this module as a script for bounded reproduction probes."
    )


if __name__ == "__main__":
    main()
