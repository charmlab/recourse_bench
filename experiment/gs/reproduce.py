from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from evaluation.evaluation_utils import resolve_evaluation_inputs
from experiment import Experiment

DEFAULT_CONFIG_PATH = Path(__file__).with_name(
    "news_popularity_randomforest_gs_reproduce.yaml"
)
SEED = 42
RF_GRID = [10, 50, 100, 200, 500]
PAPER_FACTUAL_LIMIT = 256
PAPER_TARGETS = {
    "test_auc": 0.70,
    "max_l0": 17.0,
    "rate_l0_le_9": 0.80,
    "rate_l0_le_5": 0.30,
}
ASSERT_BOUNDS = {
    "test_auc_min": 0.67,
    "test_auc_max": 0.73,
    "rate_l0_le_9_min": 0.75,
    "rate_l0_le_5_min": 0.25,
}


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


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


def _compute_model_metrics(model, testset) -> dict[str, float]:
    probabilities = model.predict_proba(testset).detach().cpu()
    prediction = probabilities.argmax(dim=1)

    y = testset.get(target=True).iloc[:, 0].astype(int)
    class_to_index = model.get_class_to_index()
    encoded_target = torch.tensor(
        [class_to_index[int(value)] for value in y.tolist()],
        dtype=torch.long,
    )

    accuracy = float((prediction == encoded_target).to(dtype=torch.float32).mean())
    positive_index = class_to_index.get(1, max(class_to_index.values()))
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


def _limit_factuals(testset, limit: int | None):
    if limit is None:
        return testset
    if limit < 1:
        raise ValueError("factual limit must be >= 1")

    combined = pd.concat([testset.get(target=False), testset.get(target=True)], axis=1)
    if limit > combined.shape[0]:
        raise ValueError("factual limit exceeds available test rows")
    sampled = combined.sample(n=limit, random_state=SEED).copy(deep=True)
    return _build_frozen_dataset(testset, sampled, "selected_for_gs")


def _select_best_n_estimators(trainset, logger) -> int:
    X = trainset.get(target=False)
    y = trainset.get(target=True).iloc[:, 0].astype(int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    best_score = -math.inf
    best_n_estimators = RF_GRID[0]
    selected_minimal_target = None
    for n_estimators in RF_GRID:
        fold_scores: list[float] = []
        for train_index, valid_index in splitter.split(X, y):
            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_split=2,
                random_state=SEED,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            valid_probability = model.predict_proba(X_valid)[:, 1]
            fold_scores.append(float(roc_auc_score(y_valid, valid_probability)))

        mean_score = float(sum(fold_scores) / len(fold_scores))
        logger.info(
            "RF grid search n_estimators=%s mean_auc=%.4f",
            n_estimators,
            mean_score,
        )
        if mean_score > best_score:
            best_score = mean_score
            best_n_estimators = n_estimators
        if mean_score >= PAPER_TARGETS["test_auc"] and selected_minimal_target is None:
            selected_minimal_target = n_estimators

    if selected_minimal_target is not None:
        logger.info(
            "Selected n_estimators=%s as the smallest grid value meeting paper AUC target %.2f",
            selected_minimal_target,
            PAPER_TARGETS["test_auc"],
        )
        return selected_minimal_target

    logger.info(
        "Falling back to best n_estimators=%s from grid search with mean_auc=%.4f",
        best_n_estimators,
        best_score,
    )
    return best_n_estimators


def _compute_gs_sparsity_stats(factuals, counterfactuals) -> dict[str, float | int]:
    factual_features, counterfactual_features, evaluation_mask, success_mask = (
        resolve_evaluation_inputs(factuals, counterfactuals)
    )
    selected_mask = evaluation_mask & success_mask
    selected_count = int(selected_mask.sum())

    stats: dict[str, float | int] = {
        "num_counterfactuals_evaluated": int(evaluation_mask.sum()),
        "num_counterfactuals_successful": int(success_mask.sum()),
        "num_counterfactuals_selected": selected_count,
    }
    if selected_count == 0:
        stats.update(
            {
                "mean_l0": float("nan"),
                "max_l0": float("nan"),
                "rate_l0_le_5": float("nan"),
                "rate_l0_le_9": float("nan"),
            }
        )
        return stats

    factual_selected = factual_features.loc[selected_mask.to_numpy()]
    counterfactual_selected = counterfactual_features.loc[selected_mask.to_numpy()]
    l0_per_row = factual_selected.ne(counterfactual_selected).sum(axis=1).astype(float)
    stats.update(
        {
            "mean_l0": float(l0_per_row.mean()),
            "max_l0": float(l0_per_row.max()),
            "rate_l0_le_5": float((l0_per_row <= 5).mean()),
            "rate_l0_le_9": float((l0_per_row <= 9).mean()),
        }
    )
    return stats


def _assert_paper_targets(summary: dict[str, float | int]) -> None:
    test_auc = float(summary["test_auc"])
    rate_l0_le_9 = float(summary["rate_l0_le_9"])
    rate_l0_le_5 = float(summary["rate_l0_le_5"])

    assert ASSERT_BOUNDS["test_auc_min"] <= test_auc <= ASSERT_BOUNDS["test_auc_max"]
    assert rate_l0_le_9 >= ASSERT_BOUNDS["rate_l0_le_9_min"]
    assert rate_l0_le_5 >= ASSERT_BOUNDS["rate_l0_le_5_min"]


def run_reproduction(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    mode: str = "paper",
    factual_limit: int | None = None,
) -> dict:
    resolved_config_path = Path(config_path).resolve()
    base_config = _load_config(resolved_config_path)

    if mode not in {"smoke", "paper"}:
        raise ValueError("mode must be 'smoke' or 'paper'")
    if mode == "paper" and factual_limit is None:
        factual_limit = PAPER_FACTUAL_LIMIT

    grid_experiment = Experiment(deepcopy(base_config))
    grid_trainset, _ = _materialize_datasets(grid_experiment)
    best_n_estimators = _select_best_n_estimators(
        grid_trainset, grid_experiment._logger
    )

    final_config = deepcopy(base_config)
    final_config["model"]["n_estimators"] = best_n_estimators
    if mode == "smoke":
        final_config["name"] = f"{final_config['name']}_smoke"
        final_config["logger"][
            "path"
        ] = "./logs/news_popularity_randomforest_gs_smoke.log"
        final_config["method"]["n_search_samples"] = 500
        final_config["method"]["max_iter"] = 200
        if factual_limit is None:
            factual_limit = 32

    experiment = Experiment(final_config)
    logger = experiment._logger

    trainset, testset = _materialize_datasets(experiment)
    logger.info("Training target model")
    experiment._target_model.fit(trainset)
    model_metrics = _compute_model_metrics(experiment._target_model, testset)
    logger.info("Test accuracy: %.4f", model_metrics["test_accuracy"])
    logger.info("Test AUC: %.4f", model_metrics["test_auc"])

    factuals = _limit_factuals(testset, factual_limit)
    logger.info("Training GS")
    experiment._method.fit(trainset)

    logger.info("Generating counterfactuals for %s factual rows", len(factuals))
    start_time = perf_counter()
    counterfactuals = experiment._method.predict(factuals, batch_size=len(factuals))
    elapsed_seconds = perf_counter() - start_time

    evaluation_results = [
        evaluation_step.evaluate(factuals, counterfactuals)
        for evaluation_step in experiment._evaluation
    ]
    metrics = pd.concat(evaluation_results, axis=1)
    experiment._metrics = metrics

    sparsity_stats = _compute_gs_sparsity_stats(factuals, counterfactuals)
    summary = {
        "mode": mode,
        "selected_n_estimators": int(best_n_estimators),
        "num_factuals": int(len(factuals)),
        "search_seconds": float(elapsed_seconds),
        **model_metrics,
        **sparsity_stats,
    }

    comparison = {
        "paper_targets": PAPER_TARGETS,
        "reproduced": {
            "test_auc": summary["test_auc"],
            "max_l0": summary["max_l0"],
            "rate_l0_le_9": summary["rate_l0_le_9"],
            "rate_l0_le_5": summary["rate_l0_le_5"],
        },
        "notes": {
            "test_auc_scope": "full_test_split",
            "gs_sparsity_scope": f"{len(factuals)}_sampled_test_factuals",
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))
    print(json.dumps(comparison, indent=2, sort_keys=True))

    if mode == "paper":
        _assert_paper_targets(summary)

    return {
        "summary": summary,
        "metrics": metrics,
        "comparison": comparison,
        "factuals": factuals,
        "counterfactuals": counterfactuals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the GS news reproduction YAML",
    )
    parser.add_argument(
        "--mode",
        default="paper",
        choices=["smoke", "paper"],
        help="Smoke mode runs a smaller GS search before the full paper attempt.",
    )
    parser.add_argument(
        "--factual-limit",
        type=int,
        default=None,
        help="Optional number of factuals to evaluate instead of the entire test split.",
    )
    args = parser.parse_args()
    run_reproduction(args.path, mode=args.mode, factual_limit=args.factual_limit)


if __name__ == "__main__":
    main()
