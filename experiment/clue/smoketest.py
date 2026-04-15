from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import yaml

from evaluation.distance import DistanceEvaluation
from evaluation.validity import ValidityEvaluation
from experiment import Experiment

DEFAULT_CONFIG_PATH = Path(__file__).with_name(
    "compas_mlp_bayesian_clue_smoketest.yaml"
)


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Smoketest config must parse to a dictionary")
    return config


def run_smoketest(config_path: Path = DEFAULT_CONFIG_PATH):
    config = _load_config(config_path)
    device = _resolve_device()
    config["model"]["device"] = device
    config["method"]["device"] = device

    experiment = Experiment(config)
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

    trainset, testset = experiment._resolve_train_test(datasets)
    experiment._target_model.fit(trainset)
    experiment._method.fit(trainset)

    factuals = testset.clone()
    combined = pd.concat([testset.get(target=False), testset.get(target=True)], axis=1)
    sampled_df = combined.iloc[:16].copy(deep=True)
    factuals.update("testset", True, df=sampled_df)
    factuals.freeze()

    counterfactuals = experiment._method.predict(factuals, batch_size=16)
    metrics = pd.concat(
        [
            DistanceEvaluation(metrics=["l1"]).evaluate(factuals, counterfactuals),
            ValidityEvaluation().evaluate(factuals, counterfactuals),
        ],
        axis=1,
    )
    required_columns = {"distance_l1", "validity"}
    missing = required_columns.difference(metrics.columns)
    if missing:
        raise AssertionError(f"Missing smoketest metrics: {sorted(missing)}")
    return metrics


def main() -> None:
    metrics = run_smoketest()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
