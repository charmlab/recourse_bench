from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from experiment import Experiment


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _apply_device(config: dict, device: str) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["device"] = device
    cfg["method"]["device"] = device
    return cfg


def _materialize_datasets(experiment: Experiment) -> list[object]:
    datasets = [experiment._raw_dataset]
    for preprocess_step in experiment._preprocess:
        next_datasets: list[object] = []
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
    return datasets


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


def _build_dataset_like(dataset, df: pd.DataFrame, flag: str | None = None):
    output = dataset.clone()
    if flag is None:
        output.update("reproduced", True, df=df)
    else:
        output.update(flag, True, df=df)
    output.freeze()
    return output


def _select_factuals(
    testset,
    model,
    desired_class: int | str,
    num_factuals: int,
    seed: int,
):
    class_to_index = model.get_class_to_index()
    desired_index = int(class_to_index[desired_class])
    predictions = model.predict(testset).argmax(dim=1).detach().cpu().numpy()
    keep_mask = predictions != desired_index
    if int(keep_mask.sum()) < num_factuals:
        raise ValueError(
            "Not enough factuals outside desired_class to sample the requested count"
        )

    combined = pd.concat([testset.get(target=False), testset.get(target=True)], axis=1)
    factual_df = combined.loc[keep_mask].sample(n=num_factuals, random_state=seed)
    factual_df = factual_df.loc[:, testset.ordered_features()].copy(deep=True)
    return _build_dataset_like(testset, factual_df, flag="testset")


def _combine_future_trainset(
    current_trainset, future_full_dataset, seed: int, frac: float
):
    current_train_df = pd.concat(
        [current_trainset.get(target=False), current_trainset.get(target=True)], axis=1
    )
    future_full_df = pd.concat(
        [future_full_dataset.get(target=False), future_full_dataset.get(target=True)],
        axis=1,
    )
    shifted_subset = future_full_df.sample(frac=frac, random_state=seed).copy(deep=True)
    combined_df = pd.concat([current_train_df, shifted_subset], ignore_index=True)
    combined_df = combined_df.loc[:, current_trainset.ordered_features()].copy(
        deep=True
    )
    return _build_dataset_like(current_trainset, combined_df, flag="trainset")


def _validate_feature_alignment(current_trainset, future_full_dataset) -> None:
    current_columns = current_trainset.ordered_features()
    future_columns = future_full_dataset.ordered_features()
    if current_columns != future_columns:
        raise ValueError(
            "Current and future processed datasets do not share the same column order"
        )


def _positive_probability(model, features: pd.DataFrame) -> float:
    probabilities = model.get_prediction(features, proba=True).detach().cpu()
    positive_index = int(
        model.get_class_to_index().get(1, max(model.get_class_to_index().values()))
    )
    return float(probabilities[:, positive_index].item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-config",
        default="./experiment/rbr/german_mlp_rbr_reproduce_current.yaml",
    )
    parser.add_argument(
        "--future-config",
        default="./experiment/rbr/german_mlp_rbr_reproduce_future.yaml",
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
    quiet_future_cfg = deepcopy(future_cfg)
    quiet_future_cfg["logger"] = {"level": "CRITICAL", "path": None}

    reproduce_cfg = current_cfg.get("reproduce", {})
    num_factuals = int(reproduce_cfg.get("num_factuals", 10))
    factual_seed = int(reproduce_cfg.get("factual_seed", 54321))
    future_fraction = float(reproduce_cfg.get("future_fraction", 0.2))
    future_seeds = list(reproduce_cfg.get("future_seeds", [1, 2, 3, 4, 5]))
    current_validity_min = float(reproduce_cfg.get("current_validity_min", 0.9))
    future_validity_min = float(reproduce_cfg.get("future_validity_min", 0.7))

    current_experiment = Experiment(current_cfg)
    current_datasets = _materialize_datasets(current_experiment)
    current_trainset, current_testset = current_experiment._resolve_train_test(
        current_datasets
    )

    current_experiment._target_model.fit(current_trainset)
    current_model_metrics = _compute_model_metrics(
        current_experiment._target_model,
        current_testset,
    )

    current_experiment._method.fit(current_trainset)
    desired_class = current_experiment._method._desired_class
    if desired_class is None:
        raise ValueError("RBR reproduction requires desired_class to be set")
    factuals = _select_factuals(
        current_testset,
        current_experiment._target_model,
        desired_class=desired_class,
        num_factuals=num_factuals,
        seed=factual_seed,
    )
    counterfactuals = current_experiment._method.predict(factuals)

    future_template_experiment = Experiment(quiet_future_cfg)
    future_template_datasets = _materialize_datasets(future_template_experiment)
    if len(future_template_datasets) != 1:
        raise ValueError(
            "Future reproduction config must materialize exactly one dataset"
        )
    future_full_dataset = future_template_datasets[0]
    _validate_feature_alignment(current_trainset, future_full_dataset)

    future_models = []
    for seed in future_seeds:
        future_trainset = _combine_future_trainset(
            current_trainset,
            future_full_dataset,
            seed=int(seed),
            frac=future_fraction,
        )
        future_experiment = Experiment(quiet_future_cfg)
        future_experiment._target_model.fit(future_trainset)
        future_models.append(future_experiment._target_model)

    factual_features = factuals.get(target=False)
    counterfactual_features = counterfactuals.get(target=False)

    running_cost = 0.0
    running_current_validity = 0.0
    running_future_validity = 0.0
    num_instances = factual_features.shape[0]

    print(f"Device: {device}")
    print(
        "Current model metrics:",
        f"accuracy={current_model_metrics['test_accuracy']:.4f}",
        f"auc={current_model_metrics['test_auc']:.4f}",
    )
    print(f"Num factuals: {num_instances}")

    for row_index, factual_index in enumerate(factual_features.index):
        factual_row = factual_features.loc[[factual_index]].copy(deep=True)
        counterfactual_row = counterfactual_features.loc[[factual_index]].copy(
            deep=True
        )

        feasible = not bool(counterfactual_row.isna().any(axis=1).iloc[0])
        if feasible:
            l1_cost = float(
                np.linalg.norm(
                    (
                        counterfactual_row.to_numpy(dtype="float32")
                        - factual_row.to_numpy(dtype="float32")
                    ).reshape(-1),
                    ord=1,
                )
            )
            current_validity = float(
                _positive_probability(
                    current_experiment._target_model, counterfactual_row
                )
                >= 0.5
            )
            future_scores = [
                float(_positive_probability(model, counterfactual_row) >= 0.5)
                for model in future_models
            ]
            future_validity = float(np.mean(future_scores))
        else:
            l1_cost = float("inf")
            current_validity = 0.0
            future_validity = 0.0

        running_cost += l1_cost
        running_current_validity += current_validity
        running_future_validity += future_validity

        print(
            f"Instance {row_index}: "
            f"L1 cost = {l1_cost}, "
            f"Current Validity = {current_validity}, "
            f"Future Validity = {future_validity}, "
            f"Feasible = {feasible}"
        )

    average_cost = running_cost / num_instances
    average_current_validity = running_current_validity / num_instances
    average_future_validity = running_future_validity / num_instances

    print(
        "Average:",
        f"L1 cost = {average_cost},",
        f"Current Validity = {average_current_validity},",
        f"Future Validity = {average_future_validity}",
    )

    assert average_current_validity >= current_validity_min, (
        f"Average current validity {average_current_validity:.4f} "
        f"is below required threshold {current_validity_min:.4f}"
    )
    assert average_future_validity >= future_validity_min, (
        f"Average future validity {average_future_validity:.4f} "
        f"is below required threshold {future_validity_min:.4f}"
    )


if __name__ == "__main__":
    main()
