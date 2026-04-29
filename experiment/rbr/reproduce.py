from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset.german.german import GermanDataset
from dataset.german_roar.german_roar import GermanRoarDataset
from method.rbr.rbr import RbrMethod
from model.mlp.mlp import MlpModel

DEFAULT_CURRENT_CONFIG = "./experiment/rbr/german_mlp_rbr_reproduce_current.yaml"
DEFAULT_FUTURE_CONFIG = "./experiment/rbr/german_mlp_rbr_reproduce_future.yaml"
PAPER_GERMAN_METRICS = {
    "present_accuracy": {"mean": 0.67, "std": 0.02},
    "present_auc": {"mean": 0.60, "std": 0.03},
    "shift_accuracy": {"mean": 0.66, "std": 0.23},
    "shift_auc": {"mean": 0.60, "std": 0.04},
}
DEFAULT_SWEEP_GRID = {
    "epsilon_pe": [0.0, 0.5, 1.0],
    "delta_plus": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "epsilon_op": [0.0, 0.5, 1.0],
}
NUMERICAL_FEATURES = ["age", "amount", "duration"]
TARGET_COLUMN = "credit_risk"
CATEGORICAL_FEATURE = "personal_status_sex"


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_raw_german_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    current_dataset = GermanDataset()
    shifted_dataset = GermanRoarDataset()
    current_df = current_dataset.snapshot().copy(deep=True)
    shifted_df = shifted_dataset.snapshot().copy(deep=True)
    return current_df, shifted_df


def _build_joint_transformer(
    current_df: pd.DataFrame,
    shifted_df: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], dict[str, list[str]]]:
    combined = pd.concat(
        [
            current_df.drop(columns=[TARGET_COLUMN]),
            shifted_df.drop(columns=[TARGET_COLUMN]),
        ],
        ignore_index=True,
    )
    categorical_features = [
        column for column in combined.columns if column not in NUMERICAL_FEATURES
    ]

    try:
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", categorical_encoder, categorical_features),
        ],
        sparse_threshold=0.0,
    )
    transformer.fit(combined)

    cat_transformer = transformer.named_transformers_["cat"]
    categories = list(cat_transformer.categories_[0].tolist())
    cat_columns = [f"{CATEGORICAL_FEATURE}_cat_{category}" for category in categories]
    feature_names = list(NUMERICAL_FEATURES) + cat_columns
    encoding_map = {CATEGORICAL_FEATURE: cat_columns}
    return transformer, feature_names, encoding_map


def _transform_features(
    transformer: ColumnTransformer,
    X: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    transformed = transformer.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return pd.DataFrame(transformed, index=X.index, columns=feature_names)


def _build_processed_metadata(
    template_dataset: GermanDataset | GermanRoarDataset,
    feature_names: list[str],
    encoding_map: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
    raw_feature_type = template_dataset.attr("raw_feature_type")
    raw_feature_mutability = template_dataset.attr("raw_feature_mutability")
    raw_feature_actionability = template_dataset.attr("raw_feature_actionability")

    encoded_feature_type: dict[str, str] = {}
    encoded_feature_mutability: dict[str, bool] = {}
    encoded_feature_actionability: dict[str, str] = {}

    categorical_columns = set(encoding_map[CATEGORICAL_FEATURE])
    for feature_name in feature_names:
        if feature_name in categorical_columns:
            encoded_feature_type[feature_name] = "binary"
            encoded_feature_mutability[feature_name] = bool(
                raw_feature_mutability[CATEGORICAL_FEATURE]
            )
            encoded_feature_actionability[feature_name] = str(
                raw_feature_actionability[CATEGORICAL_FEATURE]
            )
        else:
            encoded_feature_type[feature_name] = str(raw_feature_type[feature_name])
            encoded_feature_mutability[feature_name] = bool(
                raw_feature_mutability[feature_name]
            )
            encoded_feature_actionability[feature_name] = str(
                raw_feature_actionability[feature_name]
            )
    return (
        encoded_feature_type,
        encoded_feature_mutability,
        encoded_feature_actionability,
    )


def _make_frozen_processed_dataset(
    template_dataset: GermanDataset | GermanRoarDataset,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    encoding_map: dict[str, list[str]],
    dataset_flag: str,
) -> object:
    dataset = template_dataset.clone()
    combined = pd.concat([X.loc[:, feature_names], y.rename(TARGET_COLUMN)], axis=1)
    combined = combined.loc[:, [*feature_names, TARGET_COLUMN]]

    (
        encoded_feature_type,
        encoded_feature_mutability,
        encoded_feature_actionability,
    ) = _build_processed_metadata(
        template_dataset,
        feature_names,
        encoding_map,
    )

    dataset.update("encoding", deepcopy(encoding_map), df=combined)
    dataset.update("encoded_feature_type", encoded_feature_type)
    dataset.update("encoded_feature_mutability", encoded_feature_mutability)
    dataset.update("encoded_feature_actionability", encoded_feature_actionability)
    dataset.update(dataset_flag, True)
    dataset.freeze()
    return dataset


def _compute_model_metrics(model: MlpModel, testset) -> dict[str, float]:
    probabilities = model.predict_proba(testset).detach().cpu().numpy()
    prediction = probabilities.argmax(axis=1)

    y = testset.get(target=True).iloc[:, 0]
    class_to_index = model.get_class_to_index()
    encoded_target = np.array(
        [class_to_index[int(value)] for value in y.astype(int).tolist()],
        dtype=np.int64,
    )

    accuracy = float(np.mean(prediction == encoded_target))
    positive_index = class_to_index.get(1, max(class_to_index.values()))
    if len(set(encoded_target.tolist())) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(encoded_target, probabilities[:, positive_index]))
    return {"accuracy": accuracy, "auc": auc}


def _compare_metric_to_paper(
    observed: float,
    target_mean: float,
    target_std: float,
) -> dict[str, float | bool]:
    tolerance = max(float(target_std), 1e-12)
    delta = float(observed - target_mean)
    return {
        "observed": float(observed),
        "paper_mean": float(target_mean),
        "paper_std": float(target_std),
        "delta": delta,
        "abs_delta": abs(delta),
        "within_one_std": abs(delta) <= tolerance,
        "within_two_std": abs(delta) <= (2.0 * tolerance),
    }


def _build_classifier_paper_comparison(
    current_model_metrics: dict[str, float],
    shifted_model_metrics: dict[str, float],
) -> dict[str, dict[str, float | bool]]:
    return {
        "present_accuracy": _compare_metric_to_paper(
            current_model_metrics["accuracy"],
            PAPER_GERMAN_METRICS["present_accuracy"]["mean"],
            PAPER_GERMAN_METRICS["present_accuracy"]["std"],
        ),
        "present_auc": _compare_metric_to_paper(
            current_model_metrics["auc"],
            PAPER_GERMAN_METRICS["present_auc"]["mean"],
            PAPER_GERMAN_METRICS["present_auc"]["std"],
        ),
        "shift_accuracy": _compare_metric_to_paper(
            shifted_model_metrics["accuracy"],
            PAPER_GERMAN_METRICS["shift_accuracy"]["mean"],
            PAPER_GERMAN_METRICS["shift_accuracy"]["std"],
        ),
        "shift_auc": _compare_metric_to_paper(
            shifted_model_metrics["auc"],
            PAPER_GERMAN_METRICS["shift_auc"]["mean"],
            PAPER_GERMAN_METRICS["shift_auc"]["std"],
        ),
    }


def _select_recourse_factuals(model: MlpModel, testset, experiment_cfg: dict):
    factual_selection = str(experiment_cfg.get("factual_selection", "all")).lower()
    max_factuals = experiment_cfg.get("max_factuals")
    max_factuals = None if max_factuals is None else int(max_factuals)

    prediction_probabilities = model.predict_proba(testset).detach().cpu().numpy()
    prediction_indices = pd.Series(
        prediction_probabilities.argmax(axis=1),
        index=testset.get(target=False).index,
        dtype="int64",
    )
    desired_index = model.get_class_to_index()[1]
    negative_indices = prediction_indices.index[
        prediction_indices.ne(desired_index).to_numpy()
    ]

    if factual_selection == "all":
        selected_indices = list(negative_indices)
        if max_factuals is not None:
            selected_indices = selected_indices[:max_factuals]
    elif factual_selection == "negative_class":
        num_factuals = int(experiment_cfg.get("num_factuals", max_factuals or 100))
        if len(negative_indices) < num_factuals:
            raise ValueError("Not enough negative-class factuals for sampling")
        selected_indices = (
            pd.Series(negative_indices)
            .sample(
                n=num_factuals,
                random_state=int(experiment_cfg.get("factual_sample_seed", 42)),
            )
            .tolist()
        )
    else:
        raise ValueError(f"Unsupported factual_selection: {factual_selection}")

    factuals = testset.clone()
    factual_df = pd.concat([testset.get(target=False), testset.get(target=True)], axis=1)
    factual_df = factual_df.loc[selected_indices].copy(deep=True)
    factuals.update("reproduce_factuals", True, df=factual_df)
    factuals.freeze()
    if len(factuals) == 0:
        raise ValueError("No factuals selected for RBR reproduction")
    return factuals


def _compute_distance_metrics(factuals, counterfactuals) -> tuple[dict[str, float], int]:
    factual_features = factuals.get(target=False)
    counterfactual_features = counterfactuals.get(target=False).reindex(
        index=factual_features.index,
        columns=factual_features.columns,
    )
    success_mask = ~counterfactual_features.isna().any(axis=1)
    successful_count = int(success_mask.sum())
    if successful_count == 0:
        return (
            {
                "distance_l0": float("nan"),
                "distance_l1": float("nan"),
                "distance_l2": float("nan"),
                "distance_linf": float("nan"),
                "l1_cost": float("nan"),
            },
            0,
        )

    delta = (
        counterfactual_features.loc[success_mask].to_numpy(dtype=np.float64)
        - factual_features.loc[success_mask].to_numpy(dtype=np.float64)
    )
    l0 = np.sum(~np.isclose(delta, np.zeros_like(delta), atol=1e-05), axis=1)
    l1 = np.sum(np.abs(delta), axis=1, dtype=np.float32)
    l2 = np.linalg.norm(delta, ord=2, axis=1)
    linf = np.max(np.abs(delta), axis=1).astype(np.float32)
    return (
        {
            "distance_l0": float(np.mean(l0)),
            "distance_l1": float(np.mean(l1)),
            "distance_l2": float(np.mean(l2)),
            "distance_linf": float(np.mean(linf)),
            "l1_cost": float(np.mean(l1)),
        },
        successful_count,
    )


def _compute_current_validity(current_model: MlpModel, counterfactuals) -> float:
    counterfactual_features = counterfactuals.get(target=False)
    success_mask = ~counterfactual_features.isna().any(axis=1)
    denominator = int(len(counterfactual_features))
    if denominator == 0:
        return float("nan")
    if int(success_mask.sum()) == 0:
        return 0.0

    probabilities = current_model.get_prediction(
        counterfactual_features.loc[success_mask],
        proba=True,
    ).detach().cpu().numpy()
    positive_probability = probabilities[:, current_model.get_class_to_index()[1]]
    return float(np.sum(positive_probability >= 0.5) / denominator)


def _compute_future_validity(
    future_models: list[MlpModel],
    counterfactuals,
) -> float:
    counterfactual_features = counterfactuals.get(target=False)
    success_mask = ~counterfactual_features.isna().any(axis=1)
    denominator = int(len(counterfactual_features))
    if denominator == 0:
        return float("nan")
    if int(success_mask.sum()) == 0:
        return 0.0

    cf_success = counterfactual_features.loc[success_mask]
    validities = []
    for future_model in future_models:
        probabilities = future_model.get_prediction(
            cf_success,
            proba=True,
        ).detach().cpu().numpy()
        positive_probability = probabilities[:, future_model.get_class_to_index()[1]]
        validities.append((positive_probability >= 0.5).astype(np.float32))
    stacked = np.stack(validities, axis=0)
    per_instance_future_validity = stacked.mean(axis=0)
    return float(np.sum(per_instance_future_validity) / denominator)


def _build_mlp_from_config(config: dict, device: str) -> MlpModel:
    model_cfg = deepcopy(config["model"])
    return MlpModel(
        seed=model_cfg.get("seed", 42),
        device=device,
        epochs=model_cfg.get("epochs", 1000),
        learning_rate=model_cfg.get("learning_rate", 0.001),
        batch_size=model_cfg.get("batch_size"),
        layers=model_cfg.get("layers"),
        optimizer=model_cfg.get("optimizer", "adam"),
        criterion=model_cfg.get("criterion", "bce"),
        output_activation=model_cfg.get("output_activation", "sigmoid"),
        pretrained_path=model_cfg.get("pretrained_path"),
        save_name=model_cfg.get("save_name"),
        weight_decay=model_cfg.get("weight_decay", 0.0),
        loss_reduction=model_cfg.get("loss_reduction", "mean"),
        xavier_uniform_init=model_cfg.get("xavier_uniform_init", False),
        early_stop_tol=model_cfg.get("early_stop_tol"),
        early_stop_patience=model_cfg.get("early_stop_patience"),
    )


def _instantiate_rbr_method(
    method_cfg: dict[str, Any],
    target_model: MlpModel,
    device: str,
) -> RbrMethod:
    return RbrMethod(
        target_model=target_model,
        seed=method_cfg.get("seed", 42),
        device=device,
        desired_class=method_cfg.get("desired_class", 1),
        num_samples=method_cfg.get("num_samples", 200),
        perturb_radius=method_cfg.get("perturb_radius", 0.2),
        delta_plus=method_cfg.get("delta_plus", 0.2),
        sigma=method_cfg.get("sigma", 1.0),
        epsilon_op=method_cfg.get("epsilon_op", 0.0),
        epsilon_pe=method_cfg.get("epsilon_pe", 0.0),
        max_iter=method_cfg.get("max_iter", 500),
        clamp=method_cfg.get("clamp", False),
        enforce_encoding=method_cfg.get("enforce_encoding", False),
        random_state=method_cfg.get("random_state", 42),
        verbose=method_cfg.get("verbose", False),
    )


def _split_current_data(
    current_df: pd.DataFrame,
    train_split: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = current_df.drop(columns=[TARGET_COLUMN])
    y = current_df[TARGET_COLUMN].astype(int)
    return train_test_split(
        X,
        y,
        train_size=train_split,
        random_state=random_state,
        stratify=y,
    )


def _split_shift_data(
    shifted_df: pd.DataFrame,
    train_split: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = shifted_df.drop(columns=[TARGET_COLUMN])
    y = shifted_df[TARGET_COLUMN].astype(int)
    return train_test_split(
        X,
        y,
        train_size=train_split,
        random_state=random_state,
        stratify=y,
    )


def _resolve_future_model_random_states(future_cfg: dict) -> list[int]:
    experiment_cfg = future_cfg.get("experiment", {})
    if "shifted_model_random_states" in experiment_cfg:
        return [int(value) for value in experiment_cfg["shifted_model_random_states"]]

    num_future_models = int(experiment_cfg.get("num_future_models", 100))
    future_model_seed_start = int(experiment_cfg.get("future_model_seed_start", 0))
    return list(range(future_model_seed_start, future_model_seed_start + num_future_models))


def _build_future_trainsets(
    template_dataset: GermanDataset,
    transformer: ColumnTransformer,
    feature_names: list[str],
    encoding_map: dict[str, list[str]],
    current_X_train: pd.DataFrame,
    current_y_train: pd.Series,
    shifted_df: pd.DataFrame,
    future_cfg: dict,
) -> list[object]:
    shifted_X = shifted_df.drop(columns=[TARGET_COLUMN])
    shifted_y = shifted_df[TARGET_COLUMN].astype(int)

    experiment_cfg = future_cfg.get("experiment", {})
    arrival_fraction = float(experiment_cfg.get("arrival_fraction", 0.2))
    if arrival_fraction < 0.0 or arrival_fraction > 1.0:
        raise ValueError("arrival_fraction must satisfy 0.0 <= arrival_fraction <= 1.0")

    future_trainsets = []
    for random_state in _resolve_future_model_random_states(future_cfg):
        if arrival_fraction == 0.0:
            arrival_X = shifted_X.iloc[0:0].copy(deep=True)
            arrival_y = shifted_y.iloc[0:0].copy(deep=True)
        else:
            arrival_X, _, arrival_y, _ = train_test_split(
                shifted_X,
                shifted_y,
                train_size=arrival_fraction,
                random_state=int(random_state),
                stratify=shifted_y,
            )

        future_X_raw = pd.concat([current_X_train, arrival_X], ignore_index=True)
        future_y = pd.concat([current_y_train, arrival_y], ignore_index=True)
        future_X = _transform_features(transformer, future_X_raw, feature_names)
        future_trainsets.append(
            _make_frozen_processed_dataset(
                template_dataset=template_dataset,
                X=future_X,
                y=future_y,
                feature_names=feature_names,
                encoding_map=encoding_map,
                dataset_flag="trainset",
            )
        )
    return future_trainsets


def _resolve_sweep_grid(current_cfg: dict) -> dict[str, list[float]]:
    experiment_cfg = current_cfg.get("experiment", {})
    configured = experiment_cfg.get("sweep", {})
    resolved: dict[str, list[float]] = {}
    for parameter, default_values in DEFAULT_SWEEP_GRID.items():
        raw_values = configured.get(parameter, default_values)
        resolved[parameter] = [float(value) for value in raw_values]
    return resolved


def _build_method_variants(current_cfg: dict) -> list[dict[str, Any]]:
    experiment_cfg = current_cfg.get("experiment", {})
    reproduction_mode = str(
        experiment_cfg.get("reproduction_mode", "paper_sweep")
    ).lower()
    base_method_cfg = deepcopy(current_cfg["method"])

    if reproduction_mode == "single_point":
        return [
            {
                "variant_label": "single_point",
                "sweep_parameter": None,
                "sweep_value": None,
                "method_cfg": base_method_cfg,
            }
        ]

    if reproduction_mode != "paper_sweep":
        raise ValueError(f"Unsupported reproduction_mode: {reproduction_mode}")

    variants: list[dict[str, Any]] = []
    for parameter, values in _resolve_sweep_grid(current_cfg).items():
        for value in values:
            method_cfg = deepcopy(base_method_cfg)
            method_cfg[parameter] = float(value)
            variants.append(
                {
                    "variant_label": f"{parameter}={value:.4f}",
                    "sweep_parameter": parameter,
                    "sweep_value": float(value),
                    "method_cfg": method_cfg,
                }
            )
    return variants


def _evaluate_method_variant(
    variant: dict[str, Any],
    current_model: MlpModel,
    current_trainset,
    factuals,
    future_models: list[MlpModel],
    device: str,
) -> dict[str, Any]:
    method_cfg = deepcopy(variant["method_cfg"])
    rbr_method = _instantiate_rbr_method(
        method_cfg=method_cfg,
        target_model=current_model,
        device=device,
    )
    rbr_method.fit(current_trainset)
    counterfactuals = rbr_method.predict(factuals)

    distance_metrics, num_successful = _compute_distance_metrics(factuals, counterfactuals)
    current_validity = _compute_current_validity(current_model, counterfactuals)
    future_validity = _compute_future_validity(future_models, counterfactuals)

    return {
        "variant_label": str(variant["variant_label"]),
        "sweep_parameter": variant["sweep_parameter"],
        "sweep_value": variant["sweep_value"],
        "method": {
            "delta_plus": float(method_cfg["delta_plus"]),
            "sigma": float(method_cfg["sigma"]),
            "epsilon_op": float(method_cfg["epsilon_op"]),
            "epsilon_pe": float(method_cfg["epsilon_pe"]),
            "num_samples": int(method_cfg["num_samples"]),
            "perturb_radius": float(method_cfg["perturb_radius"]),
            "max_iter": int(method_cfg["max_iter"]),
        },
        "num_factuals": int(len(factuals)),
        "num_successful": int(num_successful),
        "current_validity": float(current_validity),
        "future_validity": float(future_validity),
        **distance_metrics,
        "train_max_distance": float(rbr_method._train_max_distance),
        "effective_perturb_radius": float(
            rbr_method._perturb_radius * rbr_method._train_max_distance
        ),
    }


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}

    best_future = max(
        results,
        key=lambda item: (
            float(item["future_validity"]),
            float(item["current_validity"]),
            -float(item["l1_cost"]) if np.isfinite(item["l1_cost"]) else float("-inf"),
        ),
    )

    fully_valid = [
        item
        for item in results
        if np.isfinite(item["current_validity"]) and float(item["current_validity"]) >= 0.999999
    ]
    lowest_cost_full_valid = None
    if fully_valid:
        lowest_cost_full_valid = min(
            fully_valid,
            key=lambda item: float(item["l1_cost"]),
        )

    return {
        "best_future_validity": best_future,
        "lowest_cost_full_current_validity": lowest_cost_full_valid,
    }


def _build_warnings(
    classifier_comparison: dict[str, dict[str, float | bool]],
    results: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    for metric_name, comparison in classifier_comparison.items():
        if not bool(comparison["within_two_std"]):
            warnings.append(
                f"{metric_name} is outside two paper standard deviations: "
                f"observed={comparison['observed']:.4f}, "
                f"paper={comparison['paper_mean']:.4f}±{comparison['paper_std']:.4f}"
            )

    if results and not any(item["num_successful"] > 0 for item in results):
        warnings.append("No successful counterfactuals were found for any evaluated RBR setting.")

    return warnings


def run_reproduction(
    current_config_path: str = DEFAULT_CURRENT_CONFIG,
    future_config_path: str = DEFAULT_FUTURE_CONFIG,
) -> dict[str, Any]:
    device = _resolve_device()
    current_cfg = _load_config((PROJECT_ROOT / current_config_path).resolve())
    future_cfg = _load_config((PROJECT_ROOT / future_config_path).resolve())

    current_raw_df, shifted_raw_df = _load_raw_german_frames()
    transformer, feature_names, encoding_map = _build_joint_transformer(
        current_raw_df,
        shifted_raw_df,
    )

    current_experiment_cfg = current_cfg.get("experiment", {})
    train_split = float(current_experiment_cfg.get("train_split", 0.8))
    split_random_state = int(current_experiment_cfg.get("split_random_state", 42))
    current_X_train_raw, current_X_test_raw, current_y_train, current_y_test = (
        _split_current_data(
            current_raw_df,
            train_split=train_split,
            random_state=split_random_state,
        )
    )
    current_X_train = _transform_features(
        transformer,
        current_X_train_raw,
        feature_names,
    )
    current_X_test = _transform_features(
        transformer,
        current_X_test_raw,
        feature_names,
    )

    current_template_dataset = GermanDataset()
    current_trainset = _make_frozen_processed_dataset(
        template_dataset=current_template_dataset,
        X=current_X_train,
        y=current_y_train,
        feature_names=feature_names,
        encoding_map=encoding_map,
        dataset_flag="trainset",
    )
    current_testset = _make_frozen_processed_dataset(
        template_dataset=current_template_dataset,
        X=current_X_test,
        y=current_y_test,
        feature_names=feature_names,
        encoding_map=encoding_map,
        dataset_flag="testset",
    )

    current_model = _build_mlp_from_config(current_cfg, device=device)
    current_model.fit(current_trainset)
    current_model_metrics = _compute_model_metrics(current_model, current_testset)

    shift_experiment_cfg = future_cfg.get("experiment", {})
    shift_train_split = float(shift_experiment_cfg.get("shift_train_split", 0.8))
    shift_split_random_state = int(
        shift_experiment_cfg.get("shift_split_random_state", 42)
    )
    shifted_X_train_raw, shifted_X_test_raw, shifted_y_train, shifted_y_test = (
        _split_shift_data(
            shifted_raw_df,
            train_split=shift_train_split,
            random_state=shift_split_random_state,
        )
    )
    shifted_X_train = _transform_features(
        transformer,
        shifted_X_train_raw,
        feature_names,
    )
    shifted_X_test = _transform_features(
        transformer,
        shifted_X_test_raw,
        feature_names,
    )
    shifted_template_dataset = GermanRoarDataset()
    shifted_trainset = _make_frozen_processed_dataset(
        template_dataset=shifted_template_dataset,
        X=shifted_X_train,
        y=shifted_y_train,
        feature_names=feature_names,
        encoding_map=encoding_map,
        dataset_flag="trainset",
    )
    shifted_testset = _make_frozen_processed_dataset(
        template_dataset=shifted_template_dataset,
        X=shifted_X_test,
        y=shifted_y_test,
        feature_names=feature_names,
        encoding_map=encoding_map,
        dataset_flag="testset",
    )

    shifted_model = _build_mlp_from_config(future_cfg, device=device)
    shifted_model.fit(shifted_trainset)
    shifted_model_metrics = _compute_model_metrics(shifted_model, shifted_testset)

    factuals = _select_recourse_factuals(
        current_model,
        current_testset,
        current_experiment_cfg,
    )

    future_trainsets = _build_future_trainsets(
        template_dataset=current_template_dataset,
        transformer=transformer,
        feature_names=feature_names,
        encoding_map=encoding_map,
        current_X_train=current_X_train_raw,
        current_y_train=current_y_train,
        shifted_df=shifted_raw_df,
        future_cfg=future_cfg,
    )
    future_models: list[MlpModel] = []
    future_model_metrics_on_present_test: list[dict[str, float]] = []
    for future_trainset in future_trainsets:
        future_model = _build_mlp_from_config(future_cfg, device=device)
        future_model.fit(future_trainset)
        future_models.append(future_model)
        future_model_metrics_on_present_test.append(
            _compute_model_metrics(future_model, current_testset)
        )

    if future_model_metrics_on_present_test:
        simulated_future_metrics = {
            "accuracy": float(
                np.mean(
                    [
                        metrics["accuracy"]
                        for metrics in future_model_metrics_on_present_test
                    ]
                )
            ),
            "auc": float(
                np.mean(
                    [metrics["auc"] for metrics in future_model_metrics_on_present_test]
                )
            ),
            "num_future_models": int(len(future_model_metrics_on_present_test)),
        }
    else:
        simulated_future_metrics = {
            "accuracy": float("nan"),
            "auc": float("nan"),
            "num_future_models": 0,
        }

    results = [
        _evaluate_method_variant(
            variant=variant,
            current_model=current_model,
            current_trainset=current_trainset,
            factuals=factuals,
            future_models=future_models,
            device=device,
        )
        for variant in _build_method_variants(current_cfg)
    ]

    classifier_comparison = _build_classifier_paper_comparison(
        current_model_metrics=current_model_metrics,
        shifted_model_metrics=shifted_model_metrics,
    )
    warnings = _build_warnings(classifier_comparison, results)

    return {
        "device": device,
        "setup": {
            "train_split_d1": train_split,
            "arrival_fraction_d2": float(shift_experiment_cfg.get("arrival_fraction", 0.2)),
            "num_future_models": int(len(future_models)),
            "reproduction_mode": str(
                current_experiment_cfg.get("reproduction_mode", "paper_sweep")
            ).lower(),
            "factual_selection": str(current_experiment_cfg.get("factual_selection", "all")).lower(),
            "max_factuals": int(current_experiment_cfg.get("max_factuals", len(factuals))),
            "selected_factuals": int(len(factuals)),
        },
        "classifier_metrics": {
            "present_d1": current_model_metrics,
            "shift_d2": shifted_model_metrics,
            "simulated_future_models_on_present_test": simulated_future_metrics,
            "paper_comparison": classifier_comparison,
        },
        "results": results,
        "summary": _summarize_results(results),
        "warnings": warnings,
    }


def test_run_experiment():
    return run_reproduction()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-config", default=DEFAULT_CURRENT_CONFIG)
    parser.add_argument("--future-config", default=DEFAULT_FUTURE_CONFIG)
    args = parser.parse_args()

    summary = run_reproduction(
        current_config_path=args.current_config,
        future_config_path=args.future_config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
