from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

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
    "present_accuracy": "0.67 ± 0.02",
    "present_auc": "0.60 ± 0.03",
    "shift_accuracy": "0.66 ± 0.23",
    "shift_auc": "0.60 ± 0.04",
}
RECOURSE_BENCHMARKS_CURRENT_VALIDITY_MIN = 0.9
RECOURSE_BENCHMARKS_FUTURE_VALIDITY_MIN = 0.7
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


def _build_reference_transformer(
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
        column
        for column in combined.columns
        if column not in NUMERICAL_FEATURES
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
    return pd.DataFrame(
        transformed,
        index=X.index,
        columns=feature_names,
    )


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
    combined = pd.concat(
        [X.loc[:, feature_names], y.rename(TARGET_COLUMN)],
        axis=1,
    )
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
    return {"test_accuracy": accuracy, "test_auc": auc}


def _select_recourse_factuals(model: MlpModel, testset, experiment_cfg: dict):
    factual_selection = str(
        experiment_cfg.get("factual_selection", "all")
    ).lower()
    prediction_probabilities = model.predict_proba(testset).detach().cpu().numpy()
    prediction_indices = pd.Series(
        prediction_probabilities.argmax(axis=1),
        index=testset.get(target=False).index,
        dtype="int64",
    )
    desired_index = model.get_class_to_index()[1]
    negative_mask = prediction_indices.ne(desired_index)

    if factual_selection == "all":
        keep_mask = negative_mask
    elif factual_selection == "negative_class":
        num_factuals = int(experiment_cfg.get("num_factuals", 5))
        negative_indices = prediction_indices.index[negative_mask.to_numpy()]
        if len(negative_indices) < num_factuals:
            raise ValueError("Not enough negative-class factuals for sampling")
        chosen = (
            pd.Series(negative_indices)
            .sample(n=num_factuals, random_state=42)
            .tolist()
        )
        keep_mask = pd.Series(False, index=prediction_indices.index, dtype=bool)
        keep_mask.loc[chosen] = True
    else:
        raise ValueError(f"Unsupported factual_selection: {factual_selection}")

    factuals = testset.clone()
    factual_df = pd.concat([testset.get(target=False), testset.get(target=True)], axis=1)
    factual_df = factual_df.loc[keep_mask].copy(deep=True)
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


def _build_rbr_from_config(
    config: dict,
    target_model: MlpModel,
    device: str,
) -> RbrMethod:
    method_cfg = deepcopy(config["method"])
    return RbrMethod(
        target_model=target_model,
        seed=method_cfg.get("seed", 42),
        device=device,
        desired_class=method_cfg.get("desired_class", 1),
        num_samples=method_cfg.get("num_samples", 200),
        perturb_radius=method_cfg.get("perturb_radius", 0.2),
        delta_plus=method_cfg.get("delta_plus", 1.0),
        sigma=method_cfg.get("sigma", 1.0),
        epsilon_op=method_cfg.get("epsilon_op", 0.5),
        epsilon_pe=method_cfg.get("epsilon_pe", 1.0),
        max_iter=method_cfg.get("max_iter", 1000),
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
    shifted_train_fraction = float(experiment_cfg.get("shifted_train_fraction", 0.5))
    shifted_model_random_states = list(
        experiment_cfg.get("shifted_model_random_states", [1, 2, 3, 4, 5])
    )

    future_trainsets = []
    for random_state in shifted_model_random_states:
        shifted_X_train, _, shifted_y_train, _ = train_test_split(
            shifted_X,
            shifted_y,
            train_size=shifted_train_fraction,
            random_state=int(random_state),
            stratify=shifted_y,
        )
        future_X_raw = pd.concat([current_X_train, shifted_X_train], ignore_index=True)
        future_y = pd.concat([current_y_train, shifted_y_train], ignore_index=True)
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


def _build_summary(
    current_model_metrics: dict[str, float],
    future_model_metrics: dict[str, float],
    distance_metrics: dict[str, float],
    num_factuals: int,
    num_successful: int,
    current_validity: float,
    future_validity: float,
    device: str,
) -> dict[str, float | int | str]:
    return {
        "device": device,
        "num_factuals": num_factuals,
        "num_successful": num_successful,
        "current_validity": current_validity,
        "future_validity": future_validity,
        **distance_metrics,
        "current_model_test_accuracy": current_model_metrics["test_accuracy"],
        "current_model_test_auc": current_model_metrics["test_auc"],
        "future_model_test_accuracy": future_model_metrics["test_accuracy"],
        "future_model_test_auc": future_model_metrics["test_auc"],
    }


def _assert_summary(summary: dict[str, float | int | str]) -> None:
    current_validity = float(summary["current_validity"])
    future_validity = float(summary["future_validity"])
    assert current_validity >= RECOURSE_BENCHMARKS_CURRENT_VALIDITY_MIN, (
        "current_validity does not satisfy recourse_benchmarks RBR threshold: "
        f"{current_validity} < {RECOURSE_BENCHMARKS_CURRENT_VALIDITY_MIN}"
    )
    assert future_validity >= RECOURSE_BENCHMARKS_FUTURE_VALIDITY_MIN, (
        "future_validity does not satisfy recourse_benchmarks RBR threshold: "
        f"{future_validity} < {RECOURSE_BENCHMARKS_FUTURE_VALIDITY_MIN}"
    )


def run_reproduction(
    current_config_path: str = DEFAULT_CURRENT_CONFIG,
    future_config_path: str = DEFAULT_FUTURE_CONFIG,
) -> dict[str, float | int | str]:
    device = _resolve_device()
    current_cfg = _load_config((PROJECT_ROOT / current_config_path).resolve())
    future_cfg = _load_config((PROJECT_ROOT / future_config_path).resolve())

    current_raw_df, shifted_raw_df = _load_raw_german_frames()
    transformer, feature_names, encoding_map = _build_reference_transformer(
        current_raw_df,
        shifted_raw_df,
    )

    train_split = float(current_cfg.get("experiment", {}).get("train_split", 0.8))
    split_random_state = int(
        current_cfg.get("experiment", {}).get("split_random_state", 42)
    )
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

    rbr_method = _build_rbr_from_config(current_cfg, current_model, device=device)
    rbr_method.fit(current_trainset)
    factuals = _select_recourse_factuals(
        current_model,
        current_testset,
        current_cfg.get("experiment", {}),
    )
    counterfactuals = rbr_method.predict(factuals)

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
    future_models = []
    future_model_metrics = []
    for future_trainset in future_trainsets:
        future_model = _build_mlp_from_config(future_cfg, device=device)
        future_model.fit(future_trainset)
        future_models.append(future_model)
        future_model_metrics.append(_compute_model_metrics(future_model, current_testset))

    mean_future_model_metrics = {
        "test_accuracy": float(
            np.mean([metric["test_accuracy"] for metric in future_model_metrics])
        ),
        "test_auc": float(
            np.mean([metric["test_auc"] for metric in future_model_metrics])
        ),
    }

    distance_metrics, num_successful = _compute_distance_metrics(
        factuals,
        counterfactuals,
    )
    current_validity = _compute_current_validity(current_model, counterfactuals)
    future_validity = _compute_future_validity(future_models, counterfactuals)

    summary = _build_summary(
        current_model_metrics=current_model_metrics,
        future_model_metrics=mean_future_model_metrics,
        distance_metrics=distance_metrics,
        num_factuals=len(factuals),
        num_successful=num_successful,
        current_validity=current_validity,
        future_validity=future_validity,
        device=device,
    )
    _assert_summary(summary)
    return summary


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
