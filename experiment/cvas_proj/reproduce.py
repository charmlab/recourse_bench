from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dataset  # noqa: F401
import method  # noqa: F401
import model  # noqa: F401
import preprocess  # noqa: F401
import numpy as np
import pandas as pd
import torch
import yaml
from model.mlp import mlp as mlp_module
from preprocess.common import EncodePreProcess, ScalePreProcess, SplitPreProcess
from preprocess.preprocess_object import PreProcessObject
from sklearn.model_selection import KFold, train_test_split
from utils.caching import set_cache_dir
from utils.registry import get_registry
from utils.seed import seed_context

PROFILE_DEFAULTS = {
    "smoke": {
        "current_epochs": 100,
        "future_epochs": 100,
        "num_samples": 100,
        "num_future_models": 3,
        "max_ins": 5,
        "rho_neg_values": [0.0, 5.0],
    },
    "german_reference": {
        "current_epochs": None,
        "future_epochs": None,
        "num_samples": None,
        "num_future_models": 100,
        "max_ins": 200,
        "rho_neg_values": [float(value) for value in range(11)],
    },
}


class CvasGermanReferencePreProcess(PreProcessObject):
    _PAIR_DATASET_NAME = {
        "german": "german_roar",
        "german_roar": "german",
    }

    _PERSONAL_STATUS_MAP = {
        "german": {
            1: "A91",
            2: "A92",
            3: "A94",
            5: "A93",
        },
        "german_roar": {
            1: "A91",
            2: "A92",
            3: "A93",
            4: "A94",
        },
    }

    def __init__(
        self,
        seed: int | None = None,
        paired_dataset_name: str | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._paired_dataset_name = paired_dataset_name

    def _resolve_dataset_name(self, input_dataset) -> str:
        dataset_name = str(input_dataset.attr("name")).lower()
        if dataset_name not in self._PAIR_DATASET_NAME:
            raise ValueError(
                "CvasGermanReferencePreProcess supports ['german', 'german_roar'] only"
            )
        return dataset_name

    def _load_paired_snapshot(self, paired_dataset_name: str) -> pd.DataFrame:
        dataset_registry = get_registry("Dataset")
        if paired_dataset_name not in dataset_registry:
            raise ValueError(f"Unknown paired dataset: {paired_dataset_name}")
        paired_dataset = dataset_registry[paired_dataset_name]()
        return paired_dataset.snapshot()

    def _normalize_personal_status(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        target_column: str,
    ) -> pd.DataFrame:
        if "personal_status_sex" not in df.columns or target_column == "personal_status_sex":
            return df.copy(deep=True)

        mapping = self._PERSONAL_STATUS_MAP[dataset_name]
        series = df["personal_status_sex"]
        if series.dropna().map(lambda value: isinstance(value, str)).all():
            return df.copy(deep=True)

        unique_values = set(pd.Index(series.dropna().unique()).tolist())
        missing_values = sorted(unique_values.difference(mapping))
        if missing_values:
            raise ValueError(
                "CvasGermanReferencePreProcess found unmapped personal_status_sex "
                f"values for dataset '{dataset_name}': {missing_values}"
            )

        normalized = df.copy(deep=True)
        normalized["personal_status_sex"] = series.map(mapping)
        return normalized

    def transform(self, input_dataset):
        with seed_context(self._seed):
            from preprocess.preprocess_utils import ensure_flag_absent

            for flag in (
                "range",
                "scaling",
                "encoding",
                "encoded_feature_type",
                "encoded_feature_mutability",
                "encoded_feature_actionability",
                "cvas_reference",
            ):
                ensure_flag_absent(input_dataset, flag)

            dataset_name = self._resolve_dataset_name(input_dataset)
            paired_dataset_name = (
                self._paired_dataset_name or self._PAIR_DATASET_NAME[dataset_name]
            ).lower()
            target_column = input_dataset.target_column

            raw_df = self._normalize_personal_status(
                input_dataset.snapshot(),
                dataset_name=dataset_name,
                target_column=target_column,
            )
            paired_df = self._normalize_personal_status(
                self._load_paired_snapshot(paired_dataset_name),
                dataset_name=paired_dataset_name,
                target_column=target_column,
            )
            joint_df = pd.concat([raw_df, paired_df], axis=0, ignore_index=True)

            raw_feature_type = input_dataset.attr("raw_feature_type")
            input_dataset.update(
                "range",
                ScalePreProcess._compute_feature_ranges(
                    raw_df,
                    target_column=target_column,
                    feature_type=raw_feature_type,
                ),
            )

            scaling_map: dict[str, str] = {}
            encoding_map: dict[str, list[str]] = {}
            encoded_sources: dict[str, str] = {}
            final_parts: list[pd.DataFrame] = []

            for column in raw_df.columns:
                if column == target_column:
                    continue

                feature_type = str(raw_feature_type.get(column, "")).lower()
                if feature_type == "numerical":
                    scaling_map[column] = "standardize"
                    series = raw_df[column].astype("float64")
                    joint_series = joint_df[column].astype("float64")
                    mean_value = float(joint_series.mean())
                    std_value = float(joint_series.std(ddof=0))
                    if std_value == 0.0:
                        scaled_series = pd.Series(0.0, index=series.index, name=column)
                    else:
                        scaled_series = (series - mean_value) / std_value
                    final_parts.append(scaled_series.to_frame(name=column))
                    encoding_map[column] = [column]
                    continue

                if feature_type == "categorical":
                    categories = sorted(pd.Index(joint_df[column].dropna().unique()).tolist())
                    categorical = pd.Categorical(
                        raw_df[column],
                        categories=categories,
                        ordered=True,
                    )
                    encoded = pd.get_dummies(categorical, dtype="float64")
                    encoded.index = raw_df.index
                    encoded.columns = [
                        f"{column}_cat_{EncodePreProcess._format_category_suffix(category)}"
                        for category in categories
                    ]
                    final_parts.append(encoded)
                    encoding_map[column] = list(encoded.columns)
                    for encoded_column in encoded.columns:
                        encoded_sources[encoded_column] = column
                    continue

                passthrough = raw_df.loc[:, [column]].copy(deep=True)
                final_parts.append(passthrough)
                encoding_map[column] = [column]

            final_parts.append(raw_df.loc[:, [target_column]].copy(deep=True))
            final_df = pd.concat(final_parts, axis=1)

            (
                encoded_feature_type,
                encoded_feature_mutability,
                encoded_feature_actionability,
            ) = EncodePreProcess._build_encoded_metadata(
                dataset=input_dataset,
                final_columns=list(final_df.columns),
                encoded_sources=encoded_sources,
            )

            input_dataset.update("scaling", scaling_map)
            input_dataset.update("encoding", encoding_map, df=final_df)
            input_dataset.update("encoded_feature_type", encoded_feature_type)
            input_dataset.update("encoded_feature_mutability", encoded_feature_mutability)
            input_dataset.update("encoded_feature_actionability", encoded_feature_actionability)
            input_dataset.update("cvas_reference", True)
            return input_dataset


class StratifiedSplitPreProcess(PreProcessObject):
    def __init__(
        self,
        seed: int | None = None,
        split: float | int = 0.2,
        sample: int | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._split: float | int = SplitPreProcess._resolve_split(split)
        self._sample: int | None = SplitPreProcess._resolve_sample(sample)

    def transform(self, input_dataset):
        with seed_context(self._seed):
            from preprocess.preprocess_utils import ensure_flag_absent

            ensure_flag_absent(input_dataset, "trainset")
            ensure_flag_absent(input_dataset, "testset")

            df = input_dataset.snapshot()
            target = df.loc[:, input_dataset.target_column]
            train_df, test_df = train_test_split(
                df,
                test_size=self._split,
                random_state=self._seed,
                stratify=target,
                shuffle=True,
            )
            train_df = train_df.copy(deep=True)
            test_df = test_df.copy(deep=True)

            if self._sample is not None:
                if self._sample > test_df.shape[0]:
                    raise ValueError(
                        "StratifiedSplitPreProcess sample exceeds split testset size"
                    )
                test_df = test_df.sample(
                    n=self._sample,
                    random_state=self._seed,
                ).copy(deep=True)

            testset = input_dataset.clone()
            trainset = input_dataset
            trainset.update("trainset", True, df=train_df)
            testset.update("testset", True, df=test_df)
            return trainset, testset


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _normalize_config(config: dict) -> dict:
    normalized = deepcopy(config)
    preprocess_cfg = list(normalized.get("preprocess", []))
    if not any(item.get("name", "").lower() == "finalize" for item in preprocess_cfg):
        preprocess_cfg.append({"name": "finalize"})
    normalized["preprocess"] = preprocess_cfg
    return normalized


def _resolve_device(device: str) -> str:
    resolved = str(device).lower()
    if resolved == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if resolved not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    if resolved == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available in the current environment")
    return resolved


def _resolve_profile(profile: str, smoke: bool) -> dict:
    selected = "smoke" if smoke else profile
    if selected not in PROFILE_DEFAULTS:
        raise ValueError(f"Unsupported profile: {selected}")
    defaults = deepcopy(PROFILE_DEFAULTS[selected])
    defaults["profile_name"] = selected
    return defaults


def _prepare_config(
    config: dict,
    device: str,
    epochs: int | None,
    model_seed: int | None,
) -> dict:
    cfg = _normalize_config(config)
    cfg["model"]["device"] = device
    cfg["method"]["device"] = device
    cfg["model"]["save_name"] = None
    if epochs is not None:
        cfg["model"]["epochs"] = int(epochs)
    if model_seed is not None:
        cfg["model"]["seed"] = int(model_seed)
    return cfg


def _disable_mlp_progress() -> None:
    def _quiet_tqdm(iterable, **kwargs):
        return iterable

    mlp_module.tqdm = _quiet_tqdm


def _instantiate_dataset(dataset_cfg: dict):
    cfg = deepcopy(dataset_cfg)
    name = cfg.pop("name")
    dataset_class = get_registry("Dataset")[name]
    return dataset_class(**cfg)


def _instantiate_preprocess(preprocess_cfg: list[dict]) -> list:
    registry = get_registry("PreProcess")
    instances = []
    for cfg in preprocess_cfg:
        item_cfg = deepcopy(cfg)
        name = item_cfg.pop("name")
        normalized_name = str(name).lower()
        if normalized_name == "cvas_german_reference":
            instances.append(CvasGermanReferencePreProcess(**item_cfg))
        elif normalized_name == "stratified_split":
            instances.append(StratifiedSplitPreProcess(**item_cfg))
        else:
            instances.append(registry[name](**item_cfg))
    return instances


def _materialize_full_dataset(config: dict):
    cfg = _normalize_config(config)
    cfg["preprocess"] = [
        deepcopy(step)
        for step in cfg.get("preprocess", [])
        if step.get("name", "").lower() not in {"split", "stratified_split"}
    ]

    dataset_obj = _instantiate_dataset(cfg["dataset"])
    datasets = [dataset_obj]
    for preprocess_step in _instantiate_preprocess(cfg.get("preprocess", [])):
        next_datasets = []
        for current_dataset in datasets:
            transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                next_datasets.append(transformed)
        datasets = next_datasets

    if len(datasets) != 1:
        raise ValueError("Expected exactly one frozen dataset after removing splits")
    return datasets[0]


def _snapshot_frozen_dataset(dataset_obj) -> pd.DataFrame:
    return pd.concat([dataset_obj.get(target=False), dataset_obj.get(target=True)], axis=1)


def _build_frozen_dataset(template_dataset, df: pd.DataFrame, marker: str):
    ordered_columns = template_dataset.ordered_features()
    dataset_obj = template_dataset.clone()
    dataset_obj.update(marker, True, df=df.loc[:, ordered_columns].copy(deep=True))
    dataset_obj.freeze()
    return dataset_obj


def _extract_split_settings(config: dict) -> dict[str, object]:
    for preprocess_cfg in _normalize_config(config).get("preprocess", []):
        name = str(preprocess_cfg.get("name", "")).lower()
        if name in {"split", "stratified_split"}:
            return {
                "name": name,
                "split": preprocess_cfg.get("split", 0.2),
                "seed": preprocess_cfg.get("seed"),
            }
    raise ValueError("Expected a split or stratified_split preprocess in the config")


def _split_dataframe(
    df: pd.DataFrame,
    target_column: str,
    split_settings: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = split_settings["split"]
    seed = split_settings["seed"]
    split_name = str(split_settings["name"]).lower()
    target = df.loc[:, target_column]
    split_kwargs = {
        "random_state": seed,
        "shuffle": True,
    }
    if split_name == "stratified_split":
        split_kwargs["stratify"] = target

    if isinstance(split, float):
        train_df, test_df = train_test_split(
            df,
            train_size=1.0 - float(split),
            **split_kwargs,
        )
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=int(split),
            **split_kwargs,
        )

    return train_df.copy(deep=True), test_df.copy(deep=True)


def _resolve_fold_training(
    train_df: pd.DataFrame,
    fold_index: int,
    num_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fold_index < 0 or fold_index >= num_splits:
        raise ValueError(f"fold_index must be in [0, {num_splits - 1}]")

    splits = list(KFold(n_splits=num_splits).split(train_df))
    train_index, holdout_index = splits[fold_index]
    return (
        train_df.iloc[train_index].copy(deep=True),
        train_df.iloc[holdout_index].copy(deep=True),
    )


def _instantiate_model(model_cfg: dict):
    cfg = deepcopy(model_cfg)
    name = cfg.pop("name")
    model_class = get_registry("TargetModel")[name]
    return model_class(**cfg)


def _instantiate_method(method_cfg: dict, target_model):
    cfg = deepcopy(method_cfg)
    name = cfg.pop("name")
    method_class = get_registry("Method")[name]
    return method_class(target_model=target_model, **cfg)


def _predict_label_indices(model_obj, features: pd.DataFrame) -> np.ndarray:
    if features.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    probabilities = model_obj.get_prediction(features, proba=True).detach().cpu().numpy()
    return probabilities.argmax(axis=1).astype(np.int64, copy=False)


def _ensure_matching_feature_space(current_dataset, future_dataset) -> None:
    if list(current_dataset.get(target=False).columns) != list(
        future_dataset.get(target=False).columns
    ):
        raise ValueError(
            "Current and shifted datasets must share the same preprocessed feature columns"
        )


def _prepare_shared_state(current_cfg: dict, future_cfg: dict) -> dict[str, object]:
    current_dataset = _materialize_full_dataset(current_cfg)
    future_dataset = _materialize_full_dataset(future_cfg)
    _ensure_matching_feature_space(current_dataset, future_dataset)

    split_settings = _extract_split_settings(current_cfg)
    current_df = _snapshot_frozen_dataset(current_dataset)
    train_df, test_df = _split_dataframe(
        current_df,
        target_column=current_dataset.target_column,
        split_settings=split_settings,
    )

    future_df = _snapshot_frozen_dataset(future_dataset)
    return {
        "current_dataset": current_dataset,
        "future_dataset": future_dataset,
        "train_df": train_df,
        "test_df": test_df,
        "future_df": future_df,
    }


def _prepare_future_models(
    future_cfg: dict,
    future_dataset,
    future_df: pd.DataFrame,
    num_future_models: int,
) -> list:
    if num_future_models < 1:
        raise ValueError("num_future_models must be >= 1")

    target = future_df.loc[:, future_dataset.target_column]
    future_models = []
    for future_index in range(num_future_models):
        train_df, _test_df = train_test_split(
            future_df,
            train_size=0.8,
            random_state=future_index,
            stratify=target,
            shuffle=True,
        )
        trainset = _build_frozen_dataset(future_dataset, train_df, "trainset")
        future_model = _instantiate_model(future_cfg["model"])
        future_model.fit(trainset)
        future_models.append(future_model)
    return future_models


def _build_factual_dataset(
    template_dataset,
    present_test_df: pd.DataFrame,
    fold_train_df: pd.DataFrame,
    current_model,
    desired_class: int | str | None,
    max_ins: int | None,
):
    if desired_class is None:
        raise ValueError(
            "The reproduction expects method.desired_class to be the favorable class"
        )

    factual_pool_df = pd.concat([present_test_df, fold_train_df], axis=0)
    desired_index = current_model.get_class_to_index()[desired_class]
    predicted_indices = _predict_label_indices(
        current_model,
        factual_pool_df.drop(columns=[template_dataset.target_column]),
    )
    selected_df = factual_pool_df.iloc[predicted_indices != desired_index].copy(deep=True)
    if max_ins is not None:
        selected_df = selected_df.iloc[: int(max_ins)].copy(deep=True)
    return _build_frozen_dataset(template_dataset, selected_df, "testset")


def _compute_instance_metrics(
    factuals,
    raw_counterfactuals: pd.DataFrame,
    current_model,
    future_models: list,
    desired_class: int | str,
) -> dict[str, list[float]]:
    factual_features = factuals.get(target=False)
    if raw_counterfactuals.shape[0] != factual_features.shape[0]:
        raise ValueError("Raw counterfactual rows must match factual rows")

    desired_index = current_model.get_class_to_index()[desired_class]
    feasible_mask = ~raw_counterfactuals.isna().any(axis=1)
    costs = np.zeros(factual_features.shape[0], dtype=np.float64)
    current_validity = np.zeros(factual_features.shape[0], dtype=np.float64)
    future_validity = np.zeros(factual_features.shape[0], dtype=np.float64)
    feasibility = feasible_mask.to_numpy(dtype=np.float64, copy=False)

    if bool(feasible_mask.any()):
        feasible_factuals = factual_features.loc[feasible_mask].copy(deep=True)
        feasible_counterfactuals = raw_counterfactuals.loc[feasible_mask].copy(deep=True)

        diff = np.abs(
            feasible_counterfactuals.to_numpy(dtype=np.float64)
            - feasible_factuals.to_numpy(dtype=np.float64)
        )
        costs[feasible_mask.to_numpy()] = diff.sum(axis=1)

        current_prediction = _predict_label_indices(current_model, feasible_counterfactuals)
        current_validity[feasible_mask.to_numpy()] = (
            current_prediction == desired_index
        ).astype(np.float64, copy=False)

        if future_models:
            future_hits = np.zeros(feasible_counterfactuals.shape[0], dtype=np.float64)
            for future_model in future_models:
                future_prediction = _predict_label_indices(
                    future_model,
                    feasible_counterfactuals,
                )
                future_hits += (
                    future_prediction == desired_index
                ).astype(np.float64, copy=False)
            future_validity[feasible_mask.to_numpy()] = future_hits / float(len(future_models))

    return {
        "cost": costs.tolist(),
        "current_validity": current_validity.tolist(),
        "future_validity": future_validity.tolist(),
        "feasible": feasibility.tolist(),
        "num_factuals": int(factual_features.shape[0]),
    }


def _to_padded_array(values: list[list[float]]) -> np.ndarray:
    if not values:
        return np.zeros((0, 0), dtype=np.float64)
    width = max(len(row) for row in values)
    return np.array(
        [row + [0.0] * (width - len(row)) for row in values],
        dtype=np.float64,
    )


def _summarize_fold_metrics(fold_metrics: list[dict[str, list[float]]]) -> dict[str, float]:
    if not fold_metrics:
        raise ValueError("fold_metrics must not be empty")

    cost = _to_padded_array([fold["cost"] for fold in fold_metrics])
    current = _to_padded_array([fold["current_validity"] for fold in fold_metrics])
    future = _to_padded_array([fold["future_validity"] for fold in fold_metrics])
    feasible = _to_padded_array([fold["feasible"] for fold in fold_metrics])

    joint_feasible = np.ones_like(feasible, dtype=np.float64)
    valid_folds = np.sum(joint_feasible, axis=1) > 0

    fold_cost = np.sum(cost * joint_feasible, axis=1) / np.sum(joint_feasible, axis=1)
    fold_current = np.sum(current * joint_feasible, axis=1) / np.sum(
        joint_feasible, axis=1
    )
    fold_future = np.sum(future * joint_feasible, axis=1) / np.sum(
        joint_feasible, axis=1
    )

    return {
        "cost_mean": float(np.mean(fold_cost[valid_folds])),
        "cost_std": float(np.std(fold_cost[valid_folds])),
        "current_validity_mean": float(np.mean(fold_current[valid_folds])),
        "current_validity_std": float(np.std(fold_current[valid_folds])),
        "future_validity_mean": float(np.mean(fold_future[valid_folds])),
        "future_validity_std": float(np.std(fold_future[valid_folds])),
        "feasibility_mean": float(np.mean(feasible)),
        "num_factuals_per_fold": [int(fold["num_factuals"]) for fold in fold_metrics],
    }


def _run_single_fold(
    current_cfg: dict,
    shared_state: dict[str, object],
    future_models: list,
    fold_index: int,
    surrogate_method: str,
    rho_neg: float,
    max_ins: int | None,
) -> dict[str, list[float]]:
    fold_train_df, _fold_holdout_df = _resolve_fold_training(
        shared_state["train_df"],
        fold_index=fold_index,
    )

    trainset = _build_frozen_dataset(
        shared_state["current_dataset"],
        fold_train_df,
        "trainset",
    )

    current_model = _instantiate_model(current_cfg["model"])
    current_model.fit(trainset)

    factuals = _build_factual_dataset(
        template_dataset=shared_state["current_dataset"],
        present_test_df=shared_state["test_df"],
        fold_train_df=fold_train_df,
        current_model=current_model,
        desired_class=current_cfg["method"].get("desired_class"),
        max_ins=max_ins,
    )

    method_cfg = deepcopy(current_cfg["method"])
    method_cfg["surrogate_method"] = surrogate_method
    method_cfg["rho_neg"] = 0.0 if surrogate_method == "mpm" else float(rho_neg)
    method_obj = _instantiate_method(method_cfg, current_model)
    method_obj.fit(trainset)
    raw_counterfactuals = method_obj.get_unvalidated_counterfactuals(
        factuals.get(target=False)
    )

    return _compute_instance_metrics(
        factuals=factuals,
        raw_counterfactuals=raw_counterfactuals,
        current_model=current_model,
        future_models=future_models,
        desired_class=current_cfg["method"]["desired_class"],
    )


def _print_aggregate_summary(
    device: str,
    surrogate_method: str,
    rho_neg: float,
    num_future_models: int,
    max_ins: int | None,
    summary: dict[str, float],
) -> None:
    print("mode: aggregate_table1")
    print(f"device: {device}")
    print(f"method: {surrogate_method}")
    print(f"rho_neg: {rho_neg:.1f}")
    print(f"num_future_models: {num_future_models}")
    print("max_ins: " + ("all" if max_ins is None else str(int(max_ins))))
    print()
    print(
        "method          cost_mean  cost_std  current_mean  current_std  future_mean  future_std  feasible_mean"
    )
    print(
        f"{surrogate_method:<15} "
        f"{summary['cost_mean']:.6f}  "
        f"{summary['cost_std']:.6f}  "
        f"{summary['current_validity_mean']:.6f}  "
        f"{summary['current_validity_std']:.6f}  "
        f"{summary['future_validity_mean']:.6f}  "
        f"{summary['future_validity_std']:.6f}  "
        f"{summary['feasibility_mean']:.6f}"
    )


def _is_dominated(row: pd.Series, candidates: pd.DataFrame) -> bool:
    for _, other in candidates.iterrows():
        if (
            float(other["cost_mean"]) <= float(row["cost_mean"])
            and float(other["future_mean"]) >= float(row["future_mean"])
            and (
                float(other["cost_mean"]) < float(row["cost_mean"])
                or float(other["future_mean"]) > float(row["future_mean"])
            )
        ):
            return True
    return False


def _print_sweep_summary(
    device: str,
    surrogate_method: str,
    num_future_models: int,
    max_ins: int | None,
    rows: list[dict[str, float]],
) -> None:
    results = pd.DataFrame(rows).sort_values("rho_neg", ignore_index=True)

    print("mode: frontier_sweep")
    print(f"device: {device}")
    print(f"method: {surrogate_method}")
    print(f"num_future_models: {num_future_models}")
    print("max_ins: " + ("all" if max_ins is None else str(int(max_ins))))
    print()
    print(results.to_string(index=False))

    pareto_rows = results.loc[
        ~results.apply(lambda row: _is_dominated(row, results), axis=1)
    ].reset_index(drop=True)
    print()
    print("pareto_frontier:")
    print(
        pareto_rows[
            [
                "rho_neg",
                "cost_mean",
                "current_mean",
                "future_mean",
                "feasible_mean",
            ]
        ].to_string(index=False)
    )


def _print_single_fold_debug(
    device: str,
    surrogate_method: str,
    rho_neg: float,
    fold_index: int,
    num_future_models: int,
    max_ins: int | None,
    fold_metrics: dict[str, list[float]],
) -> None:
    cost = np.asarray(fold_metrics["cost"], dtype=np.float64)
    current = np.asarray(fold_metrics["current_validity"], dtype=np.float64)
    future = np.asarray(fold_metrics["future_validity"], dtype=np.float64)
    feasible = np.asarray(fold_metrics["feasible"], dtype=np.float64)

    print("mode: single_fold_debug")
    print(f"device: {device}")
    print(f"method: {surrogate_method}")
    print(f"rho_neg: {rho_neg:.1f}")
    print(f"fold_index: {fold_index}")
    print(f"num_future_models: {num_future_models}")
    print("max_ins: " + ("all" if max_ins is None else str(int(max_ins))))
    print()
    print(
        "rho_neg  num_factuals  cost_mean  current_mean  future_mean  feasible_mean"
    )
    print(
        f"{rho_neg:7.1f}  "
        f"{fold_metrics['num_factuals']:12d}  "
        f"{float(np.mean(cost)):.6f}  "
        f"{float(np.mean(current)):.6f}  "
        f"{float(np.mean(future)):.6f}  "
        f"{float(np.mean(feasible)):.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-config",
        default="./experiment/cvas_proj/german_mlp_cvas_proj_reproduce_current.yaml",
    )
    parser.add_argument(
        "--future-config",
        default="./experiment/cvas_proj/german_mlp_cvas_proj_reproduce_future.yaml",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--profile",
        choices=list(PROFILE_DEFAULTS),
        default="german_reference",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["aggregate_table1", "frontier_sweep", "single_fold_debug"],
        default="aggregate_table1",
    )
    parser.add_argument("--surrogate-method", default="fr_rmpm")
    parser.add_argument("--rho-neg", type=float, default=None)
    parser.add_argument("--rho-neg-values", nargs="*", type=float, default=None)
    parser.add_argument("--num-future-models", type=int, default=None)
    parser.add_argument("--max-ins", type=int, default=None)
    parser.add_argument("--max-factuals", type=int, default=None)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=123)
    args = parser.parse_args()

    _disable_mlp_progress()

    profile_defaults = _resolve_profile(args.profile, args.smoke)
    device = _resolve_device(args.device)

    current_cfg = _prepare_config(
        _load_config((PROJECT_ROOT / args.current_config).resolve()),
        device=device,
        epochs=profile_defaults["current_epochs"],
        model_seed=args.model_seed,
    )
    future_cfg = _prepare_config(
        _load_config((PROJECT_ROOT / args.future_config).resolve()),
        device=device,
        epochs=profile_defaults["future_epochs"],
        model_seed=args.model_seed,
    )
    if profile_defaults["num_samples"] is not None:
        current_cfg["method"]["num_samples"] = int(profile_defaults["num_samples"])

    set_cache_dir(current_cfg.get("caching", {}).get("path", "./cache/"))

    max_ins = args.max_ins
    if max_ins is None and args.max_factuals is not None:
        max_ins = int(args.max_factuals)
    if max_ins is None:
        max_ins = profile_defaults["max_ins"]
    if max_ins is not None and max_ins < 1:
        raise ValueError("max_ins must be >= 1 when provided")

    surrogate_method = str(args.surrogate_method).lower()
    shared_state = _prepare_shared_state(current_cfg, future_cfg)
    num_future_models = (
        int(args.num_future_models)
        if args.num_future_models is not None
        else int(profile_defaults["num_future_models"])
    )
    future_models = _prepare_future_models(
        future_cfg=future_cfg,
        future_dataset=shared_state["future_dataset"],
        future_df=shared_state["future_df"],
        num_future_models=num_future_models,
    )

    if args.mode == "aggregate_table1":
        if args.rho_neg_values:
            raise ValueError("--rho-neg-values is only supported in frontier_sweep mode")
        rho_neg = (
            float(args.rho_neg)
            if args.rho_neg is not None
            else float(current_cfg["method"]["rho_neg"])
        )
        fold_metrics = [
            _run_single_fold(
                current_cfg=current_cfg,
                shared_state=shared_state,
                future_models=future_models,
                fold_index=fold_index,
                surrogate_method=surrogate_method,
                rho_neg=rho_neg,
                max_ins=max_ins,
            )
            for fold_index in range(5)
        ]
        summary = _summarize_fold_metrics(fold_metrics)
        _print_aggregate_summary(
            device=device,
            surrogate_method=surrogate_method,
            rho_neg=rho_neg,
            num_future_models=num_future_models,
            max_ins=max_ins,
            summary=summary,
        )
        return

    if args.mode == "single_fold_debug":
        if args.rho_neg_values:
            raise ValueError("--rho-neg-values is only supported in frontier_sweep mode")
        rho_neg = (
            float(args.rho_neg)
            if args.rho_neg is not None
            else float(current_cfg["method"]["rho_neg"])
        )
        fold_metrics = _run_single_fold(
            current_cfg=current_cfg,
            shared_state=shared_state,
            future_models=future_models,
            fold_index=int(args.fold_index),
            surrogate_method=surrogate_method,
            rho_neg=rho_neg,
            max_ins=max_ins,
        )
        _print_single_fold_debug(
            device=device,
            surrogate_method=surrogate_method,
            rho_neg=rho_neg,
            fold_index=int(args.fold_index),
            num_future_models=num_future_models,
            max_ins=max_ins,
            fold_metrics=fold_metrics,
        )
        return

    rho_neg_values = (
        [float(value) for value in args.rho_neg_values]
        if args.rho_neg_values
        else list(profile_defaults["rho_neg_values"])
    )
    rows = []
    for rho_neg in rho_neg_values:
        fold_metrics = [
            _run_single_fold(
                current_cfg=current_cfg,
                shared_state=shared_state,
                future_models=future_models,
                fold_index=fold_index,
                surrogate_method=surrogate_method,
                rho_neg=float(rho_neg),
                max_ins=max_ins,
            )
            for fold_index in range(5)
        ]
        summary = _summarize_fold_metrics(fold_metrics)
        rows.append(
            {
                "rho_neg": float(rho_neg),
                "cost_mean": summary["cost_mean"],
                "current_mean": summary["current_validity_mean"],
                "future_mean": summary["future_validity_mean"],
                "feasible_mean": summary["feasibility_mean"],
            }
        )

    _print_sweep_summary(
        device=device,
        surrogate_method=surrogate_method,
        num_future_models=num_future_models,
        max_ins=max_ins,
        rows=rows,
    )


if __name__ == "__main__":
    main()
