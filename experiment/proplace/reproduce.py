from __future__ import annotations

import argparse
import sys
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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

# Trigger standard registration.
import dataset  # noqa: F401
import method  # noqa: F401
import model  # noqa: F401
import preprocess  # noqa: F401
from dataset.compas_carla.compas_carla import CompasCarlaDataset
from preprocess.common import EncodePreProcess
from method.proplace.support import (
    OptSolver,
    build_inn,
    build_proplace_dataset,
    extract_scalar_network,
)
from utils.caching import set_cache_dir
from utils.logger import setup_logger
from utils.registry import get_registry
from utils.seed import seed_context

REFERENCE_PROPLACE_ROW = {
    "validity": 1.0,
    "delta_validity": 1.0,
    "m2_validity": 1.0,
    "l1": 0.039,
    "pct_inlier_class_1": 0.86,
    "lof_class_1": 1.241,
}


@dataclass
class ReferenceCompasData:
    full_dataset: object
    x1_dataset: object
    x1_trainset: object
    x1_testset: object
    x2_trainset: object
    X1: pd.DataFrame
    y1: pd.DataFrame
    X1_train: pd.DataFrame
    y1_train: pd.DataFrame
    X1_test: pd.DataFrame
    y1_test: pd.DataFrame
    X2_train: pd.DataFrame
    y2_train: pd.DataFrame


class ProplaceCompasCarlaDataset(CompasCarlaDataset):
    """Reference-faithful COMPAS loader used only by the ProPlace reproduction."""

    def _read_df(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(Path(path) / "compas_carla.csv")
        return df.dropna().reset_index(drop=True)


class ProplaceEncodePreProcess(EncodePreProcess):
    @staticmethod
    def _resolve_encoding(encoding: str | None) -> str | None:
        if encoding is None:
            return None
        if not isinstance(encoding, str):
            raise TypeError("encoding must be str or None")

        encoding = encoding.lower()
        if encoding not in {"none", "onehot", "onehot_drop_binary", "thermometer"}:
            raise ValueError(f"Unsupported encoding option: {encoding}")
        return encoding

    @staticmethod
    def _resolve_binary_positive_category(
        binary_positive_category: dict[str, object] | None,
    ) -> dict[str, object] | None:
        if binary_positive_category is None:
            return None
        if not isinstance(binary_positive_category, dict):
            raise TypeError(
                "ProplaceEncodePreProcess binary_positive_category must be a dict[str, object] or None"
            )

        resolved_mapping: dict[str, object] = {}
        for feature_name, category in binary_positive_category.items():
            if not isinstance(feature_name, str):
                raise TypeError(
                    "ProplaceEncodePreProcess binary_positive_category keys must be str feature names only"
                )
            resolved_mapping[feature_name] = category
        return resolved_mapping

    @staticmethod
    def _build_onehot_drop_binary(
        series: pd.Series,
        feature_name: str,
        positive_category: object | None = None,
    ) -> pd.DataFrame:
        categories = sorted(pd.Index(series.dropna().unique()).tolist())
        if len(categories) != 2:
            if positive_category is not None:
                raise ValueError(
                    "ProplaceEncodePreProcess binary_positive_category requires exactly two categories "
                    f"for {feature_name}"
                )
            return EncodePreProcess._build_onehot(series, feature_name)

        resolved_positive = categories[-1]
        if positive_category is not None:
            matched_categories = [
                category
                for category in categories
                if category == positive_category or str(category) == str(positive_category)
            ]
            if not matched_categories:
                raise ValueError(
                    "ProplaceEncodePreProcess binary_positive_category contains invalid "
                    f"category for {feature_name}: {positive_category}"
                )
            resolved_positive = matched_categories[0]

        column_name = (
            f"{feature_name}_cat_"
            f"{EncodePreProcess._format_category_suffix(resolved_positive)}"
        )
        encoded = (series == resolved_positive).astype("float64")
        return pd.DataFrame({column_name: encoded}, index=series.index)

    def __init__(
        self,
        seed: int | None = None,
        encoding: str | None = "onehot",
        override: dict[str, str] | None = None,
        binary_positive_category: dict[str, object] | None = None,
        **kwargs,
    ):
        super().__init__(seed=seed, encoding=encoding, override=override, **kwargs)
        self._binary_positive_category = self._resolve_binary_positive_category(
            binary_positive_category
        )

    def transform(self, input):
        with seed_context(self._seed):
            from preprocess.preprocess_utils import ensure_flag_absent

            ensure_flag_absent(input, "encoding")

            if self._encoding is None:
                return input

            df = input.snapshot()
            target_column = input.target_column
            raw_feature_type = input.attr("raw_feature_type")
            selected_modes = self._resolve_encode_modes(
                df=df,
                target_column=target_column,
                raw_feature_type=raw_feature_type,
                default_mode=self._encoding,
                override=self._override,
            )

            binary_positive_category = self._binary_positive_category or {}
            for feature_name in binary_positive_category:
                if feature_name == target_column:
                    raise ValueError(
                        "ProplaceEncodePreProcess binary_positive_category must not contain the target feature"
                    )
                if feature_name not in df.columns:
                    raise ValueError(
                        "ProplaceEncodePreProcess binary_positive_category contains unknown feature: "
                        f"{feature_name}"
                    )
                if raw_feature_type.get(feature_name, "").lower() != "categorical":
                    raise ValueError(
                        "ProplaceEncodePreProcess binary_positive_category contains non-categorical feature: "
                        f"{feature_name}"
                    )

            final_parts: list[pd.DataFrame] = []
            encoded_sources: dict[str, str] = {}
            encoding_map: dict[str, list[str]] = {}
            seen_columns: set[str] = set()
            remaining_columns = set(df.columns)
            for column in df.columns:
                remaining_columns.discard(column)

                if column == target_column:
                    passthrough = df.loc[:, [column]].copy(deep=True)
                    self._ensure_output_columns_available(
                        [column], seen_columns, remaining_columns
                    )
                    final_parts.append(passthrough)
                    seen_columns.add(column)
                elif column not in selected_modes:
                    passthrough = df.loc[:, [column]].copy(deep=True)
                    self._ensure_output_columns_available(
                        [column], seen_columns, remaining_columns
                    )
                    final_parts.append(passthrough)
                    seen_columns.add(column)
                else:
                    mode = selected_modes[column]
                    if mode == "onehot":
                        encoded = self._build_onehot(df[column], column)
                        new_columns = list(encoded.columns)
                        self._ensure_output_columns_available(
                            new_columns, seen_columns, remaining_columns
                        )
                        final_parts.append(encoded)
                        encoding_map[column] = new_columns
                        seen_columns.update(new_columns)
                        for encoded_column in new_columns:
                            encoded_sources[encoded_column] = column
                    elif mode == "onehot_drop_binary":
                        encoded = self._build_onehot_drop_binary(
                            df[column],
                            column,
                            positive_category=binary_positive_category.get(column),
                        )
                        new_columns = list(encoded.columns)
                        self._ensure_output_columns_available(
                            new_columns, seen_columns, remaining_columns
                        )
                        final_parts.append(encoded)
                        encoding_map[column] = new_columns
                        seen_columns.update(new_columns)
                        for encoded_column in new_columns:
                            encoded_sources[encoded_column] = column
                    elif mode == "thermometer":
                        encoded = self._build_thermometer(df[column], column)
                        new_columns = list(encoded.columns)
                        self._ensure_output_columns_available(
                            new_columns, seen_columns, remaining_columns
                        )
                        final_parts.append(encoded)
                        encoding_map[column] = new_columns
                        seen_columns.update(new_columns)
                        for encoded_column in new_columns:
                            encoded_sources[encoded_column] = column
                    elif mode == "none":
                        passthrough = df.loc[:, [column]].copy(deep=True)
                        self._ensure_output_columns_available(
                            [column], seen_columns, remaining_columns
                        )
                        final_parts.append(passthrough)
                        encoding_map[column] = [column]
                        seen_columns.add(column)
                    else:
                        raise ValueError(
                            f"Unsupported encoding option for feature {column}: {mode}"
                        )

            final_df = pd.concat(final_parts, axis=1)
            (
                encoded_feature_type,
                encoded_feature_mutability,
                encoded_feature_actionability,
            ) = self._build_encoded_metadata(
                dataset=input,
                final_columns=list(final_df.columns),
                encoded_sources=encoded_sources,
            )

            input.update("encoding", encoding_map, df=final_df)
            input.update("encoded_feature_type", encoded_feature_type)
            input.update("encoded_feature_mutability", encoded_feature_mutability)
            input.update("encoded_feature_actionability", encoded_feature_actionability)
            return input


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _apply_device(config: dict, device: str) -> dict:
    cfg = deepcopy(config)
    cfg["model"]["device"] = device
    cfg["method"]["device"] = device
    return cfg


def _normalize_config(config: dict) -> dict:
    cfg = deepcopy(config)
    preprocess_cfg = list(cfg.get("preprocess", []))
    if not any(item.get("name", "").lower() == "finalize" for item in preprocess_cfg):
        preprocess_cfg.append({"name": "finalize"})
    cfg["preprocess"] = preprocess_cfg
    return cfg


def _build_dataset(config: dict):
    dataset_cfg = deepcopy(config["dataset"])
    dataset_name = dataset_cfg.pop("name")
    if dataset_name == "compas_carla":
        return ProplaceCompasCarlaDataset(**dataset_cfg)
    dataset_class = get_registry("dataset")[dataset_name]
    return dataset_class(**dataset_cfg)


def _build_preprocess(config: dict) -> list:
    preprocess_objects = []
    preprocess_registry = get_registry("preprocess")
    for preprocess_cfg in config.get("preprocess", []):
        item_cfg = deepcopy(preprocess_cfg)
        preprocess_name = item_cfg.pop("name")
        if preprocess_name == "encode" and str(item_cfg.get("encoding", "")).lower() == "onehot_drop_binary":
            preprocess_objects.append(ProplaceEncodePreProcess(**item_cfg))
            continue
        preprocess_objects.append(preprocess_registry[preprocess_name](**item_cfg))
    return preprocess_objects


def _build_model(config: dict):
    model_cfg = deepcopy(config["model"])
    model_name = model_cfg.pop("name")
    model_class = get_registry("model")[model_name]
    return model_class(**model_cfg)


def _build_method(config: dict, target_model):
    method_cfg = deepcopy(config["method"])
    method_name = method_cfg.pop("name")
    method_class = get_registry("method")[method_name]
    return method_class(target_model=target_model, **method_cfg)


def _materialize_processed_dataset(raw_dataset, preprocess_steps):
    datasets = [raw_dataset]
    for preprocess_step in preprocess_steps:
        next_datasets = []
        for current_dataset in datasets:
            transformed = preprocess_step.transform(current_dataset)
            if isinstance(transformed, tuple):
                next_datasets.extend(list(transformed))
            else:
                next_datasets.append(transformed)
        datasets = next_datasets

    if len(datasets) != 1:
        raise ValueError(
            "ProPlace Compas reproduction expects a single processed dataset before custom splitting"
        )
    return datasets[0]


def _build_dataset_clone(
    base_dataset,
    features: pd.DataFrame,
    target: pd.DataFrame,
    flag: str,
):
    dataset = base_dataset.clone()
    target_column = base_dataset.target_column
    combined = pd.concat(
        [features.copy(deep=True), target.copy(deep=True)],
        axis=1,
    )
    combined = combined.loc[:, [*features.columns, target_column]]
    dataset.update(flag, True, df=combined)
    dataset.freeze()
    return dataset


def _split_xy(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        df.drop(columns=[target_column]).copy(deep=True),
        df.loc[:, [target_column]].copy(deep=True),
    )


def _build_reference_splits(
    processed_dataset,
    config: dict,
    row_limit: int | None = None,
) -> ReferenceCompasData:
    reproduction_cfg = deepcopy(config["reproduction"])
    split_cfg = deepcopy(reproduction_cfg["split"])
    target_column = processed_dataset.target_column

    full_df = pd.concat(
        [processed_dataset.get(target=False), processed_dataset.get(target=True)],
        axis=1,
    )
    if row_limit is not None:
        if row_limit < 2:
            raise ValueError("row_limit must be >= 2 when provided")
        full_df = full_df.iloc[:row_limit].copy(deep=True)
    size = full_df.shape[0]
    half_size = int(size / 2) - 1
    half_rng = np.random.RandomState(int(split_cfg["half_selection_seed"]))
    idx_1 = np.sort(half_rng.choice(size, half_size, replace=False))
    idx_2 = np.setdiff1d(np.arange(size), idx_1, assume_unique=False)

    df1 = full_df.iloc[idx_1].copy(deep=True)
    df2 = full_df.iloc[idx_2].copy(deep=True)
    X1, y1 = _split_xy(df1, target_column)
    X2, y2 = _split_xy(df2, target_column)

    stratify_1 = y1.iloc[:, 0]
    stratify_2 = y2.iloc[:, 0]
    test_size = float(split_cfg["train_test_test_size"])
    split_seed = int(split_cfg["train_test_split_seed"])

    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1,
        y1,
        stratify=stratify_1,
        test_size=test_size,
        shuffle=True,
        random_state=split_seed,
    )
    X2_train, _, y2_train, _ = train_test_split(
        X2,
        y2,
        stratify=stratify_2,
        test_size=test_size,
        shuffle=True,
        random_state=split_seed,
    )

    x1_dataset = _build_dataset_clone(processed_dataset, X1, y1, "reference_x1")
    x1_trainset = _build_dataset_clone(processed_dataset, X1_train, y1_train, "trainset")
    x1_testset = _build_dataset_clone(processed_dataset, X1_test, y1_test, "testset")
    x2_trainset = _build_dataset_clone(processed_dataset, X2_train, y2_train, "reference_x2_train")

    return ReferenceCompasData(
        full_dataset=processed_dataset,
        x1_dataset=x1_dataset,
        x1_trainset=x1_trainset,
        x1_testset=x1_testset,
        x2_trainset=x2_trainset,
        X1=X1,
        y1=y1,
        X1_train=X1_train,
        y1_train=y1_train,
        X1_test=X1_test,
        y1_test=y1_test,
        X2_train=X2_train,
        y2_train=y2_train,
    )


def _predict_binary_labels(model, features: pd.DataFrame) -> np.ndarray:
    prediction = model.get_prediction(features, proba=False)
    return prediction.detach().cpu().numpy().argmax(axis=1)


def _predict_positive_probability(model, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.get_prediction(features, proba=True).detach().cpu().numpy()
    return probabilities[:, 1]


def _compute_model_metrics(model, testset) -> dict[str, float]:
    features = testset.get(target=False)
    target = testset.get(target=True).iloc[:, 0].astype(int).to_numpy()
    prediction = _predict_binary_labels(model, features)
    probability = _predict_positive_probability(model, features)
    metrics = {"test_accuracy": float(accuracy_score(target, prediction))}
    if len(np.unique(target)) < 2:
        metrics["test_auc"] = float("nan")
    else:
        metrics["test_auc"] = float(roc_auc_score(target, probability))
    return metrics


def _instantiate_model(config: dict, device: str, **overrides):
    model_cfg = deepcopy(config["model"])
    model_name = model_cfg.pop("name")
    model_cfg["device"] = device
    model_cfg.update(overrides)
    model_class = get_registry("model")[model_name]
    return model_class(**model_cfg)


def _train_retrained_models(
    data: ReferenceCompasData,
    config: dict,
    device: str,
) -> list:
    reproduction_cfg = config["reproduction"]
    ensemble_cfg = reproduction_cfg["ensemble"]

    union_X = pd.concat([data.X1_train, data.X2_train], axis=0).copy(deep=True)
    union_y = pd.concat([data.y1_train, data.y2_train], axis=0).copy(deep=True)
    union_trainset = _build_dataset_clone(
        data.full_dataset,
        union_X,
        union_y,
        "reference_union_trainset",
    )

    models = []
    retrain_count = int(ensemble_cfg["retrain_count"])
    retrain_seed_start = int(ensemble_cfg["retrain_seed_start"])
    retrain_hidden_size = int(ensemble_cfg["hidden_size"])
    retrain_epochs = int(ensemble_cfg["epochs"])

    retrain_iterator = tqdm(
        range(retrain_count),
        desc="m2-retrain-union",
        leave=False,
    )
    for offset in retrain_iterator:
        seed = retrain_seed_start + offset + 1
        model = _instantiate_model(
            config,
            device=device,
            seed=seed,
            layers=[retrain_hidden_size, retrain_hidden_size],
            epochs=retrain_epochs,
            save_name=None,
        )
        model.fit(union_trainset)
        models.append(model)

    leave_fraction = float(ensemble_cfg["leave_out_fraction"])
    leave_size = int(leave_fraction * data.X1_train.shape[0])
    leave_rng = np.random.RandomState(int(ensemble_cfg["leave_out_selection_seed"]))
    drop_indices = leave_rng.choice(
        data.X1_train.index.to_numpy(),
        leave_size,
        replace=False,
    )
    X1_train_leave = data.X1_train.drop(drop_indices)
    y1_train_leave = data.y1_train.drop(drop_indices)
    leave_trainset = _build_dataset_clone(
        data.full_dataset,
        X1_train_leave,
        y1_train_leave,
        "reference_leave_out_trainset",
    )

    leave_count = int(ensemble_cfg["leave_out_count"])
    leave_seed_start = int(ensemble_cfg["leave_out_seed_start"])
    leave_iterator = tqdm(
        range(leave_count),
        desc="m2-retrain-leave-out",
        leave=False,
    )
    for offset in leave_iterator:
        seed = leave_seed_start + offset + 1
        model = _instantiate_model(
            config,
            device=device,
            seed=seed,
            layers=[retrain_hidden_size, retrain_hidden_size],
            epochs=retrain_epochs,
            save_name=None,
        )
        model.fit(leave_trainset)
        models.append(model)

    return models


def _select_reference_factuals(
    main_model,
    m2s: list,
    data: ReferenceCompasData,
    config: dict,
    override_count: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    reproduction_cfg = config["reproduction"]
    selection_cfg = reproduction_cfg["factual_selection"]
    num_factuals = int(
        override_count
        if override_count is not None
        else selection_cfg["num_factuals"]
    )
    negative_mask = _predict_binary_labels(main_model, data.X1_train) == 0
    candidate_pool = data.X1_train.loc[negative_mask].copy(deep=True)

    if candidate_pool.empty:
        raise RuntimeError("Main model produced no class-0 candidates in X1_train")

    if m2s:
        all_zero_mask = np.ones(candidate_pool.shape[0], dtype=bool)
        for model in [main_model, *m2s]:
            all_zero_mask &= _predict_binary_labels(model, candidate_pool) == 0
        candidate_pool = candidate_pool.loc[all_zero_mask].copy(deep=True)

    if candidate_pool.empty:
        raise RuntimeError("No factual candidates remain after M2 consensus filtering")

    sample_rng = np.random.RandomState(int(selection_cfg["sample_seed"]))
    sampled_positions = sample_rng.choice(
        np.arange(candidate_pool.shape[0]),
        size=num_factuals,
        replace=bool(selection_cfg.get("sample_with_replacement", True)),
    )
    factuals = candidate_pool.iloc[sampled_positions].reset_index(drop=True)
    return factuals, {
        "candidate_pool_size_before_consensus": int(negative_mask.sum()),
        "candidate_pool_size_after_consensus": int(candidate_pool.shape[0]),
        "num_sampled_factuals": int(num_factuals),
    }


def _normalised_l1(counterfactual: np.ndarray, factual: np.ndarray) -> float:
    return float(np.sum(np.abs(counterfactual - factual)) / counterfactual.shape[0])


def _evaluate_proplace(
    main_model,
    m2s: list,
    counterfactuals: pd.DataFrame,
    factuals: pd.DataFrame,
    method_fit_dataset,
    data: ReferenceCompasData,
    config: dict,
) -> tuple[dict[str, float], pd.DataFrame]:
    reproduction_cfg = config["reproduction"]
    metric_cfg = reproduction_cfg["metrics"]
    target_index = int(metric_cfg["target_class"])
    delta = float(metric_cfg["delta"])
    epsilon = float(metric_cfg["epsilon"])
    robustness_big_m = float(metric_cfg["robustness_big_m"])

    cf_predictions = _predict_binary_labels(main_model, counterfactuals)
    validity = float(np.mean(cf_predictions == target_index))

    if m2s:
        m2_scores = [
            float(np.mean(_predict_binary_labels(model, counterfactuals) == target_index))
            for model in m2s
        ]
        m2_validity = float(np.mean(m2_scores))
    else:
        m2_validity = float("nan")

    dataset_spec, _ = build_proplace_dataset(method_fit_dataset)
    scalar_network = extract_scalar_network(main_model, target_index)
    interval_network = build_inn(scalar_network, delta)

    positive_class_points = data.X1.loc[
        data.y1.iloc[:, 0].astype(int).to_numpy() == target_index
    ].to_numpy(dtype=np.float64)
    novelty_lof = LocalOutlierFactor(n_neighbors=10, novelty=True)
    novelty_lof.fit(positive_class_points)

    valid_len = counterfactuals.shape[0]
    delta_validity_sum = 0.0
    avg_bound_sum = 0.0
    avg_l1_sum = 0.0
    inlier_sum = 0.0
    lof_sum = 0.0
    per_row_records: list[dict[str, float | int | bool]] = []

    factual_array = factuals.to_numpy(dtype=np.float64)
    cf_array = counterfactuals.to_numpy(dtype=np.float64)
    evaluation_iterator = tqdm(
        enumerate(cf_array),
        total=int(counterfactuals.shape[0]),
        desc="proplace-eval",
        leave=False,
    )
    for row_index, counterfactual in evaluation_iterator:
        factual = factual_array[row_index]
        failed = bool(np.isnan(counterfactual).any())
        record: dict[str, float | int | bool] = {
            "row_index": row_index,
            "failed": failed,
        }

        if failed:
            valid_len -= 1
            record.update(
                {
                    "main_prediction": -1,
                    "delta_valid": -1,
                    "delta_bound": float("nan"),
                    "l1": float("nan"),
                    "lof_novel_inlier": float("nan"),
                    "lof_class_1": float("nan"),
                }
            )
            per_row_records.append(record)
            continue

        solver = OptSolver(
            dataset=dataset_spec,
            inn=interval_network,
            y_prime=target_index,
            x=counterfactual,
            mode=1,
            eps=epsilon,
            big_m=robustness_big_m,
            x_prime=counterfactual,
            solver_config=getattr(main_model, "_solver_config", None),
        )
        found, bound = solver.compute_inn_bounds()
        delta_validity_sum += found
        if bound is not None:
            avg_bound_sum += float(bound)

        l1 = _normalised_l1(counterfactual, factual)
        avg_l1_sum += l1

        novelty_score = 0.0
        if novelty_lof.predict(counterfactual.reshape(1, -1))[0] != -1:
            novelty_score = 1.0
        inlier_sum += novelty_score

        lof = LocalOutlierFactor(n_neighbors=10)
        lof.fit(
            np.concatenate(
                [counterfactual.reshape(1, -1), positive_class_points],
                axis=0,
            )
        )
        lof_value = float(-1.0 * lof.negative_outlier_factor_[0])
        lof_sum += lof_value

        record.update(
            {
                "main_prediction": int(cf_predictions[row_index]),
                "delta_valid": int(found),
                "delta_bound": float(bound) if bound is not None else float("nan"),
                "l1": l1,
                "lof_novel_inlier": novelty_score,
                "lof_class_1": lof_value,
            }
        )
        per_row_records.append(record)

    if valid_len <= 0:
        metrics = {
            "validity": validity,
            "delta_validity": float("nan"),
            "m2_validity": m2_validity,
            "l1": float("nan"),
            "pct_inlier_class_1": float("nan"),
            "lof_class_1": float("nan"),
            "avg_bound": float("nan"),
            "valid_counterfactual_count": 0,
        }
    else:
        metrics = {
            "validity": validity,
            "delta_validity": float(delta_validity_sum / valid_len),
            "m2_validity": m2_validity,
            "l1": float(avg_l1_sum / valid_len),
            "pct_inlier_class_1": float(inlier_sum / valid_len),
            "lof_class_1": float(lof_sum / valid_len),
            "avg_bound": float(avg_bound_sum / valid_len),
            "valid_counterfactual_count": int(valid_len),
        }

    factual_prefixed = factuals.add_prefix("factual_")
    cf_prefixed = counterfactuals.add_prefix("counterfactual_")
    detail_df = pd.concat(
        [
            factual_prefixed.reset_index(drop=True),
            cf_prefixed.reset_index(drop=True),
            pd.DataFrame(per_row_records),
        ],
        axis=1,
    )
    return metrics, detail_df


def _build_reference_comparison(metrics: dict[str, float]) -> pd.DataFrame:
    rows = []
    for key, target_value in REFERENCE_PROPLACE_ROW.items():
        reproduced = metrics[key]
        rows.append(
            {
                "metric": key,
                "reference": target_value,
                "reproduced": reproduced,
                "abs_diff": abs(reproduced - target_value)
                if pd.notna(reproduced)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yml")),
    )
    parser.add_argument("--max-factuals", type=int, default=None)
    parser.add_argument("--row-limit", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    config = _normalize_config(_load_config(config_path))
    device = _resolve_device()
    config = _apply_device(config, device)
    logger_cfg = config.get("logger", {})
    logger = setup_logger(
        level=logger_cfg.get("level", "INFO"),
        path=logger_cfg.get("path"),
        name=config.get("name", "proplace_compas_reproduce"),
    )
    logger.info("Experiment config:\n%s", yaml.safe_dump(config, sort_keys=False))
    set_cache_dir(config.get("caching", {}).get("path", "./cache/"))

    raw_dataset = _build_dataset(config)
    preprocess_steps = _build_preprocess(config)
    print("Materializing processed COMPAS dataset...", flush=True)
    processed_dataset = _materialize_processed_dataset(raw_dataset, preprocess_steps)
    data = _build_reference_splits(processed_dataset, config, row_limit=args.row_limit)

    print("Training main reference model...", flush=True)
    main_model = _build_model(config)
    main_model.fit(data.x1_trainset)
    main_model_metrics = _compute_model_metrics(main_model, data.x1_testset)

    print("Training retrained M2 ensemble...", flush=True)
    m2s = _train_retrained_models(data, config, device)
    print(f"Trained {len(m2s)} ensemble models.", flush=True)
    print("Selecting reference factual pool...", flush=True)
    factuals, factual_metadata = _select_reference_factuals(
        main_model,
        m2s,
        data,
        config,
        override_count=args.max_factuals,
    )

    print("Fitting ProplaceMethod on X1 reference pool...", flush=True)
    method = _build_method(config, main_model)
    method.fit(data.x1_dataset)
    print("Generating ProPlace counterfactuals...", flush=True)
    counterfactuals = method.get_counterfactuals(factuals)

    print("Evaluating reproduced ProPlace metrics...", flush=True)
    metrics, detail_df = _evaluate_proplace(
        main_model,
        m2s,
        counterfactuals,
        factuals,
        data.x1_dataset,
        data,
        config,
    )
    comparison_df = _build_reference_comparison(metrics)

    summary = {
        "device": device,
        "processed_feature_names": list(processed_dataset.get(target=False).columns),
        "processed_row_count": int(processed_dataset.get(target=False).shape[0]),
        "main_model_metrics": main_model_metrics,
        "factual_selection": factual_metadata,
        "metrics": metrics,
        "reference_proplace_row": REFERENCE_PROPLACE_ROW,
    }

    print("Main model metrics:")
    print(pd.DataFrame([main_model_metrics]).to_string(index=False))
    print("\nProPlace metrics:")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print("\nReference comparison:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
