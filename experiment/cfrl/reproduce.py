from __future__ import annotations

import argparse
import math
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from experiment import Experiment
from method.cfrl.cfrl_tabular import get_he_preprocessor
from model.model_object import ModelObject, process_nan
from model.model_utils import resolve_device
from preprocess.preprocess_object import PreProcessObject
from utils.caching import get_cache_dir
from utils.registry import register


@dataclass
class ExperimentConfig:
    test_size: float = 0.2
    rf_max_depth: int = 15
    rf_min_samples_split: int = 10
    rf_n_estimators: int = 50
    seed: int = 0
    autoencoder_batch_size: int = 128
    autoencoder_target_steps: int = 100_000
    autoencoder_lr: float = 1e-3
    autoencoder_latent_dim: int = 15
    autoencoder_hidden_dim: int = 128
    cfrl_coeff_sparsity: float = 0.5
    cfrl_coeff_consistency: float = 0.5
    cfrl_train_steps: int = 100_000
    cfrl_batch_size: int = 128
    sample_size: int = 1000
    immutable_features: Tuple[str, ...] = (
        "Marital Status",
        "Relationship",
        "Race",
        "Sex",
    )
    constrained_ranges: Dict[str, List[float]] = field(
        default_factory=lambda: {"Age": [0.0, 1.0]}
    )


NUMERIC_FEATURE_TYPES = {
    "Age": int,
    "Capital Gain": int,
    "Capital Loss": int,
    "Hours per week": int,
}


def _artifact_paths(artifact_name: str) -> tuple[Path, Path]:
    artifact_root = Path(get_cache_dir("cfrl"))
    model_path = artifact_root / f"{artifact_name}_rf.pkl"
    preprocessor_path = artifact_root / f"{artifact_name}_preprocessor.pkl"
    return model_path, preprocessor_path


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _feature_layout(
    dataset,
) -> tuple[list[str], list[int], list[int], dict[int, list[str]]]:
    feature_names = list(dataset.get(target=False).columns)
    raw_feature_type = dataset.attr("raw_feature_type")
    category_values = dataset.attr("cfrl_category_values")

    categorical_ids: list[int] = []
    numerical_ids: list[int] = []
    category_map: dict[int, list[str]] = {}
    for index, feature_name in enumerate(feature_names):
        if str(raw_feature_type[feature_name]).lower() == "categorical":
            categorical_ids.append(index)
            category_map[index] = [
                str(value) for value in category_values[feature_name]
            ]
        else:
            numerical_ids.append(index)
    return feature_names, categorical_ids, numerical_ids, category_map


def _predict_with_rf(
    model: RandomForestClassifier,
    preprocessor: ColumnTransformer,
    feature_names: list[str],
    X: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        frame = X.loc[:, feature_names].copy(deep=True)
    else:
        frame = pd.DataFrame(np.asarray(X), columns=feature_names)
    X_pre = preprocessor.transform(frame)
    return np.asarray(model.predict_proba(X_pre), dtype=np.float32)


@register("cfrl_adult_stratified_split")
class CfrlAdultStratifiedSplitPreProcess(PreProcessObject):
    def __init__(
        self,
        seed: int | None = None,
        test_size: float = 0.2,
        sample_size: int = 1000,
        **kwargs,
    ):
        self._seed = seed
        self._test_size = float(test_size)
        self._sample_size = int(sample_size)

    def transform(self, input: DatasetObject):
        df = input.snapshot()
        y = df[input.target_column]
        train_df, test_df = train_test_split(
            df,
            test_size=self._test_size,
            random_state=self._seed,
            stratify=y,
        )

        train_df = train_df.reset_index(drop=True).copy(deep=True)
        full_test_df = test_df.reset_index(drop=True).copy(deep=True)

        rng = np.random.default_rng(self._seed)
        sample_size = min(self._sample_size, full_test_df.shape[0])
        sample_idx = rng.choice(full_test_df.shape[0], size=sample_size, replace=False)
        sampled_test_df = (
            full_test_df.iloc[sample_idx].reset_index(drop=True).copy(deep=True)
        )

        trainset = input
        testset = input.clone()

        trainset.update("trainset", True, df=train_df)
        trainset.update("cfrl_full_train_df", train_df)
        trainset.update("cfrl_full_test_df", full_test_df)

        testset.update("testset", True, df=sampled_test_df)
        testset.update("cfrl_full_train_df", train_df)
        testset.update("cfrl_full_test_df", full_test_df)

        return trainset, testset


@register("cfrl_adult_rf")
class CfrlAdultRandomForestModel(ModelObject):
    def __init__(
        self,
        seed: int | None = None,
        device: str = "cpu",
        rf_max_depth: int = 15,
        rf_min_samples_split: int = 10,
        rf_n_estimators: int = 50,
        artifact_name: str | None = None,
        **kwargs,
    ):
        self._seed = seed
        self._device = resolve_device(device)
        self._need_grad = False
        self._is_trained = False
        self._rf_max_depth = int(rf_max_depth)
        self._rf_min_samples_split = int(rf_min_samples_split)
        self._rf_n_estimators = int(rf_n_estimators)
        self._artifact_name = artifact_name
        self._preprocessor: ColumnTransformer | None = None
        self._feature_names: list[str] = []
        self._class_to_index = {0: 0, 1: 1}

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError(
                "trainset is required for CfrlAdultRandomForestModel.fit()"
            )

        X = trainset.get(target=False)
        y = trainset.get(target=True).iloc[:, 0].astype(int).to_numpy()
        self._feature_names, categorical_ids, numerical_ids, _ = _feature_layout(
            trainset
        )

        cat_transf = OneHotEncoder(
            categories=[
                list(range(int(X.iloc[:, idx].max()) + 1)) for idx in categorical_ids
            ],
            handle_unknown="ignore",
        )
        num_transf = StandardScaler()
        self._preprocessor = ColumnTransformer(
            [
                ("cat", cat_transf, categorical_ids),
                ("num", num_transf, numerical_ids),
            ],
            sparse_threshold=0,
        )
        X_pre = self._preprocessor.fit_transform(X)

        self._model = RandomForestClassifier(
            max_depth=self._rf_max_depth,
            min_samples_split=self._rf_min_samples_split,
            n_estimators=self._rf_n_estimators,
            random_state=self._seed,
        )
        self._model.fit(X_pre, y)
        self._is_trained = True

        if self._artifact_name is not None:
            model_path, preprocessor_path = _artifact_paths(self._artifact_name)
            with model_path.open("wb") as file:
                pickle.dump(self._model, file)
            with preprocessor_path.open("wb") as file:
                pickle.dump(
                    {
                        "preprocessor": self._preprocessor,
                        "feature_names": self._feature_names,
                    },
                    file,
                )

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained or self._preprocessor is None:
            raise RuntimeError("Target model is not trained")
        probabilities = _predict_with_rf(
            self._model,
            self._preprocessor,
            self._feature_names,
            X,
        )
        output = torch.tensor(probabilities, dtype=torch.float32, device=self._device)
        if proba:
            return output
        indices = output.argmax(dim=1)
        return torch.nn.functional.one_hot(indices, num_classes=2).to(
            dtype=torch.float32
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._is_trained or self._preprocessor is None:
            raise RuntimeError("Target model is not trained")
        frame = pd.DataFrame(
            X.detach().cpu().numpy(),
            columns=self._feature_names,
        )
        probabilities = _predict_with_rf(
            self._model,
            self._preprocessor,
            self._feature_names,
            frame,
        )
        return torch.tensor(probabilities, dtype=torch.float32, device=self._device)


@register("cfrl_paper_metrics")
class CfrlPaperMetricsEvaluation(EvaluationObject):
    def __init__(
        self,
        artifact_name: str,
        seed: int,
        immutable_features: list[str],
        **kwargs,
    ):
        self._artifact_name = str(artifact_name)
        self._seed = int(seed)
        self._immutable_features = tuple(str(name) for name in immutable_features)
        self._model = None
        self._preprocessor = None
        self._feature_names: list[str] | None = None

    def _ensure_loaded(self) -> None:
        if (
            self._model is not None
            and self._preprocessor is not None
            and self._feature_names is not None
        ):
            return
        model_path, preprocessor_path = _artifact_paths(self._artifact_name)
        with model_path.open("rb") as file:
            self._model = pickle.load(file)
        with preprocessor_path.open("rb") as file:
            payload = pickle.load(file)
        self._preprocessor = payload["preprocessor"]
        self._feature_names = list(payload["feature_names"])

    def _predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        self._ensure_loaded()
        assert self._preprocessor is not None
        assert self._feature_names is not None
        assert self._model is not None
        return _predict_with_rf(
            self._model,
            self._preprocessor,
            self._feature_names,
            X,
        )

    @staticmethod
    def _build_he_preprocessor(
        X_train: np.ndarray,
        feature_names: list[str],
        category_map: dict[int, list[str]],
    ):
        return get_he_preprocessor(
            X=X_train,
            feature_names=feature_names,
            category_map=category_map,
            feature_types=NUMERIC_FEATURE_TYPES,
        )[0]

    @staticmethod
    def _rbf_mmd(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
        def _pairwise_sq(u: np.ndarray, v: np.ndarray) -> np.ndarray:
            u_norm = (u**2).sum(axis=1)[:, None]
            v_norm = (v**2).sum(axis=1)[None, :]
            return u_norm + v_norm - 2 * (u @ v.T)

        z = np.concatenate([x, y], axis=0)
        if z.shape[0] > 2000:
            idx = rng.choice(z.shape[0], size=2000, replace=False)
            z_sample = z[idx]
        else:
            z_sample = z
        dists = _pairwise_sq(z_sample, z_sample)
        median = np.median(dists[dists > 0])
        sigma = math.sqrt(median) if median > 0 else 1.0
        gamma = 1.0 / (2 * sigma**2)
        k_xx = np.exp(-gamma * _pairwise_sq(x, x))
        k_yy = np.exp(-gamma * _pairwise_sq(y, y))
        k_xy = np.exp(-gamma * _pairwise_sq(x, y))
        return float(max(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean(), 0.0))

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        self._ensure_loaded()
        assert self._feature_names is not None
        factual_df = factuals.get(target=False)
        raw_cf_array = np.asarray(
            counterfactuals.attr("cfrl_raw_counterfactual_array"),
            dtype=np.float32,
        )
        full_train_df = counterfactuals.attr("cfrl_full_train_df")
        full_test_df = counterfactuals.attr("cfrl_full_test_df")
        categorical_idx = [
            int(value) for value in counterfactuals.attr("cfrl_categorical_indices")
        ]
        numeric_stats = {
            int(key): (float(value[0]), float(value[1]))
            for key, value in counterfactuals.attr("cfrl_numeric_stats").items()
        }
        X_train = np.asarray(
            counterfactuals.attr("cfrl_train_reference"), dtype=np.float32
        )
        rng = np.random.default_rng(self._seed)
        _ = rng.choice(
            full_test_df.shape[0],
            size=min(factual_df.shape[0], full_test_df.shape[0]),
            replace=False,
        )

        full_test_X = full_test_df.loc[:, self._feature_names]
        full_test_y = full_test_df.loc[:, factuals.target_column].astype(int).to_numpy()
        accuracy = accuracy_score(
            full_test_y, self._predict(full_test_X).argmax(axis=1)
        )

        orig = factual_df.to_numpy(dtype=np.float32)
        cf = raw_cf_array
        pred_labels = self._predict(orig).argmax(axis=1)
        target_labels = 1 - pred_labels
        success = self._predict(cf).argmax(axis=1) == target_labels

        categorical_idx_set = set(categorical_idx)
        immutable_idx = {
            self._feature_names.index(name) for name in self._immutable_features
        }
        numerical_idx = [
            i for i in range(len(self._feature_names)) if i not in categorical_idx
        ]

        cat_l0: list[float] = []
        num_l1: list[float] = []
        violation_flags: list[bool] = []
        for row_orig, row_cf, is_valid in zip(orig, cf, success):
            if not is_valid:
                continue
            cat_changes = 0
            l1_sum = 0.0
            violation = False
            for idx, (val_o, val_c) in enumerate(zip(row_orig, row_cf)):
                if idx in categorical_idx_set:
                    changed = val_o != val_c
                    if changed:
                        cat_changes += 1
                    if idx in immutable_idx:
                        violation |= changed
                    continue
                diff_std = abs(float(val_o) - float(val_c)) / max(
                    numeric_stats[idx][1], 1e-8
                )
                l1_sum += diff_std
                if idx in immutable_idx:
                    violation |= diff_std > 1e-6
            cat_l0.append(cat_changes / max(1, len(categorical_idx)))
            num_l1.append(l1_sum / max(1, len(numerical_idx)))
            violation_flags.append(violation)

        category_values = factuals.attr("cfrl_category_values")
        category_map = {
            idx: [str(value) for value in category_values[name]]
            for idx, name in enumerate(self._feature_names)
            if name in category_values
        }
        he_preprocessor = self._build_he_preprocessor(
            X_train,
            self._feature_names,
            category_map,
        )

        train_preds = self._predict(X_train).argmax(axis=1)
        mmd_per_class: dict[int, float] = {}
        for cls in np.unique(target_labels):
            cls_mask = target_labels == cls
            cf_cls = cf[cls_mask]
            train_target = X_train[train_preds == cls]
            if train_target.shape[0] == 0 or cf_cls.shape[0] == 0:
                continue
            cf_pre = he_preprocessor(cf_cls)
            train_pre = he_preprocessor(train_target)
            hidden_input_dim = cf_pre.shape[1]
            h1_dim, h2_dim, h3_dim = 32, 16, 5
            shared_w1 = rng.standard_normal((hidden_input_dim, h1_dim)) / math.sqrt(
                max(1, hidden_input_dim)
            )
            shared_w2 = rng.standard_normal((h1_dim, h2_dim)) / math.sqrt(h1_dim)
            shared_w3 = rng.standard_normal((h2_dim, h3_dim)) / math.sqrt(h2_dim)

            def _encode_with_weights(
                data: np.ndarray,
                w1: np.ndarray,
                w2: np.ndarray,
                w3: np.ndarray,
            ) -> np.ndarray:
                h1 = np.maximum(0, data @ w1)
                h2 = np.maximum(0, h1 @ w2)
                return h2 @ w3

            shared_cf_enc = _encode_with_weights(
                cf_pre, shared_w1, shared_w2, shared_w3
            )
            shared_ref_enc = _encode_with_weights(
                train_pre, shared_w1, shared_w2, shared_w3
            )
            shared_mmd = self._rbf_mmd(shared_cf_enc, shared_ref_enc, rng)

            separate_cf_w1 = rng.standard_normal(
                (hidden_input_dim, h1_dim)
            ) / math.sqrt(max(1, hidden_input_dim))
            separate_cf_w2 = rng.standard_normal((h1_dim, h2_dim)) / math.sqrt(h1_dim)
            separate_cf_w3 = rng.standard_normal((h2_dim, h3_dim)) / math.sqrt(h2_dim)
            separate_ref_w1 = rng.standard_normal(
                (hidden_input_dim, h1_dim)
            ) / math.sqrt(max(1, hidden_input_dim))
            separate_ref_w2 = rng.standard_normal((h1_dim, h2_dim)) / math.sqrt(h1_dim)
            separate_ref_w3 = rng.standard_normal((h2_dim, h3_dim)) / math.sqrt(h2_dim)

            separate_cf_enc = _encode_with_weights(
                cf_pre, separate_cf_w1, separate_cf_w2, separate_cf_w3
            )
            separate_ref_enc = _encode_with_weights(
                train_pre, separate_ref_w1, separate_ref_w2, separate_ref_w3
            )
            separate_mmd = self._rbf_mmd(separate_cf_enc, separate_ref_enc, rng)

            mmd_per_class[int(cls)] = 0.5 * (shared_mmd + separate_mmd)

        results = {
            "accuracy": float(accuracy),
            "validity": float(success.mean()) if len(success) else 0.0,
            "sparsity_cat_l0": float(np.mean(cat_l0)) if cat_l0 else 0.0,
            "sparsity_num_l1": float(np.mean(num_l1)) if num_l1 else 0.0,
            "immutability_violation_rate": (
                float(np.mean(violation_flags)) if violation_flags else 0.0
            ),
            "target_conditional_mmd_cls_0": float(mmd_per_class.get(0, 0.0)),
            "target_conditional_mmd_cls_1": float(mmd_per_class.get(1, 0.0)),
        }
        return pd.DataFrame([results])


REFERENCE_RESULTS = {
    "accuracy": 0.86,
    "validity": 0.9859,
    "sparsity_cat_l0": 0.11,
    "sparsity_num_l1": 0.19,
    "immutability_violation_rate": 0.0,
    "target_conditional_mmd_cls_0": 0.36,
    "target_conditional_mmd_cls_1": 0.43,
}


def compare_results(
    results: Dict[str, object],
    reference: Dict[str, float],
    tolerances: List[float],
    raise_on_fail: bool = False,
) -> bool:
    ref_items = list(reference.items())
    if len(tolerances) != len(ref_items):
        raise ValueError(
            f"Expected {len(ref_items)} tolerances, received {len(tolerances)}"
        )
    success = True
    failure_messages: List[str] = []

    for idx, (metric_name, ref_value) in enumerate(ref_items):
        tolerance = tolerances[idx]
        result_value = results.get(metric_name)
        if result_value is None:
            message = f"[MISSING] Metric `{metric_name}` not found in results."
            print(message)
            failure_messages.append(message)
            success = False
            continue
        if not isinstance(result_value, (int, float, np.floating, np.integer)):
            message = (
                f"[TYPE] Metric `{metric_name}` is type "
                f"`{type(result_value).__name__}`, expected scalar numeric."
            )
            print(message)
            failure_messages.append(message)
            success = False
            continue

        diff = abs(float(result_value) - float(ref_value))
        if diff <= tolerance:
            print(f"[OK] `{metric_name}` diff {diff:.6f} (tolerance {tolerance})")
        else:
            message = (
                f"[DIFF] `{metric_name}` diff {diff:.6f} exceeds tolerance {tolerance}"
            )
            print(message)
            failure_messages.append(message)
            success = False

    if raise_on_fail and failure_messages:
        raise AssertionError("\n".join(failure_messages))

    return success


def _build_config(config: ExperimentConfig) -> dict:
    experiment_name = "adult_cfrl_rf_reproduce"
    artifact_name = experiment_name
    device = _resolve_device()
    return {
        "name": experiment_name,
        "logger": {
            "level": "INFO",
            "path": f"./logs/{experiment_name}.log",
        },
        "caching": {"path": "./cache/"},
        "dataset": {"name": "adult_cfrl"},
        "preprocess": [
            {
                "name": "cfrl_adult_stratified_split",
                "seed": config.seed,
                "test_size": config.test_size,
                "sample_size": config.sample_size,
            },
            {"name": "finalize"},
        ],
        "model": {
            "name": "cfrl_adult_rf",
            "seed": config.seed,
            "device": device,
            "rf_max_depth": config.rf_max_depth,
            "rf_min_samples_split": config.rf_min_samples_split,
            "rf_n_estimators": config.rf_n_estimators,
            "artifact_name": artifact_name,
        },
        "method": {
            "name": "cfrl",
            "seed": config.seed,
            "device": device,
            "desired_class": None,
            "autoencoder_batch_size": config.autoencoder_batch_size,
            "autoencoder_target_steps": config.autoencoder_target_steps,
            "autoencoder_lr": config.autoencoder_lr,
            "autoencoder_latent_dim": config.autoencoder_latent_dim,
            "autoencoder_hidden_dim": config.autoencoder_hidden_dim,
            "coeff_sparsity": config.cfrl_coeff_sparsity,
            "coeff_consistency": config.cfrl_coeff_consistency,
            "train_steps": config.cfrl_train_steps,
            "batch_size": config.cfrl_batch_size,
            "immutable_features": list(config.immutable_features),
            "constrained_ranges": config.constrained_ranges,
            "train": True,
            "store_reproduction_artifacts": True,
        },
        "evaluation": [
            {
                "name": "cfrl_paper_metrics",
                "artifact_name": artifact_name,
                "seed": config.seed,
                "immutable_features": list(config.immutable_features),
            }
        ],
    }


def run_experiment() -> Dict[str, object]:
    config = ExperimentConfig()
    experiment = Experiment(_build_config(config))
    metrics = experiment.run()
    return metrics.iloc[0].to_dict()


@pytest.mark.parametrize(
    "tolerances",
    [
        [
            1e-2,
            0.0097 * 3,
            0.10 * 3,
            0.06 * 3,
            1e-6,
            0.06 * 3,
            0.13 * 5,
        ]
    ],
)
def test_cfrl_reproduce(tolerances: List[float]) -> None:
    compare_results(
        run_experiment(),
        REFERENCE_RESULTS,
        tolerances=tolerances,
        raise_on_fail=True,
    )


def main():
    results = run_experiment()
    for key, value in results.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            print(f"{key}: {float(value):.6f}")


if __name__ == "__main__":
    main()
