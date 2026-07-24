from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

from joblib import dump, load
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.cogs.reference_support import (  # noqa: E402
    CoGSReferenceDataset,
    ReferenceBlackboxModel,
)
from experiment.utils import write_reproduction_report  # noqa: E402
from method.cogs.cogs import CogsMethod  # noqa: E402
from method.cogs.support import (  # noqa: E402
    compute_ranges_numerical_features,
    gower_distance,
    is_same_point,
    validate_counterfactuals,
)
from utils.seed import seed_context  # noqa: E402


REPORT_PATH = Path(__file__).with_name("reproduction_report.json")
DATA_DIR = Path(__file__).with_name("data")
CACHE_DIR = Path(__file__).with_name("cache")
BLACKBOX_DIR = CACHE_DIR / "blackboxes"
ARTIFACT_DIR = CACHE_DIR / "artifacts"
ARTIFACT_SCHEMA_VERSION = 2
STATUS_FILENAME = "run_status.json"
HEARTBEAT_LOG_FILENAME = "run_heartbeats.log"
DEFAULT_PARALLEL_JOBS = max(1, min(8, os.cpu_count() or 1))
DATASET_ORDER = ["credit", "adult", "boston", "garments", "compas"]
DATASET_ALIASES = {
    "credit": "Cre",
    "adult": "Inc",
    "boston": "Hou",
    "garments": "Pro",
    "compas": "Rec",
}
TABLE4_PAPER = {
    "rf": {dataset_name: {"mean": 1.0, "std": 0.0} for dataset_name in DATASET_ORDER},
    "nn": {dataset_name: {"mean": 1.0, "std": 0.0} for dataset_name in DATASET_ORDER},
}
TABLE5_PAPER = {
    "rf": {
        "C": {
            0.01: {"credit": (0.40, 0.06), "adult": (0.02, 0.00), "boston": (0.76, 0.10), "garments": (0.53, 0.05), "compas": (0.27, 0.06)},
            0.05: {"credit": (0.42, 0.07), "adult": (0.04, 0.02), "boston": (0.84, 0.09), "garments": (0.57, 0.06), "compas": (0.37, 0.07)},
            0.10: {"credit": (0.43, 0.07), "adult": (0.05, 0.02), "boston": (0.85, 0.09), "garments": (0.58, 0.06), "compas": (0.40, 0.09)},
        },
        "K": {
            0.01: {"credit": (0.37, 0.01), "adult": (0.06, 0.02), "boston": (0.33, 0.24), "garments": (0.26, 0.05), "compas": (0.04, 0.04)},
            0.05: {"credit": (0.44, 0.03), "adult": (0.40, 0.08), "boston": (0.63, 0.17), "garments": (0.37, 0.06), "compas": (0.08, 0.03)},
            0.10: {"credit": (0.46, 0.04), "adult": (0.58, 0.07), "boston": (0.67, 0.16), "garments": (0.46, 0.07), "compas": (0.12, 0.02)},
        },
        "CK": {
            0.01: {"credit": (0.23, 0.04), "adult": (0.00, 0.00), "boston": (0.21, 0.21), "garments": (0.19, 0.06), "compas": (0.03, 0.03)},
            0.05: {"credit": (0.27, 0.03), "adult": (0.00, 0.00), "boston": (0.54, 0.21), "garments": (0.26, 0.05), "compas": (0.06, 0.04)},
            0.10: {"credit": (0.30, 0.05), "adult": (0.00, 0.00), "boston": (0.60, 0.19), "garments": (0.34, 0.06), "compas": (0.08, 0.04)},
        },
    },
    "nn": {
        "C": {
            0.01: {"credit": (0.25, 0.12), "adult": (0.01, 0.01), "boston": (0.96, 0.02), "garments": (0.87, 0.05), "compas": (0.50, 0.08)},
            0.05: {"credit": (0.27, 0.12), "adult": (0.02, 0.01), "boston": (0.97, 0.02), "garments": (0.89, 0.05), "compas": (0.56, 0.05)},
            0.10: {"credit": (0.29, 0.11), "adult": (0.02, 0.01), "boston": (0.97, 0.02), "garments": (0.89, 0.05), "compas": (0.57, 0.04)},
        },
        "K": {
            0.01: {"credit": (0.13, 0.07), "adult": (0.35, 0.02), "boston": (0.07, 0.05), "garments": (0.08, 0.06), "compas": (0.01, 0.00)},
            0.05: {"credit": (0.26, 0.08), "adult": (0.52, 0.03), "boston": (0.80, 0.12), "garments": (0.42, 0.19), "compas": (0.01, 0.01)},
            0.10: {"credit": (0.39, 0.02), "adult": (0.70, 0.04), "boston": (0.93, 0.05), "garments": (0.58, 0.14), "compas": (0.02, 0.02)},
        },
        "CK": {
            0.01: {"credit": (0.02, 0.02), "adult": (0.00, 0.00), "boston": (0.07, 0.05), "garments": (0.06, 0.06), "compas": (0.00, 0.01)},
            0.05: {"credit": (0.06, 0.03), "adult": (0.00, 0.00), "boston": (0.69, 0.09), "garments": (0.38, 0.18), "compas": (0.01, 0.01)},
            0.10: {"credit": (0.12, 0.08), "adult": (0.00, 0.00), "boston": (0.93, 0.04), "garments": (0.52, 0.14), "compas": (0.01, 0.02)},
        },
    },
}


def _relative_delta(original: float | int | None, reproduced: float | int | None) -> float | None:
    if original is None or reproduced is None:
        return None
    original_value = float(original)
    reproduced_value = float(reproduced)
    denominator = max(abs(original_value), abs(reproduced_value), 1e-12)
    return float(abs(reproduced_value - original_value) / denominator)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["generate", "analyze", "all"],
        default="all",
    )
    parser.add_argument(
        "--table",
        choices=["table4", "table5", "all"],
        default="all",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=DATASET_ORDER,
    )
    parser.add_argument(
        "--blackboxes",
        nargs="+",
        choices=["rf", "nn"],
        default=["rf", "nn"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model-cv-folds", type=int, default=5)
    parser.add_argument("--fast-model-grid", action="store_true")
    parser.add_argument("--fast-search", action="store_true")
    parser.add_argument("--population-size", type=int, default=1000)
    parser.add_argument("--n-generations", type=int, default=100)
    parser.add_argument("--n-reps", type=int, default=5)
    parser.add_argument("--max-factuals-per-fold", type=int, default=None)
    parser.add_argument("--run-only-fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-robust-samples", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--model-n-jobs", type=int, default=None)
    parser.add_argument("--cf-n-jobs", type=int, default=None)
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--blackbox-dir", default=str(BLACKBOX_DIR))
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--progress", choices=["on", "off"], default="on")
    args, _ = parser.parse_known_args()
    return args


def _resolve_model_n_jobs(args: argparse.Namespace) -> int:
    if args.model_n_jobs is not None:
        return int(args.model_n_jobs)
    if args.n_jobs is not None:
        return int(args.n_jobs)
    return DEFAULT_PARALLEL_JOBS


def _resolve_cf_n_jobs(args: argparse.Namespace) -> int:
    if args.cf_n_jobs is not None:
        return int(args.cf_n_jobs)
    if args.n_jobs is not None:
        return int(args.n_jobs)
    return DEFAULT_PARALLEL_JOBS


def _build_blackbox_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
        "cv_folds": args.model_cv_folds,
        "n_jobs": _resolve_model_n_jobs(args),
        "rf_n_estimators": [50] if args.fast_model_grid else None,
        "rf_min_samples_split": [2] if args.fast_model_grid else None,
        "rf_max_features": ["sqrt"] if args.fast_model_grid else None,
        "nn_learning_rate_init": [1e-2] if args.fast_model_grid else None,
        "nn_solver": ["adam"] if args.fast_model_grid else None,
        "nn_max_iter": [200] if args.fast_model_grid else None,
    }


def _cache_enabled(args: argparse.Namespace) -> bool:
    return not bool(args.no_cache)


def _cache_root(args: argparse.Namespace) -> Path:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _stable_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _slugify(value: object) -> str:
    text = str(value)
    chars = []
    for ch in text:
        if ch.isalnum():
            chars.append(ch.lower())
        else:
            chars.append("_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "value"


def _legacy_cache_file(
    args: argparse.Namespace,
    category: str,
    payload: dict[str, object],
) -> Path:
    digest = _stable_hash(payload)
    root = _cache_root(args) / category
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{digest}.joblib"


def _cache_filename(
    category: str,
    payload: dict[str, object],
) -> str:
    digest = _stable_hash(payload)
    if category == "blackboxes":
        dataset = _slugify(payload.get("dataset", "dataset"))
        blackbox = _slugify(payload.get("blackbox", "blackbox"))
        fold_idx = int(payload.get("fold_idx", -1))
        seed = int(payload.get("seed", -1))
        folds = int(payload.get("folds", -1))
        return (
            f"{blackbox}_prob_{dataset}_fold_{fold_idx}"
            f"__seed_{seed}__cv_{folds}__{digest}.joblib"
        )
    if category == "table4_folds":
        dataset = _slugify(payload.get("dataset", "dataset"))
        blackbox = _slugify(payload.get("blackbox", "blackbox"))
        fold_idx = int(payload.get("fold_idx", -1))
        n_reps = int(payload.get("n_reps", -1))
        max_factuals = payload.get("max_factuals_per_fold")
        max_factuals_slug = "all" if max_factuals is None else _slugify(max_factuals)
        return (
            f"table4_{dataset}_{blackbox}_fold_{fold_idx}"
            f"__reps_{n_reps}__maxfact_{max_factuals_slug}__{digest}.joblib"
        )
    if category == "table5_folds":
        dataset = _slugify(payload.get("dataset", "dataset"))
        blackbox = _slugify(payload.get("blackbox", "blackbox"))
        fold_idx = int(payload.get("fold_idx", -1))
        n_reps = int(payload.get("n_reps", -1))
        k_samples = int(payload.get("k_robust_samples", -1))
        max_factuals = payload.get("max_factuals_per_fold")
        max_factuals_slug = "all" if max_factuals is None else _slugify(max_factuals)
        return (
            f"table5_{dataset}_{blackbox}_fold_{fold_idx}"
            f"__reps_{n_reps}__k_{k_samples}__maxfact_{max_factuals_slug}__{digest}.joblib"
        )
    return f"{digest}.joblib"


def _cache_file(
    args: argparse.Namespace,
    category: str,
    payload: dict[str, object],
) -> Path:
    root = _cache_root(args) / category
    root.mkdir(parents=True, exist_ok=True)
    return root / _cache_filename(category, payload)


def _artifact_summary(args: argparse.Namespace) -> dict[str, object]:
    artifact_root = Path(args.artifact_dir)
    summary: dict[str, dict[str, list[int]]] = {}
    if not artifact_root.exists():
        return {"artifacts": summary}
    for artifact in artifact_root.glob("*.joblib"):
        stem = artifact.stem
        if not stem.startswith("table"):
            continue
        parts = stem.split("_")
        if len(parts) < 5:
            continue
        table_name = parts[0]
        blackbox = parts[-3]
        if parts[-2] != "fold":
            continue
        try:
            fold_idx = int(parts[-1])
        except ValueError:
            continue
        dataset_name = "_".join(parts[1:-3])
        key = f"{table_name}:{dataset_name}:{blackbox}"
        summary.setdefault(key, {"folds": []})
        summary[key]["folds"].append(fold_idx)
    for key in summary:
        summary[key]["folds"] = sorted(set(summary[key]["folds"]))
        summary[key]["count"] = len(summary[key]["folds"])
    return {"artifacts": summary}


def _status_path(args: argparse.Namespace) -> Path:
    return _cache_root(args) / STATUS_FILENAME


def _heartbeat_log_path(args: argparse.Namespace) -> Path:
    return _cache_root(args) / HEARTBEAT_LOG_FILENAME


def _emit_heartbeat(
    args: argparse.Namespace,
    message: str,
    **fields,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "timestamp": timestamp,
        "pid": os.getpid(),
        "message": message,
        **fields,
        **_artifact_summary(args),
    }
    print(f"[{timestamp}] {message}")
    _status_path(args).write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    with _heartbeat_log_path(args).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _load_cache(
    args: argparse.Namespace,
    category: str,
    payload: dict[str, object],
):
    if not _cache_enabled(args) or args.refresh_cache:
        return None
    path = _cache_file(args, category, payload)
    if path.exists():
        return load(path)
    legacy_path = _legacy_cache_file(args, category, payload)
    if legacy_path.exists():
        return load(legacy_path)
    return None


def _save_cache(
    args: argparse.Namespace,
    category: str,
    payload: dict[str, object],
    value,
) -> None:
    if not _cache_enabled(args):
        return
    path = _cache_file(args, category, payload)
    dump(value, path)


def _blackbox_root(args: argparse.Namespace) -> Path:
    root = Path(args.blackbox_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _blackbox_artifact_path(
    args: argparse.Namespace,
    dataset_name: str,
    blackbox: str,
    fold_idx: int,
) -> Path:
    filename = f"{blackbox}_prob_{dataset_name}_fold_{int(fold_idx)}.joblib"
    return _blackbox_root(args) / filename


def _load_blackbox_artifact(
    args: argparse.Namespace,
    dataset_name: str,
    blackbox: str,
    fold_idx: int,
):
    path = _blackbox_artifact_path(args, dataset_name, blackbox, fold_idx)
    if not path.exists():
        return None
    return load(path)


def _save_blackbox_artifact(
    args: argparse.Namespace,
    dataset_name: str,
    blackbox: str,
    fold_idx: int,
    model,
) -> None:
    path = _blackbox_artifact_path(args, dataset_name, blackbox, fold_idx)
    dump(model, path)


def _artifact_root(args: argparse.Namespace) -> Path:
    root = Path(args.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _artifact_path(
    args: argparse.Namespace,
    table_name: str,
    dataset_name: str,
    blackbox: str,
    fold_idx: int,
) -> Path:
    filename = f"{table_name}_{dataset_name}_{blackbox}_fold_{int(fold_idx)}.joblib"
    return _artifact_root(args) / filename


def _load_artifact(
    args: argparse.Namespace,
    table_name: str,
    dataset_name: str,
    blackbox: str,
    fold_idx: int,
):
    path = _artifact_path(args, table_name, dataset_name, blackbox, fold_idx)
    if not path.exists():
        return None
    return load(path)


def _save_artifact(
    args: argparse.Namespace,
    table_name: str,
    dataset_name: str,
    blackbox: str,
    fold_idx: int,
    payload,
) -> None:
    path = _artifact_path(args, table_name, dataset_name, blackbox, fold_idx)
    dump(payload, path)


def _subset_dataset(dataset: CoGSReferenceDataset, row_index: pd.Index, flag: str | None = None):
    subset = dataset.clone()
    full_df = pd.concat([dataset.get(target=False), dataset.get(target=True)], axis=1)
    subset_df = full_df.loc[row_index].copy(deep=True)
    if flag is not None:
        subset.update(flag, True, df=subset_df)
    else:
        subset.update("subset", True, df=subset_df)
    subset.freeze()
    return subset


def _with_plausibility(dataset: CoGSReferenceDataset, enabled: bool):
    if enabled:
        return dataset
    relaxed = dataset.clone()
    feature_columns = dataset.get(target=False).columns.tolist()
    for feature_name in feature_columns:
        relaxed.raw_feature_mutability[feature_name] = True
        relaxed.raw_feature_actionability[feature_name] = "any"
    if hasattr(relaxed, "reference_plausibility_constraints"):
        relaxed.reference_plausibility_constraints = [None] * len(feature_columns)
    relaxed.freeze()
    return relaxed


def _split_dataset(dataset_name: str, seed: int, folds: int):
    dataset = CoGSReferenceDataset(name=dataset_name, path=DATA_DIR)
    dataset.freeze()
    target = dataset.get(target=True).iloc[:, 0].astype(int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = []
    for fold_idx, (train_pos, test_pos) in enumerate(
        splitter.split(np.zeros(len(target)), target.to_numpy())
    ):
        train_index = target.index[train_pos]
        test_index = target.index[test_pos]
        trainset = _subset_dataset(dataset, train_index, flag="trainset")
        testset = _subset_dataset(dataset, test_index, flag="testset")
        splits.append((fold_idx, trainset, testset))
    return dataset, splits


def _select_factuals(
    model: ReferenceBlackboxModel,
    testset,
    desired_class: int,
    max_factuals_per_fold: int | None,
    seed: int,
):
    predictions = model.predict(testset).argmax(dim=1).detach().cpu().numpy()
    features = testset.get(target=False)
    candidate_mask = predictions != int(model.get_class_to_index()[desired_class])
    candidate_index = features.index[candidate_mask]
    if max_factuals_per_fold is None or len(candidate_index) <= max_factuals_per_fold:
        chosen_index = candidate_index
    else:
        rng = np.random.RandomState(seed)
        positions = rng.choice(
            len(candidate_index),
            size=int(max_factuals_per_fold),
            replace=False,
        )
        chosen_index = candidate_index[np.sort(positions)]
    return _subset_dataset(testset, chosen_index), int(len(candidate_index))


def _compute_candidate_loss(
    method: CogsMethod,
    factual: np.ndarray,
    candidate: np.ndarray,
    desired_class: int,
) -> float:
    if np.isnan(candidate).any():
        return float("inf")
    num_feature_ranges = compute_ranges_numerical_features(
        method._feature_intervals,
        method._indices_categorical_features,
    )
    gower_dist = gower_distance(
        candidate,
        factual,
        num_feature_ranges,
        method._indices_categorical_features,
    )
    l0 = float(np.sum(candidate != factual) / len(factual))
    predicted = method._adapter.predict(candidate.reshape(1, -1))[0]
    failed = float(predicted != desired_class)
    return float(0.5 * gower_dist + 0.5 * l0 + failed)


def _serialize_vector(values: np.ndarray) -> str:
    return json.dumps(np.asarray(values, dtype=np.float64).tolist())


def _deserialize_vector(values: str) -> np.ndarray:
    return np.asarray(json.loads(values), dtype=np.float64)


def _compute_candidate_metrics(
    method: CogsMethod,
    factual: np.ndarray,
    candidate: np.ndarray,
    desired_class: int,
) -> tuple[float, float, float]:
    if np.isnan(candidate).any():
        return float("nan"), float("inf"), float("nan")
    num_feature_ranges = compute_ranges_numerical_features(
        method._feature_intervals,
        method._indices_categorical_features,
    )
    gower_dist = float(
        gower_distance(
            candidate,
            factual,
            num_feature_ranges,
            method._indices_categorical_features,
        )
    )
    sparsity = float(1.0 - (np.sum(candidate != factual) / len(factual)))
    loss = _compute_candidate_loss(method, factual, candidate, desired_class)
    return gower_dist, loss, sparsity


def _run_best_of_repetitions(
    method: CogsMethod,
    factual_row: pd.Series,
    desired_class: int,
    n_reps: int,
    validate_candidate: bool = True,
    metadata: dict[str, object] | None = None,
):
    factual_df = factual_row.to_frame().T
    factual_array = factual_row.to_numpy(dtype=np.float64)
    best_candidate: pd.Series | None = None
    best_loss = float("inf")
    best_runtime = float("nan")
    base_seed = 0 if method._seed is None else int(method._seed)
    repetition_records: list[dict[str, object]] = []

    for rep_idx in range(n_reps):
        rep_seed = base_seed + rep_idx
        started_at = time.perf_counter()
        with seed_context(rep_seed):
            raw_candidate = method._search_counterfactual(
                factual=factual_array,
                desired_class=desired_class,
            )
        if raw_candidate is None:
            candidate_series = pd.Series(np.nan, index=factual_row.index, dtype=np.float64)
        else:
            candidate_series = pd.Series(raw_candidate, index=factual_row.index, dtype=np.float64)
        if validate_candidate:
            candidate_df = validate_counterfactuals(
                target_model=method._target_model,
                factuals=factual_df,
                candidates=candidate_series.to_frame().T,
                desired_class=desired_class,
            )
            candidate_series = candidate_df.iloc[0].copy(deep=True)
        runtime_seconds = float(time.perf_counter() - started_at)
        candidate = candidate_series.to_numpy(dtype=np.float64)
        gower_dist, loss, sparsity = _compute_candidate_metrics(
            method=method,
            factual=factual_array,
            candidate=candidate,
            desired_class=desired_class,
        )
        pred_class_z = None
        if not np.isnan(candidate).any():
            pred_class_z = method._adapter.predict(candidate.reshape(1, -1))[0]
            if isinstance(pred_class_z, np.generic):
                pred_class_z = pred_class_z.item()
        record = {
            "rep_idx": int(rep_idx),
            "x": _serialize_vector(factual_array),
            "z": _serialize_vector(candidate),
            "pred_class_z": pred_class_z,
            "run_time": runtime_seconds,
            "gower_dist": gower_dist,
            "loss": loss,
            "sparsity": sparsity,
        }
        if metadata is not None:
            record.update(metadata)
        repetition_records.append(record)
        if loss < best_loss:
            best_loss = loss
            best_runtime = runtime_seconds
            best_candidate = candidate_series

    if best_candidate is None:
        best_candidate = pd.Series(np.nan, index=factual_row.index, dtype=np.float64)
    return best_candidate, best_loss, best_runtime, repetition_records


def _compute_success_rate(
    model: ReferenceBlackboxModel,
    desired_class: int,
    factuals,
    counterfactuals: pd.DataFrame,
) -> float:
    if counterfactuals.shape[0] == 0:
        return float("nan")
    valid_mask = ~counterfactuals.isna().any(axis=1)
    if not bool(valid_mask.any()):
        return 0.0
    predictions = (
        model.get_prediction(counterfactuals.loc[valid_mask], proba=False)
        .argmax(dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    desired_index = int(model.get_class_to_index()[desired_class])
    successes = int(np.sum(predictions == desired_index))
    return float(successes / counterfactuals.shape[0])


def _build_method_kwargs(
    desired_class: int,
    seed: int,
    robust_type: str | None,
    k_robust_samples: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    population_size = 200 if args.fast_search else int(args.population_size)
    n_generations = 30 if args.fast_search else int(args.n_generations)
    return {
        "seed": seed,
        "device": "cpu",
        "desired_class": desired_class,
        "evolution_type": "classic",
        "population_size": population_size,
        "n_generations": n_generations,
        "mutation_probability": "inv_mutable_genotype_length",
        "num_features_mutation_strength": 0.25,
        "init_temperature": None,
        "selection_name": "tournament_2",
        "noisy_evaluations": False,
        "optimize_c_robust": robust_type in {"C", "CK"},
        "optimize_k_robust": k_robust_samples if robust_type in {"K", "CK"} else 0,
        "verbose": False,
    }


def _blackbox_cache_key(
    dataset_name: str,
    blackbox: str,
    fold_idx: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "dataset": dataset_name,
        "blackbox": blackbox,
        "fold_idx": int(fold_idx),
        "seed": int(args.seed),
        "folds": int(args.folds),
        "blackbox_artifact_path": str(
            _blackbox_artifact_path(args, dataset_name, blackbox, fold_idx)
        ),
        "blackbox_kwargs": _build_blackbox_kwargs(args),
    }


def _table4_fold_cache_key(
    dataset_name: str,
    blackbox: str,
    fold_idx: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "dataset": dataset_name,
        "blackbox": blackbox,
        "fold_idx": int(fold_idx),
        "table": "table4",
        "seed": int(args.seed),
        "folds": int(args.folds),
        "n_reps": int(args.n_reps),
        "max_factuals_per_fold": args.max_factuals_per_fold,
        "candidate_mode": "raw_best_found",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "plausibility_enabled": False,
        "blackbox_kwargs": _build_blackbox_kwargs(args),
        "method_kwargs": _build_method_kwargs(
            desired_class=-1,
            seed=args.seed,
            robust_type=None,
            k_robust_samples=0,
            args=args,
        ),
    }


def _table5_fold_cache_key(
    dataset_name: str,
    blackbox: str,
    fold_idx: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "dataset": dataset_name,
        "blackbox": blackbox,
        "fold_idx": int(fold_idx),
        "table": "table5",
        "seed": int(args.seed),
        "folds": int(args.folds),
        "n_reps": int(args.n_reps),
        "max_factuals_per_fold": args.max_factuals_per_fold,
        "k_robust_samples": int(args.k_robust_samples),
        "candidate_mode": "raw_best_found",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "plausibility_enabled": True,
        "blackbox_kwargs": _build_blackbox_kwargs(args),
        "method_configs": {
            label: _build_method_kwargs(
                desired_class=-1,
                seed=args.seed,
                robust_type=robust_type,
                k_robust_samples=args.k_robust_samples,
                args=args,
            )
            for label, robust_type in {
                "base": None,
                "C": "C",
                "K": "K",
                "CK": "CK",
            }.items()
        },
    }


def _chunk_records(
    records: list[tuple[object, list[float], dict[str, object]]],
    num_chunks: int,
) -> list[list[tuple[object, list[float], dict[str, object]]]]:
    if len(records) == 0:
        return []
    chunk_count = max(1, min(int(num_chunks), len(records)))
    chunk_size = int(math.ceil(len(records) / chunk_count))
    return [
        records[start : start + chunk_size]
        for start in range(0, len(records), chunk_size)
    ]


def _run_factual_chunk(
    trainset,
    model: ReferenceBlackboxModel,
    method_kwargs: dict[str, object],
    feature_columns: list[str],
    desired_class: int,
    n_reps: int,
    validate_candidates: bool,
    chunk: list[tuple[object, list[float]]],
    base_record_metadata: dict[str, object],
) -> list[tuple[object, list[float], float, list[dict[str, object]]]]:
    method = CogsMethod(target_model=model, **method_kwargs)
    method.fit(trainset)

    results: list[tuple[object, list[float], float, list[dict[str, object]]]] = []
    for row_index, factual_values, factual_metadata in chunk:
        factual_row = pd.Series(
            factual_values,
            index=feature_columns,
            dtype=np.float64,
        )
        record_metadata = dict(base_record_metadata)
        record_metadata.update(factual_metadata)
        best_candidate, _, runtime_seconds, repetition_records = _run_best_of_repetitions(
            method=method,
            factual_row=factual_row,
            desired_class=desired_class,
            n_reps=n_reps,
            validate_candidate=validate_candidates,
            metadata=record_metadata,
        )
        results.append(
            (
                row_index,
                best_candidate.to_numpy(dtype=np.float64).tolist(),
                float(runtime_seconds),
                repetition_records,
            )
        )
    return results


def _generate_counterfactual_rows(
    trainset,
    model: ReferenceBlackboxModel,
    factuals,
    method_kwargs: dict[str, object],
    desired_class: int,
    args: argparse.Namespace,
    progress_label: str,
    base_record_metadata: dict[str, object],
    validate_candidates: bool = True,
) -> tuple[dict[object, pd.Series], list[float], pd.DataFrame]:
    factual_features = factuals.get(target=False)
    factual_targets = factuals.get(target=True).iloc[:, 0]
    feature_columns = factual_features.columns.tolist()
    cf_n_jobs = _resolve_cf_n_jobs(args)
    pred_class_x = model.predict(factuals).argmax(dim=1).detach().cpu().numpy()
    class_to_index = model.get_class_to_index()
    index_to_class = {index: label for label, index in class_to_index.items()}
    test_sample_idx_map = {
        row_index: int(position)
        for position, row_index in enumerate(factual_features.index.tolist())
    }
    factual_records = [
        (
            row_index,
            row.to_numpy(dtype=np.float64).tolist(),
            {
                "test_sample_idx": test_sample_idx_map[row_index],
                "pred_class_x": int(index_to_class[int(pred_class_x[position])]),
                "true_class_x": int(factual_targets.loc[row_index]),
                "desired_class": int(desired_class),
            },
        )
        for position, (row_index, row) in enumerate(factual_features.iterrows())
    ]
    if len(factual_records) == 0:
        return {}, [], pd.DataFrame()

    if cf_n_jobs <= 1:
        method = CogsMethod(target_model=model, **method_kwargs)
        method.fit(trainset)
        iterator = factual_features.iterrows()
        if args.progress == "on":
            iterator = tqdm(
                iterator,
                total=len(factuals),
                desc=progress_label,
                leave=False,
            )
        rows: dict[object, pd.Series] = {}
        runtimes: list[float] = []
        repetition_records: list[dict[str, object]] = []
        for position, (row_index, factual_row) in enumerate(iterator):
            record_metadata = dict(base_record_metadata)
            record_metadata.update(
                {
                    "test_sample_idx": test_sample_idx_map[row_index],
                    "pred_class_x": int(index_to_class[int(pred_class_x[position])]),
                    "true_class_x": int(factual_targets.loc[row_index]),
                    "desired_class": int(desired_class),
                }
            )
            best_candidate, _, runtime_seconds, row_records = _run_best_of_repetitions(
                method=method,
                factual_row=factual_row,
                desired_class=desired_class,
                n_reps=args.n_reps,
                validate_candidate=validate_candidates,
                metadata=record_metadata,
            )
            rows[row_index] = best_candidate
            runtimes.append(float(runtime_seconds))
            repetition_records.extend(row_records)
        return rows, runtimes, pd.DataFrame.from_records(repetition_records)

    if args.progress == "on":
        print(
            f"parallel factual search: {progress_label} with cf_n_jobs={cf_n_jobs}"
        )

    chunks = _chunk_records(
        factual_records,
        num_chunks=cf_n_jobs * 4,
    )
    chunk_results = Parallel(
        n_jobs=cf_n_jobs,
        backend="loky",
        batch_size="auto",
        pre_dispatch="2*n_jobs",
    )(
        delayed(_run_factual_chunk)(
            trainset=trainset,
            model=model,
            method_kwargs=method_kwargs,
            feature_columns=feature_columns,
            desired_class=desired_class,
            n_reps=args.n_reps,
            validate_candidates=validate_candidates,
            chunk=chunk,
            base_record_metadata=base_record_metadata,
        )
        for chunk in chunks
    )

    rows = {}
    runtimes = []
    repetition_records: list[dict[str, object]] = []
    for chunk in chunk_results:
        for row_index, candidate_values, runtime_seconds, row_records in chunk:
            rows[row_index] = pd.Series(
                candidate_values,
                index=feature_columns,
                dtype=np.float64,
            )
            runtimes.append(float(runtime_seconds))
            repetition_records.extend(row_records)
    rows = {row_index: rows[row_index] for row_index in factual_features.index}
    return rows, runtimes, pd.DataFrame.from_records(repetition_records)


def _table4_run(
    dataset_name: str,
    blackbox: str,
    fold_idx: int,
    trainset,
    testset,
    args: argparse.Namespace,
) -> dict[str, object]:
    cache_key = _table4_fold_cache_key(
        dataset_name=dataset_name,
        blackbox=blackbox,
        fold_idx=fold_idx,
        args=args,
    )
    cached = _load_cache(args, "table4_folds", cache_key)
    if cached is not None:
        return cached

    desired_class = int(trainset.best_class)
    search_trainset = _with_plausibility(trainset, enabled=False)
    model_cache_key = _blackbox_cache_key(
        dataset_name=dataset_name,
        blackbox=blackbox,
        fold_idx=fold_idx,
        args=args,
    )
    model = _load_cache(args, "blackboxes", model_cache_key)
    if model is None:
        model = _load_blackbox_artifact(args, dataset_name, blackbox, fold_idx)
        if model is None:
            model = ReferenceBlackboxModel(
                kind=blackbox,
                **_build_blackbox_kwargs(args),
            )
            model.fit(trainset)
            _save_blackbox_artifact(args, dataset_name, blackbox, fold_idx, model)
        _save_cache(args, "blackboxes", model_cache_key, model)
    test_accuracy = accuracy_score(
        testset.get(target=True).iloc[:, 0].astype(int).to_numpy(),
        model.predict(testset).argmax(dim=1).detach().cpu().numpy(),
    )
    factuals, candidate_pool_size = _select_factuals(
        model=model,
        testset=testset,
        desired_class=desired_class,
        max_factuals_per_fold=args.max_factuals_per_fold,
        seed=args.seed,
    )
    _emit_heartbeat(
        args,
        f"running table4 dataset={dataset_name} blackbox={blackbox} fold={fold_idx}",
        phase="table4",
        dataset=dataset_name,
        blackbox=blackbox,
        fold_idx=int(fold_idx),
        candidate_pool_size=int(candidate_pool_size),
        evaluated_factuals=int(len(factuals)),
    )
    method_kwargs = _build_method_kwargs(
        desired_class=desired_class,
        seed=args.seed,
        robust_type=None,
        k_robust_samples=0,
        args=args,
    )
    row_map, runtimes, raw_results = _generate_counterfactual_rows(
        trainset=search_trainset,
        model=model,
        factuals=factuals,
        method_kwargs=method_kwargs,
        desired_class=desired_class,
        args=args,
        progress_label=f"table4-{dataset_name}-{blackbox}",
        base_record_metadata={
            "dataset": dataset_name,
            "fold_idx": int(fold_idx),
            "blackbox": blackbox,
            "blackbox_test_acc": float(test_accuracy),
            "check_plausibility": False,
            "opt_C_robust": False,
            "opt_K_robust": 0,
            "overall_seed": int(args.seed),
        },
        validate_candidates=False,
    )
    counterfactuals = pd.DataFrame(
        [row_map[row_index] for row_index in factuals.get(target=False).index],
        index=factuals.get(target=False).index,
        columns=factuals.get(target=False).columns,
    )
    success_rate = _compute_success_rate(
        model=model,
        desired_class=desired_class,
        factuals=factuals,
        counterfactuals=counterfactuals,
    )
    raw_best = _reduce_to_best_found(raw_results)
    notebook_mean, notebook_std = _counterfactual_discovery_success_rate(raw_best)
    result = {
        "test_accuracy": float(test_accuracy),
        "candidate_pool_size": candidate_pool_size,
        "evaluated_factuals": int(counterfactuals.shape[0]),
        "success_rate": success_rate,
        "notebook_success_rate_mean": notebook_mean,
        "notebook_success_rate_std": notebook_std,
        "best_params": model.get_best_params(),
        "avg_selected_runtime": float(np.nanmean(runtimes)) if runtimes else float("nan"),
        "raw_results": raw_results,
    }
    _emit_heartbeat(
        args,
        f"completed table4 dataset={dataset_name} blackbox={blackbox} fold={fold_idx}",
        phase="table4",
        dataset=dataset_name,
        blackbox=blackbox,
        fold_idx=int(fold_idx),
        candidate_pool_size=int(candidate_pool_size),
        evaluated_factuals=int(counterfactuals.shape[0]),
        success_rate=float(success_rate),
    )
    _save_cache(args, "table4_folds", cache_key, result)
    return result


def _build_method(
    model: ReferenceBlackboxModel,
    desired_class: int,
    seed: int,
    check_plausibility: bool,
    robust_type: str | None,
    k_robust_samples: int,
    args: argparse.Namespace,
) -> CogsMethod:
    return CogsMethod(
        target_model=model,
        **_build_method_kwargs(
            desired_class=desired_class,
            seed=seed,
            robust_type=robust_type,
            k_robust_samples=k_robust_samples,
            args=args,
        ),
    )


def _table5_candidates(
    dataset_name: str,
    blackbox: str,
    fold_idx: int,
    trainset,
    testset,
    args: argparse.Namespace,
):
    cache_key = _table5_fold_cache_key(
        dataset_name=dataset_name,
        blackbox=blackbox,
        fold_idx=fold_idx,
        args=args,
    )
    cached = _load_cache(args, "table5_folds", cache_key)
    if cached is not None:
        return cached

    desired_class = int(trainset.best_class)
    model_cache_key = _blackbox_cache_key(
        dataset_name=dataset_name,
        blackbox=blackbox,
        fold_idx=fold_idx,
        args=args,
    )
    model = _load_cache(args, "blackboxes", model_cache_key)
    if model is None:
        model = _load_blackbox_artifact(args, dataset_name, blackbox, fold_idx)
        if model is None:
            model = ReferenceBlackboxModel(
                kind=blackbox,
                **_build_blackbox_kwargs(args),
            )
            model.fit(trainset)
            _save_blackbox_artifact(args, dataset_name, blackbox, fold_idx, model)
        _save_cache(args, "blackboxes", model_cache_key, model)
    factuals, candidate_pool_size = _select_factuals(
        model=model,
        testset=testset,
        desired_class=desired_class,
        max_factuals_per_fold=args.max_factuals_per_fold,
        seed=args.seed,
    )
    test_accuracy = float(
        accuracy_score(
            testset.get(target=True).iloc[:, 0].astype(int).to_numpy(),
            model.predict(testset).argmax(dim=1).detach().cpu().numpy(),
        )
    )

    results = {}
    factual_features = factuals.get(target=False).copy(deep=True)
    for robust_type in [None, "C", "K", "CK"]:
        method_kwargs = _build_method_kwargs(
            desired_class=desired_class,
            seed=args.seed,
            robust_type=robust_type,
            k_robust_samples=args.k_robust_samples,
            args=args,
        )
        label = "base" if robust_type is None else robust_type.lower()
        _emit_heartbeat(
            args,
            f"running table5 dataset={dataset_name} blackbox={blackbox} fold={fold_idx} robust_type={label}",
            phase="table5",
            dataset=dataset_name,
            blackbox=blackbox,
            fold_idx=int(fold_idx),
            robust_type=label,
            candidate_pool_size=int(candidate_pool_size),
            evaluated_factuals=int(len(factuals)),
        )
        rows, _, raw_results = _generate_counterfactual_rows(
            trainset=trainset,
            model=model,
            factuals=factuals,
            method_kwargs=method_kwargs,
            desired_class=desired_class,
            args=args,
            progress_label=f"table5-{dataset_name}-{blackbox}-{label}",
            base_record_metadata={
                "dataset": dataset_name,
                "fold_idx": int(fold_idx),
                "blackbox": blackbox,
                "blackbox_test_acc": test_accuracy,
                "check_plausibility": True,
                "opt_C_robust": bool(robust_type in {"C", "CK"}),
                "opt_K_robust": int(args.k_robust_samples if robust_type in {"K", "CK"} else 0),
                "overall_seed": int(args.seed),
            },
            validate_candidates=False,
        )
        results["base" if robust_type is None else robust_type] = {
            "counterfactuals": pd.DataFrame(
                [rows[row_index] for row_index in factual_features.index],
                index=factual_features.index,
                columns=factual_features.columns,
            ),
            "raw_results": raw_results,
        }
        _emit_heartbeat(
            args,
            f"completed table5 dataset={dataset_name} blackbox={blackbox} fold={fold_idx} robust_type={label}",
            phase="table5",
            dataset=dataset_name,
            blackbox=blackbox,
            fold_idx=int(fold_idx),
            robust_type=label,
            candidate_pool_size=int(candidate_pool_size),
            evaluated_factuals=int(len(factuals)),
        )

    payload = {
        "candidate_pool_size": candidate_pool_size,
        "factual_features": factual_features,
        "feature_intervals": deepcopy(factuals.feature_intervals),
        "raw_feature_type": deepcopy(factuals.raw_feature_type),
        "results": results,
    }
    _save_cache(args, "table5_folds", cache_key, payload)
    return payload


def _agreement_rate(
    factual_features: pd.DataFrame,
    feature_intervals: list[object],
    raw_feature_type: dict[str, str],
    base_rows: pd.DataFrame,
    robust_rows: pd.DataFrame,
    tol: float,
) -> float:
    if len(base_rows) == 0:
        return float("nan")
    method_feature_intervals = np.asarray(feature_intervals, dtype=object)
    categorical_indices = [
        index
        for index, feature_name in enumerate(factual_features.columns.tolist())
        if str(raw_feature_type[feature_name]).lower() != "numerical"
    ]

    matches = 0
    total = 0
    for row_index in factual_features.index:
        x = factual_features.loc[row_index].to_numpy(dtype=np.float64)
        z_base = base_rows.loc[row_index].to_numpy(dtype=np.float64)
        z_robust = robust_rows.loc[row_index].to_numpy(dtype=np.float64)
        total += 1
        if np.isnan(z_base).any() or np.isnan(z_robust).any():
            continue
        matches += int(
            is_same_point(
                z_1=z_base,
                z_2=z_robust,
                x=x,
                feature_intervals=method_feature_intervals,
                indices_categorical_features=categorical_indices,
                tol=tol,
            )
        )
    return float(matches / total) if total > 0 else float("nan")


def _reduce_to_best_found(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy(deep=True)
    cols_to_group_by = [
        "blackbox_test_acc",
        "blackbox",
        "test_sample_idx",
        "fold_idx",
        "dataset",
        "check_plausibility",
        "opt_C_robust",
        "opt_K_robust",
    ]
    best_df = df[
        df["loss"] == df.groupby(cols_to_group_by)["loss"].transform("min")
    ].copy(deep=True)
    best_df.drop_duplicates(subset=cols_to_group_by, keep="first", inplace=True)
    best_df.reset_index(drop=True, inplace=True)
    return best_df


def _counterfactual_discovery_success_rate(df: pd.DataFrame) -> tuple[float, float]:
    if df.empty:
        return float("nan"), float("nan")
    curr_df = df.copy(deep=True)
    curr_df["success"] = curr_df["desired_class"] == curr_df["pred_class_z"]
    fold_rates = []
    for fold_idx in curr_df["fold_idx"].drop_duplicates().tolist():
        fold_df = curr_df[curr_df["fold_idx"] == fold_idx]
        if len(fold_df) == 0:
            continue
        fold_rates.append(float(fold_df["success"].sum() / len(fold_df)))
    if not fold_rates:
        return float("nan"), float("nan")
    return float(np.mean(fold_rates)), float(np.std(fold_rates))


def _match_rate_from_raw_results(
    factual_features: pd.DataFrame,
    feature_intervals: list[object],
    raw_feature_type: dict[str, str],
    base_df: pd.DataFrame,
    robust_df: pd.DataFrame,
    tol: float,
) -> float:
    if base_df.empty or robust_df.empty:
        return float("nan")
    merged = base_df.merge(
        robust_df,
        on=["fold_idx", "test_sample_idx"],
        suffixes=("_base", "_robust"),
        how="inner",
    )
    if merged.empty:
        return float("nan")

    method_feature_intervals = np.asarray(feature_intervals, dtype=object)
    categorical_indices = [
        index
        for index, feature_name in enumerate(factual_features.columns.tolist())
        if str(raw_feature_type[feature_name]).lower() != "numerical"
    ]
    factual_map = {
        int(position): factual_features.loc[row_index].to_numpy(dtype=np.float64)
        for position, row_index in enumerate(factual_features.index.tolist())
    }

    matches = 0
    total = 0
    for _, row in merged.iterrows():
        sample_idx = int(row["test_sample_idx"])
        if sample_idx not in factual_map:
            continue
        total += 1
        z_base = _deserialize_vector(row["z_base"])
        z_robust = _deserialize_vector(row["z_robust"])
        if np.isnan(z_base).any() or np.isnan(z_robust).any():
            continue
        matches += int(
            is_same_point(
                z_1=z_base,
                z_2=z_robust,
                x=factual_map[sample_idx],
                feature_intervals=method_feature_intervals,
                indices_categorical_features=categorical_indices,
                tol=tol,
            )
        )
    return float(matches / total) if total > 0 else float("nan")


def _table4_metric_deltas(metrics: dict[str, object]) -> dict[str, float | None]:
    paper = metrics["paper"]
    return {
        "mean": _relative_delta(paper["mean"], metrics["mean_success_rate"]),
        "std": _relative_delta(paper["std"], metrics["std_success_rate"]),
    }


def _table5_metric_deltas(metrics: dict[str, object]) -> dict[str, float | None]:
    paper_mean, paper_std = metrics["paper"]
    return {
        "mean": _relative_delta(paper_mean, metrics["mean_match_rate"]),
        "std": _relative_delta(paper_std, metrics["std_match_rate"]),
    }


def _required_tables(args: argparse.Namespace) -> set[str]:
    if args.table == "all":
        return {"table4", "table5"}
    return {str(args.table)}


def _iter_requested_splits(args: argparse.Namespace):
    for dataset_name in args.datasets:
        _, splits = _split_dataset(dataset_name, seed=args.seed, folds=args.folds)
        for fold_idx, trainset, testset in splits:
            if args.run_only_fold is not None and fold_idx != args.run_only_fold:
                continue
            yield dataset_name, fold_idx, trainset, testset


def _generate_artifacts(args: argparse.Namespace) -> None:
    required_tables = _required_tables(args)
    _emit_heartbeat(
        args,
        "starting CoGS artifact generation",
        phase="generate",
        requested_tables=sorted(required_tables),
        datasets=list(args.datasets),
        blackboxes=list(args.blackboxes),
        folds=int(args.folds),
        n_reps=int(args.n_reps),
        k_robust_samples=int(args.k_robust_samples),
    )
    for blackbox in args.blackboxes:
        for dataset_name, fold_idx, trainset, testset in _iter_requested_splits(args):
            table4_artifact = _artifact_path(args, "table4", dataset_name, blackbox, fold_idx)
            if "table4" in required_tables and (args.refresh_cache or not table4_artifact.exists()):
                fold_result = _table4_run(
                    dataset_name=dataset_name,
                    blackbox=blackbox,
                    fold_idx=fold_idx,
                    trainset=trainset,
                    testset=testset,
                    args=args,
                )
                payload = {
                    "table": "table4",
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "dataset": dataset_name,
                    "blackbox": blackbox,
                    "fold_idx": int(fold_idx),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "result": fold_result,
                }
                _save_artifact(args, "table4", dataset_name, blackbox, fold_idx, payload)

            table5_artifact = _artifact_path(args, "table5", dataset_name, blackbox, fold_idx)
            if "table5" in required_tables and (args.refresh_cache or not table5_artifact.exists()):
                fold_candidates = _table5_candidates(
                    dataset_name=dataset_name,
                    blackbox=blackbox,
                    fold_idx=fold_idx,
                    trainset=trainset,
                    testset=testset,
                    args=args,
                )
                payload = {
                    "table": "table5",
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "dataset": dataset_name,
                    "blackbox": blackbox,
                    "fold_idx": int(fold_idx),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "payload": fold_candidates,
                }
                _save_artifact(args, "table5", dataset_name, blackbox, fold_idx, payload)
            else:
                if "table5" in required_tables:
                    _emit_heartbeat(
                        args,
                        f"cache hit for table5 dataset={dataset_name} blackbox={blackbox} fold={fold_idx}",
                        phase="table5",
                        dataset=dataset_name,
                        blackbox=blackbox,
                        fold_idx=int(fold_idx),
                        artifact_path=str(table5_artifact),
                    )
    _emit_heartbeat(
        args,
        "completed CoGS artifact generation",
        phase="generate",
        requested_tables=sorted(required_tables),
    )


def _run_table4(args: argparse.Namespace) -> dict[str, object]:
    results: dict[str, object] = {}
    for blackbox in args.blackboxes:
        results[blackbox] = {}
        for dataset_name in args.datasets:
            fold_success_rates = []
            fold_accuracies = []
            fold_details = []
            for fold_idx in range(int(args.folds)):
                if args.run_only_fold is not None and fold_idx != args.run_only_fold:
                    continue
                artifact = _load_artifact(args, "table4", dataset_name, blackbox, fold_idx)
                if artifact is None:
                    raise FileNotFoundError(
                        f"Missing table4 artifact for dataset={dataset_name}, blackbox={blackbox}, fold={fold_idx}. "
                        "Run with --mode generate or --mode all first."
                    )
                fold_result = artifact["result"]
                fold_success_rates.append(
                    float(
                        fold_result.get(
                            "notebook_success_rate_mean",
                            fold_result["success_rate"],
                        )
                    )
                )
                fold_accuracies.append(float(fold_result["test_accuracy"]))
                fold_summary = {
                    key: value
                    for key, value in fold_result.items()
                    if key != "raw_results"
                }
                if "raw_results" in fold_result:
                    fold_summary["raw_result_rows"] = int(len(fold_result["raw_results"]))
                fold_details.append({"fold_idx": fold_idx, **fold_summary})
            dataset_metrics = {
                "mean_success_rate": float(np.mean(fold_success_rates)) if fold_success_rates else float("nan"),
                "std_success_rate": float(np.std(fold_success_rates)) if fold_success_rates else float("nan"),
                "mean_test_accuracy": float(np.mean(fold_accuracies)) if fold_accuracies else float("nan"),
                "paper": TABLE4_PAPER[blackbox][dataset_name],
                "folds": fold_details,
            }
            dataset_metrics["delta"] = _table4_metric_deltas(dataset_metrics)
            results[blackbox][dataset_name] = dataset_metrics
    return results


def _run_table5(args: argparse.Namespace) -> dict[str, object]:
    tolerances = [0.01, 0.05, 0.10]
    results: dict[str, object] = {}
    for blackbox in args.blackboxes:
        results[blackbox] = {}
        fold_cache: dict[str, list[dict[str, object]]] = {}
        for dataset_name in args.datasets:
            cached_folds = []
            for fold_idx in range(int(args.folds)):
                if args.run_only_fold is not None and fold_idx != args.run_only_fold:
                    continue
                artifact = _load_artifact(args, "table5", dataset_name, blackbox, fold_idx)
                if artifact is None:
                    raise FileNotFoundError(
                        f"Missing table5 artifact for dataset={dataset_name}, blackbox={blackbox}, fold={fold_idx}. "
                        "Run with --mode generate or --mode all first."
                    )
                fold_candidates = artifact["payload"]
                cached_folds.append({"fold_idx": fold_idx, "payload": fold_candidates})
            fold_cache[dataset_name] = cached_folds
        for robust_type in ["C", "K", "CK"]:
            results[blackbox][robust_type] = {}
            for tol in tolerances:
                results[blackbox][robust_type][tol] = {}
                for dataset_name in args.datasets:
                    fold_match_rates = []
                    fold_details = []
                    for fold_entry in fold_cache[dataset_name]:
                        fold_idx = int(fold_entry["fold_idx"])
                        fold_candidates = fold_entry["payload"]
                        base_payload = fold_candidates["results"]["base"]
                        robust_payload = fold_candidates["results"][robust_type]
                        if isinstance(base_payload, pd.DataFrame):
                            base_rows = base_payload
                            robust_rows = robust_payload
                            fold_rate = _agreement_rate(
                                factual_features=fold_candidates["factual_features"],
                                feature_intervals=fold_candidates["feature_intervals"],
                                raw_feature_type=fold_candidates["raw_feature_type"],
                                base_rows=base_rows,
                                robust_rows=robust_rows,
                                tol=tol,
                            )
                        else:
                            base_rows = base_payload["counterfactuals"]
                            robust_rows = robust_payload["counterfactuals"]
                            fold_rate = _match_rate_from_raw_results(
                                factual_features=fold_candidates["factual_features"],
                                feature_intervals=fold_candidates["feature_intervals"],
                                raw_feature_type=fold_candidates["raw_feature_type"],
                                base_df=_reduce_to_best_found(base_payload["raw_results"]),
                                robust_df=_reduce_to_best_found(robust_payload["raw_results"]),
                                tol=tol,
                            )
                        fold_match_rates.append(fold_rate)
                        fold_details.append(
                            {
                                "fold_idx": fold_idx,
                                "candidate_pool_size": fold_candidates["candidate_pool_size"],
                                "evaluated_factuals": int(base_rows.shape[0]),
                                "match_rate": fold_rate,
                            }
                        )
                    dataset_metrics = {
                        "mean_match_rate": float(np.mean(fold_match_rates)) if fold_match_rates else float("nan"),
                        "std_match_rate": float(np.std(fold_match_rates)) if fold_match_rates else float("nan"),
                        "paper": TABLE5_PAPER[blackbox][robust_type][tol][dataset_name],
                        "folds": fold_details,
                    }
                    dataset_metrics["delta"] = _table5_metric_deltas(dataset_metrics)
                    results[blackbox][robust_type][tol][dataset_name] = dataset_metrics
    return results


def _print_table4(results: dict[str, object]) -> None:
    print("Table 4 (CoGS row only)")
    for blackbox, blackbox_results in results.items():
        print(f"[{blackbox}]")
        for dataset_name in DATASET_ORDER:
            if dataset_name not in blackbox_results:
                continue
            metrics = blackbox_results[dataset_name]
            print(
                f"  {DATASET_ALIASES[dataset_name]}: "
                f"{metrics['mean_success_rate']:.2f} +- {metrics['std_success_rate']:.2f} "
                f"(paper {metrics['paper']['mean']:.2f} +- {metrics['paper']['std']:.2f})"
            )


def _print_table5(results: dict[str, object]) -> None:
    print("Table 5")
    for blackbox, blackbox_results in results.items():
        print(f"[{blackbox}]")
        for robust_type in ["C", "K", "CK"]:
            print(f"  {robust_type}")
            for tol in [0.01, 0.05, 0.10]:
                row = [f"    tol={int(tol * 100)}%"]
                for dataset_name in DATASET_ORDER:
                    if dataset_name not in blackbox_results[robust_type][tol]:
                        continue
                    metrics = blackbox_results[robust_type][tol][dataset_name]
                    paper_mean, paper_std = metrics["paper"]
                    row.append(
                        f"{DATASET_ALIASES[dataset_name]} "
                        f"{metrics['mean_match_rate']:.2f} +- {metrics['std_match_rate']:.2f} "
                        f"(paper {paper_mean:.2f} +- {paper_std:.2f})"
                    )
                print(" | ".join(row))


def _write_report(args: argparse.Namespace, table4: dict[str, object] | None, table5: dict[str, object] | None) -> Path:
    experiments_data: dict[str, dict[str, object]] = {}

    if table4 is not None:
        for blackbox, blackbox_results in table4.items():
            for dataset_name, dataset_metrics in blackbox_results.items():
                paper_metrics = dataset_metrics["paper"]
                experiments_data[f"table4_{blackbox}_{dataset_name}"] = {
                    "configuration": {
                        "table": "table4",
                        "blackbox": blackbox,
                        "dataset": dataset_name,
                        "fold_count": len(dataset_metrics.get("folds", [])),
                    },
                    "metrics": {
                        "mean_success_rate": {
                            "original": paper_metrics["mean"],
                            "reproduced": dataset_metrics["mean_success_rate"],
                        },
                        "std_success_rate": {
                            "original": paper_metrics["std"],
                            "reproduced": dataset_metrics["std_success_rate"],
                        },
                        "mean_test_accuracy": {
                            "original": None,
                            "reproduced": dataset_metrics["mean_test_accuracy"],
                        },
                    },
                }

    if table5 is not None:
        for blackbox, blackbox_results in table5.items():
            for robust_type, robust_results in blackbox_results.items():
                for tolerance, tolerance_results in robust_results.items():
                    for dataset_name, dataset_metrics in tolerance_results.items():
                        paper_mean, paper_std = dataset_metrics["paper"]
                        experiments_data[
                            f"table5_{blackbox}_{robust_type}_{_slugify(tolerance)}_{dataset_name}"
                        ] = {
                            "configuration": {
                                "table": "table5",
                                "blackbox": blackbox,
                                "robust_type": robust_type,
                                "tolerance": float(tolerance),
                                "dataset": dataset_name,
                                "fold_count": len(dataset_metrics.get("folds", [])),
                            },
                            "metrics": {
                                "mean_match_rate": {
                                    "original": paper_mean,
                                    "reproduced": dataset_metrics["mean_match_rate"],
                                },
                                "std_match_rate": {
                                    "original": paper_std,
                                    "reproduced": dataset_metrics["std_match_rate"],
                                },
                            },
                        }

    return write_reproduction_report(
        output_path=Path(args.report_path),
        paper_id="cogs_tables_4_5",
        reproduction_metadata={
            "timestamp": datetime.now(timezone.utc),
            "framework_version": "1.0.0",
            "source_script": Path(__file__).name,
            "arguments": vars(args),
            "tables_included": [
                table_name
                for table_name, payload in (("table4", table4), ("table5", table5))
                if payload is not None
            ],
        },
        experiments_data=experiments_data,
    )


@pytest.mark.slow
def test_reproduce() -> None:
    args = _parse_args()
    table4 = None
    table5 = None
    if args.mode in {"generate", "all"}:
        _generate_artifacts(args)
    if args.mode in {"analyze", "all"} and args.table in {"table4", "all"}:
        table4 = _run_table4(args)
        _print_table4(table4)
    if args.mode in {"analyze", "all"} and args.table in {"table5", "all"}:
        table5 = _run_table5(args)
        _print_table5(table5)
    if args.mode in {"analyze", "all"}:
        report_path = _write_report(args, table4=table4, table5=table5)
        print(f"reproduction_report_path: {report_path}")
        debug_dump = {
            "table4": table4,
            "table5": table5,
        }
        print("debug_json:")
        print(json.dumps(debug_dump, indent=2, default=str))


if __name__ == "__main__":
    test_reproduce()
