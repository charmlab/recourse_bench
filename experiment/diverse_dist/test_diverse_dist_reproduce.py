from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

import pytest

os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("TEMP", "/tmp")
os.environ.setdefault("TMP", "/tmp")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import dataset  # noqa: F401
import method  # noqa: F401
import model  # noqa: F401
import preprocess  # noqa: F401
from dataset.dataset_object import DatasetObject
from experiment.utils import write_reproduction_report
from method.diverse_dist.support import DiverseDistModelAdapter, DiverseDistTrace
from utils.logger import setup_logger
from utils.registry import get_registry

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
REPORT_PATH = Path(__file__).with_name("reproduction_report.json")
LOCAL_DATA_ROOT = Path(__file__).with_name("data")


@dataclass(frozen=True)
class Condition:
    norm: int
    opt: bool


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    loader_kind: str
    reference_csv: Path | None
    method_overrides: dict[str, object]
    paper_targets: dict[Condition, dict[str, float]]


@dataclass
class DatasetRunResult:
    dataset_name: str
    feature_count: int
    test_accuracy: float
    observed_results: dict[Condition, dict[str, float | int]]


class ReferenceCsvDataset(DatasetObject):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        name: str,
        target_column: str,
        raw_feature_type: dict[str, str] | None = None,
        raw_feature_mutability: dict[str, bool] | None = None,
        raw_feature_actionability: dict[str, str] | None = None,
        **kwargs,
    ):
        self._rawdf = df.copy(deep=True)
        self._freeze = False
        self.name = name
        self.target_column = target_column
        self.feature_order = list(self._rawdf.columns)
        self.raw_feature_type = raw_feature_type or {
            column: "binary" if column == target_column else "numerical"
            for column in self._rawdf.columns
        }
        self.raw_feature_mutability = raw_feature_mutability or {
            column: column != target_column for column in self._rawdf.columns
        }
        self.raw_feature_actionability = raw_feature_actionability or {
            column: "none" if column == target_column else "any"
            for column in self._rawdf.columns
        }

    def _read_df(self, path: str) -> pd.DataFrame:
        raise NotImplementedError("ReferenceCsvDataset is constructed from a DataFrame")


PAPER_DATASET_SPECS: dict[str, DatasetSpec] = {
    "diabetes": DatasetSpec(
        name="diabetes",
        display_name="diabetes",
        loader_kind="benchmark",
        reference_csv=None,
        method_overrides={},
        paper_targets={
            Condition(norm=1, opt=True): {
                "validity": 100.0,
                "k_distance_mean": 1.1356122951450993,
                "k_distance_std": 0.4345114915961585,
                "k_diversity_mean": 1.3891334020759285,
                "k_diversity_std": 0.46404387689018917,
                "set_distance_sum_mean": 0.21001440954026088,
                "set_distance_sum_std": 0.20520751186424946,
                "set_distance_max_mean": 0.5193340607256643,
                "set_distance_max_std": 0.438927379640319,
                "time_mean": 0.02246472391031556,
                "time_std": 0.004527869777919189,
            },
            Condition(norm=2, opt=True): {
                "validity": 100.0,
                "k_distance_mean": 0.521379531942026,
                "k_distance_std": 0.2013895759724448,
                "k_diversity_mean": 0.6299450517127192,
                "k_diversity_std": 0.21781486904831945,
                "set_distance_sum_mean": 0.0966081962615141,
                "set_distance_sum_std": 0.09563006446337553,
                "set_distance_max_mean": 0.23849450500831512,
                "set_distance_max_std": 0.20251878656648942,
                "time_mean": 0.02061738806255793,
                "time_std": 0.004091528813477416,
            },
            Condition(norm=1, opt=False): {
                "validity": 100.0,
                "k_distance_mean": 1.389182553084757,
                "k_distance_std": 0.291344368462992,
                "k_diversity_mean": 1.7118196467708162,
                "k_diversity_std": 0.30187489393115613,
                "set_distance_sum_mean": 0.22359984183146248,
                "set_distance_sum_std": 0.244943426702313,
                "set_distance_max_mean": 0.6324752906912724,
                "set_distance_max_std": 0.5683733757450508,
                "time_mean": 0.010467921273183014,
                "time_std": 0.001495682777856654,
            },
            Condition(norm=2, opt=False): {
                "validity": 100.0,
                "k_distance_mean": 0.6431523377632499,
                "k_distance_std": 0.14193282557234704,
                "k_diversity_mean": 0.7818753589678238,
                "k_diversity_std": 0.15164176344982927,
                "set_distance_sum_mean": 0.10433898230740925,
                "set_distance_sum_std": 0.11493570931665867,
                "set_distance_max_mean": 0.2958539824632799,
                "set_distance_max_std": 0.26735974736193696,
                "time_mean": 0.010600069821891139,
                "time_std": 0.001600492824113005,
            },
        },
    ),
    "no2": DatasetSpec(
        name="no2",
        display_name="no2",
        loader_kind="reference_csv",
        reference_csv=LOCAL_DATA_ROOT / "no2.csv",
        method_overrides={},
        paper_targets={
            Condition(norm=1, opt=True): {
                "validity": 100.0,
                "k_distance_mean": 0.6195697341733506,
                "k_distance_std": 0.23376784661693936,
                "k_diversity_mean": 0.7877654376535028,
                "k_diversity_std": 0.2842564609400587,
                "set_distance_sum_mean": 0.16236953732885087,
                "set_distance_sum_std": 0.1250212304713393,
                "set_distance_max_mean": 0.335030530077427,
                "set_distance_max_std": 0.23795109461626193,
                "time_mean": 0.018551729343555593,
                "time_std": 0.014841148530193417,
            },
            Condition(norm=2, opt=True): {
                "validity": 100.0,
                "k_distance_mean": 0.3048553725583726,
                "k_distance_std": 0.11443811074656235,
                "k_diversity_mean": 0.38542409662069654,
                "k_diversity_std": 0.1401568646680589,
                "set_distance_sum_mean": 0.07740982157846708,
                "set_distance_sum_std": 0.05945554812100612,
                "set_distance_max_mean": 0.16005675948724732,
                "set_distance_max_std": 0.11401052639165324,
                "time_mean": 0.018258037390532316,
                "time_std": 0.014400465978400282,
            },
            Condition(norm=1, opt=False): {
                "validity": 100.0,
                "k_distance_mean": 0.867544349972541,
                "k_distance_std": 0.2069646610069502,
                "k_diversity_mean": 1.1078796183253683,
                "k_diversity_std": 0.23683152035958221,
                "set_distance_sum_mean": 0.15070948545722446,
                "set_distance_sum_std": 0.1636990764999631,
                "set_distance_max_mean": 0.389747858146783,
                "set_distance_max_std": 0.3532081463443451,
                "time_mean": 0.009372600802668819,
                "time_std": 0.0014526652762984562,
            },
            Condition(norm=2, opt=False): {
                "validity": 100.0,
                "k_distance_mean": 0.4259955778213827,
                "k_distance_std": 0.09472083734072355,
                "k_diversity_mean": 0.5351792592515182,
                "k_diversity_std": 0.1091282589192789,
                "set_distance_sum_mean": 0.07375574688240437,
                "set_distance_sum_std": 0.08024609148471444,
                "set_distance_max_mean": 0.18741673907707035,
                "set_distance_max_std": 0.1690608353521021,
                "time_mean": 0.009374009238349067,
                "time_std": 0.0014850562213812887,
            },
        },
    ),
    "news": DatasetSpec(
        name="news",
        display_name="news",
        loader_kind="reference_csv",
        reference_csv=LOCAL_DATA_ROOT / "news.csv",
        method_overrides={"alpha": 1000},
        paper_targets={
            Condition(norm=1, opt=True): {
                "validity": 100.0,
                "k_distance_mean": 2.7082812984139935,
                "k_distance_std": 0.9792371909063994,
                "k_diversity_mean": 3.4528531924520034,
                "k_diversity_std": 1.2457051074035783,
                "set_distance_sum_mean": 0.8827430525206145,
                "set_distance_sum_std": 0.7854665964315622,
                "set_distance_max_mean": 1.9481909303121976,
                "set_distance_max_std": 1.4079245606743451,
                "time_mean": 0.30924982844658616,
                "time_std": 0.008624294788833095,
            },
            Condition(norm=2, opt=True): {
                "validity": 100.0,
                "k_distance_mean": 0.753167686377201,
                "k_distance_std": 0.2773105381613848,
                "k_diversity_mean": 0.9455304743790431,
                "k_diversity_std": 0.35630206825583965,
                "set_distance_sum_mean": 0.21922616960121646,
                "set_distance_sum_std": 0.21329538887315122,
                "set_distance_max_mean": 0.5230442320759029,
                "set_distance_max_std": 0.43612427243410695,
                "time_mean": 0.29450775542349183,
                "time_std": 0.004744213758779755,
            },
            Condition(norm=1, opt=False): {
                "validity": 100.0,
                "k_distance_mean": 3.530076156957467,
                "k_distance_std": 0.9028430904918028,
                "k_diversity_mean": 4.355430504464352,
                "k_diversity_std": 1.179512381009224,
                "set_distance_sum_mean": 0.7947714878410156,
                "set_distance_sum_std": 0.9701452060470587,
                "set_distance_max_mean": 2.1449439565635027,
                "set_distance_max_std": 1.8755567437282197,
                "time_mean": 0.2825630655828512,
                "time_std": 0.010090933446743526,
            },
            Condition(norm=2, opt=False): {
                "validity": 100.0,
                "k_distance_mean": 0.9826232194767097,
                "k_distance_std": 0.26502041297129036,
                "k_diversity_mean": 1.199459429323792,
                "k_diversity_std": 0.3440401847367622,
                "set_distance_sum_mean": 0.2206396335817396,
                "set_distance_sum_std": 0.26820636713781054,
                "set_distance_max_mean": 0.6147140340813039,
                "set_distance_max_std": 0.5619741369655994,
                "time_mean": 0.27688173977833874,
                "time_std": 0.004744227960527341,
            },
        },
    ),
}


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Reproduction config must parse to a dictionary")
    return config


def _build_object(registry_type: str, config: dict, **kwargs):
    registry = get_registry(registry_type)
    object_config = deepcopy(config)
    name = object_config.pop("name")
    cls = registry[name]
    return cls(**object_config, **kwargs)


def _materialize_registered_dataset(config: dict):
    dataset_obj = _build_object("dataset", config["dataset"])
    for preprocess_config in config.get("preprocess", []):
        preprocess_step = _build_object("preprocess", preprocess_config)
        dataset_obj = preprocess_step.transform(dataset_obj)
    finalize = get_registry("preprocess")["finalize"]()
    return finalize.transform(dataset_obj)


def _build_reference_dataset(spec: DatasetSpec):
    if spec.reference_csv is None:
        raise ValueError(f"Dataset {spec.name} does not have a reference CSV")

    df = pd.read_csv(spec.reference_csv).dropna().reset_index(drop=True)
    if spec.name == "no2":
        df = df.replace(to_replace={"N": 0, "P": 1})

    target_column = "Outcome"
    feature_columns = [column for column in df.columns if column != target_column]
    min_vals = df.loc[:, feature_columns].min(axis=0)
    max_vals = df.loc[:, feature_columns].max(axis=0)
    denominators = (max_vals - min_vals).replace(0, 1.0)

    scaled_df = df.copy(deep=True)
    for column in feature_columns:
        scaled_df[column] = (
            scaled_df[column].astype("float64") - float(min_vals[column])
        ) / float(denominators[column])
    scaled_df.loc[:, target_column] = scaled_df[target_column].astype(int)

    dataset_obj = ReferenceCsvDataset(
        scaled_df,
        name=spec.name,
        target_column=target_column,
    )
    dataset_obj.freeze()
    return dataset_obj


def _clone_with_df(dataset_obj, df: pd.DataFrame, flag: str):
    cloned = dataset_obj.clone()
    cloned.update(flag, True, df=df.copy(deep=True))
    cloned.freeze()
    return cloned


def _split_dataset(dataset_obj, split_seed: int, test_size: float):
    feature_df = dataset_obj.get(target=False)
    target_df = dataset_obj.get(target=True)
    target_series = target_df.iloc[:, 0]
    combined = pd.concat([feature_df, target_df], axis=1)

    train_df, test_df = train_test_split(
        combined,
        test_size=float(test_size),
        random_state=int(split_seed),
        shuffle=True,
        stratify=target_series,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return (
        _clone_with_df(dataset_obj, train_df, "trainset"),
        _clone_with_df(dataset_obj, test_df, "testset"),
    )


def _apply_noise(
    x: np.ndarray,
    train_min: np.ndarray,
    train_max: np.ndarray,
    noise_percent: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    variance = float(noise_percent) * (train_max - train_min) / 100.0
    covariance = np.diag(variance.astype(np.float64, copy=False))
    noisy = rng.multivariate_normal(x.astype(np.float64), covariance)
    return noisy.clip(min=train_min, max=train_max).astype(np.float32, copy=False)


def _compute_distance(x: np.ndarray, y: np.ndarray, norm: int) -> float:
    if int(norm) == 1:
        return float(np.sum(np.abs(x - y)))
    if int(norm) == 2:
        return float(np.linalg.norm(x - y, ord=2))
    raise ValueError("norm must be 1 or 2")


def _compute_k_distance(
    factual: np.ndarray,
    counterfactuals: list[np.ndarray],
    norm: int,
) -> float:
    return float(
        sum(
            _compute_distance(factual, counterfactual, norm)
            for counterfactual in counterfactuals
        )
        / len(counterfactuals)
    )


def _compute_k_diversity(counterfactuals: list[np.ndarray], norm: int) -> float:
    if len(counterfactuals) < 2:
        return 0.0

    values: list[float] = []
    for left_index in range(len(counterfactuals)):
        for right_index in range(left_index + 1, len(counterfactuals)):
            values.append(
                _compute_distance(
                    counterfactuals[left_index],
                    counterfactuals[right_index],
                    norm,
                )
            )
    return float(sum(values) / len(values))


def _compute_set_distance(
    counterfactuals: list[np.ndarray],
    noisy_counterfactuals: list[np.ndarray],
    norm: int,
) -> float:
    forward = [
        min(
            _compute_distance(counterfactual, noisy_counterfactual, norm)
            for noisy_counterfactual in noisy_counterfactuals
        )
        for counterfactual in counterfactuals
    ]
    backward = [
        min(
            _compute_distance(noisy_counterfactual, counterfactual, norm)
            for counterfactual in counterfactuals
        )
        for noisy_counterfactual in noisy_counterfactuals
    ]
    return float(
        sum(forward) / (2 * len(counterfactuals))
        + sum(backward) / (2 * len(noisy_counterfactuals))
    )


def _compute_set_distance_max(
    counterfactuals: list[np.ndarray],
    noisy_counterfactuals: list[np.ndarray],
    norm: int,
) -> float:
    forward = [
        min(
            _compute_distance(counterfactual, noisy_counterfactual, norm)
            for noisy_counterfactual in noisy_counterfactuals
        )
        for counterfactual in counterfactuals
    ]
    backward = [
        min(
            _compute_distance(noisy_counterfactual, counterfactual, norm)
            for counterfactual in counterfactuals
        )
        for noisy_counterfactual in noisy_counterfactuals
    ]
    return float(0.5 * (max(forward) + max(backward)))


def _extract_counterfactual_set(method_obj) -> list[np.ndarray]:
    traces = getattr(method_obj, "_last_explanation_sets", [])
    if len(traces) != 1:
        return []

    trace = traces[0]
    if not isinstance(trace, DiverseDistTrace):
        return []

    return [
        np.asarray(counterfactual, dtype=np.float32).reshape(-1)
        for counterfactual in trace.counterfactuals
    ]


def _is_valid_counterfactual_set(
    adapter: DiverseDistModelAdapter,
    factual: np.ndarray,
    counterfactuals: list[np.ndarray],
) -> bool:
    if not counterfactuals:
        return False

    factual_label = int(adapter.predict_label_indices(factual.reshape(1, -1))[0])
    counterfactual_labels = adapter.predict_label_indices(
        np.asarray(counterfactuals, dtype=np.float32)
    )
    return bool(np.all(counterfactual_labels != factual_label))


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def _format_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "nan"
    return f"{mean:.2f} +/- {std:.2f}"


def _evaluate_condition(
    config: dict,
    trainset,
    testset,
    model_obj,
    feature_names: list[str],
    train_min: np.ndarray,
    train_max: np.ndarray,
    condition: Condition,
    logger,
    dataset_name: str,
) -> dict[str, float | int]:
    method_config = deepcopy(config["method"])
    method_config["norm"] = int(condition.norm)
    method_config["opt"] = bool(condition.opt)
    method_obj = _build_object("method", method_config, target_model=model_obj)
    method_obj.fit(trainset)

    adapter = DiverseDistModelAdapter(model_obj, feature_names)
    reproduction_config = config["reproduction"]
    start_index = int(reproduction_config["start_index"])
    num_inputs = int(reproduction_config["num_inputs"])
    repeat_times = int(reproduction_config["repeat_times"])
    noise_percent = float(reproduction_config["noise_percent"])
    noise_seed = int(reproduction_config["noise_seed"])
    show_progress = bool(reproduction_config.get("show_progress", True))

    test_features = testset.get(target=False).reset_index(drop=True)
    selected = test_features.iloc[start_index : start_index + num_inputs].copy(deep=True)
    rng = np.random.RandomState(noise_seed)

    results: list[dict[str, float]] = []
    skipped_noise_label_mismatch = 0
    eligible_runs = 0

    iterator = selected.iterrows()
    if show_progress:
        iterator = tqdm(
            iterator,
            total=selected.shape[0],
            desc=f"{dataset_name}-norm{condition.norm}-opt{int(condition.opt)}",
        )

    for _, factual_row in iterator:
        factual = factual_row.to_numpy(dtype=np.float32, copy=True)
        factual_label = int(adapter.predict_label_indices(factual.reshape(1, -1))[0])
        factual_df = pd.DataFrame([factual], columns=feature_names)

        for _ in range(repeat_times):
            factual_noisy = _apply_noise(
                x=factual,
                train_min=train_min,
                train_max=train_max,
                noise_percent=noise_percent,
                rng=rng,
            )
            noisy_label = int(
                adapter.predict_label_indices(factual_noisy.reshape(1, -1))[0]
            )
            if noisy_label != factual_label:
                skipped_noise_label_mismatch += 1
                continue

            eligible_runs += 1
            factual_noisy_df = pd.DataFrame([factual_noisy], columns=feature_names)

            start_time = time.perf_counter()
            method_obj.get_counterfactuals(factual_df)
            runtime = time.perf_counter() - start_time
            counterfactuals = _extract_counterfactual_set(method_obj)

            method_obj.get_counterfactuals(factual_noisy_df)
            noisy_counterfactuals = _extract_counterfactual_set(method_obj)

            if not counterfactuals or not noisy_counterfactuals:
                continue
            if not _is_valid_counterfactual_set(adapter, factual, counterfactuals):
                continue
            if not _is_valid_counterfactual_set(
                adapter,
                factual_noisy,
                noisy_counterfactuals,
            ):
                continue

            results.append(
                {
                    "k_distance": _compute_k_distance(
                        factual,
                        counterfactuals,
                        condition.norm,
                    ),
                    "k_diversity": _compute_k_diversity(
                        counterfactuals,
                        condition.norm,
                    ),
                    "set_distance_sum": _compute_set_distance(
                        counterfactuals,
                        noisy_counterfactuals,
                        condition.norm,
                    ),
                    "set_distance_max": _compute_set_distance_max(
                        counterfactuals,
                        noisy_counterfactuals,
                        condition.norm,
                    ),
                    "time": float(runtime),
                }
            )

    logger.warning(
        "Completed dataset=%s norm=%s opt=%s with %s successful runs, %s eligible runs, %s skipped noisy-label mismatches",
        dataset_name,
        condition.norm,
        condition.opt,
        len(results),
        eligible_runs,
        skipped_noise_label_mismatch,
    )

    validity = (
        100.0 * float(len(results)) / float(eligible_runs)
        if eligible_runs > 0
        else float("nan")
    )
    k_distance_mean, k_distance_std = _mean_std(
        [result["k_distance"] for result in results]
    )
    k_diversity_mean, k_diversity_std = _mean_std(
        [result["k_diversity"] for result in results]
    )
    set_distance_sum_mean, set_distance_sum_std = _mean_std(
        [result["set_distance_sum"] for result in results]
    )
    set_distance_max_mean, set_distance_max_std = _mean_std(
        [result["set_distance_max"] for result in results]
    )
    time_mean, time_std = _mean_std([result["time"] for result in results])

    return {
        "validity": validity,
        "k_distance_mean": k_distance_mean,
        "k_distance_std": k_distance_std,
        "k_diversity_mean": k_diversity_mean,
        "k_diversity_std": k_diversity_std,
        "set_distance_sum_mean": set_distance_sum_mean,
        "set_distance_sum_std": set_distance_sum_std,
        "set_distance_max_mean": set_distance_max_mean,
        "set_distance_max_std": set_distance_max_std,
        "time_mean": time_mean,
        "time_std": time_std,
        "successful_runs": len(results),
        "eligible_runs": eligible_runs,
        "skipped_noise_label_mismatch": skipped_noise_label_mismatch,
    }


def _build_comparison_row(
    condition: Condition,
    observed: dict[str, float | int],
    target_metrics: dict[str, float] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "opt": bool(condition.opt),
        "norm": int(condition.norm),
        "validity": f"{observed['validity']:.1f}%",
        "k_distance": _format_mean_std(
            float(observed["k_distance_mean"]),
            float(observed["k_distance_std"]),
        ),
        "k_diversity": _format_mean_std(
            float(observed["k_diversity_mean"]),
            float(observed["k_diversity_std"]),
        ),
        "set_d_sum": _format_mean_std(
            float(observed["set_distance_sum_mean"]),
            float(observed["set_distance_sum_std"]),
        ),
        "set_d_max": _format_mean_std(
            float(observed["set_distance_max_mean"]),
            float(observed["set_distance_max_std"]),
        ),
        "time": _format_mean_std(
            float(observed["time_mean"]),
            float(observed["time_std"]),
        ),
        "successful": int(observed["successful_runs"]),
        "eligible": int(observed["eligible_runs"]),
        "skipped_noise": int(observed["skipped_noise_label_mismatch"]),
    }

    if target_metrics is not None:
        row["target_validity"] = f"{target_metrics['validity']:.1f}%"
        row["target_k_distance"] = _format_mean_std(
            target_metrics["k_distance_mean"],
            target_metrics["k_distance_std"],
        )
        row["target_k_diversity"] = _format_mean_std(
            target_metrics["k_diversity_mean"],
            target_metrics["k_diversity_std"],
        )
        row["target_set_d_sum"] = _format_mean_std(
            target_metrics["set_distance_sum_mean"],
            target_metrics["set_distance_sum_std"],
        )
        row["target_set_d_max"] = _format_mean_std(
            target_metrics["set_distance_max_mean"],
            target_metrics["set_distance_max_std"],
        )
        row["target_time"] = _format_mean_std(
            target_metrics["time_mean"],
            target_metrics["time_std"],
        )

    return row


def _print_comparison(
    dataset_name: str,
    observed_results: dict[Condition, dict[str, float | int]],
    target_metrics: dict[Condition, dict[str, float]],
) -> None:
    rows = []
    for condition, observed in observed_results.items():
        rows.append(
            _build_comparison_row(
                condition=condition,
                observed=observed,
                target_metrics=target_metrics.get(condition),
            )
        )

    comparison = pd.DataFrame(rows)
    print(f"[{dataset_name}] Observed metrics vs paper targets")
    print(comparison.to_string(index=False))


def _resolve_dataset_names(args: argparse.Namespace, config: dict) -> list[str]:
    if args.dataset:
        if len(args.dataset) == 1 and args.dataset[0] == "all":
            return list(PAPER_DATASET_SPECS)
        return list(dict.fromkeys(args.dataset))

    configured = config.get("reproduction", {}).get("datasets")
    if configured:
        return [str(value) for value in configured]

    return list(PAPER_DATASET_SPECS)


def _build_dataset_config(base_config: dict, spec: DatasetSpec) -> dict:
    config = deepcopy(base_config)
    config["name"] = f"diverse_dist_{spec.name}_reproduction"
    config.setdefault("logger", {})
    config["logger"]["level"] = config["logger"].get("level", "WARNING")
    config["logger"]["path"] = None
    config["method"] = deepcopy(config["method"])
    config["method"].update(deepcopy(spec.method_overrides))

    if spec.loader_kind == "benchmark":
        config["dataset"] = {"name": spec.name}
    return config


def _materialize_dataset_for_spec(config: dict, spec: DatasetSpec):
    if spec.loader_kind == "benchmark":
        return _materialize_registered_dataset(config)
    if spec.loader_kind == "reference_csv":
        return _build_reference_dataset(spec)
    raise ValueError(f"Unsupported loader kind: {spec.loader_kind}")


def _run_single_dataset(
    base_config: dict,
    spec: DatasetSpec,
) -> DatasetRunResult:
    config = _build_dataset_config(base_config, spec)
    configured_device = str(config.get("model", {}).get("device", "cpu")).lower()
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested in config but is not available")

    logger = setup_logger(
        level=str(config.get("logger", {}).get("level", "WARNING")),
        path=None,
        name=config.get("name", f"diverse_dist_{spec.name}_reproduction"),
    )

    dataset_obj = _materialize_dataset_for_spec(config, spec)
    trainset, testset = _split_dataset(
        dataset_obj,
        split_seed=int(config["reproduction"]["split_seed"]),
        test_size=float(config["reproduction"]["test_size"]),
    )
    model_obj = _build_object("model", config["model"])
    model_obj.fit(trainset)

    train_features = trainset.get(target=False)
    train_array = train_features.to_numpy(dtype=np.float32)
    train_min = train_array.min(axis=0)
    train_max = train_array.max(axis=0)
    feature_names = list(train_features.columns)

    adapter = DiverseDistModelAdapter(model_obj, feature_names)
    test_features = testset.get(target=False)
    test_targets = testset.get(target=True).iloc[:, 0].to_numpy(dtype=np.int64)
    test_predictions = adapter.predict_label_indices(test_features)
    test_accuracy = float(np.mean(test_predictions == test_targets))

    logger.warning(
        "Running DiverseDist reproduction for dataset=%s on %s with test accuracy %.4f",
        spec.name,
        configured_device,
        test_accuracy,
    )

    observed_results: dict[Condition, dict[str, float | int]] = {}
    for opt in config["reproduction"]["opt_settings"]:
        for norm in config["reproduction"]["norms"]:
            condition = Condition(norm=int(norm), opt=bool(opt))
            logger.warning(
                "Running dataset=%s norm=%s opt=%s",
                spec.name,
                condition.norm,
                condition.opt,
            )
            observed_results[condition] = _evaluate_condition(
                config=config,
                trainset=trainset,
                testset=testset,
                model_obj=model_obj,
                feature_names=feature_names,
                train_min=train_min,
                train_max=train_max,
                condition=condition,
                logger=logger,
                dataset_name=spec.name,
            )

    print(f"[{spec.name}] Test accuracy: {test_accuracy:.4f}")
    _print_comparison(
        dataset_name=spec.name,
        observed_results=observed_results,
        target_metrics=spec.paper_targets,
    )

    return DatasetRunResult(
        dataset_name=spec.name,
        feature_count=len(feature_names),
        test_accuracy=test_accuracy,
        observed_results=observed_results,
    )


def _build_report_experiments(
    dataset_results: list[DatasetRunResult],
) -> dict[str, dict[str, object]]:
    experiments_data: dict[str, dict[str, object]] = {}
    for dataset_result in dataset_results:
        spec = PAPER_DATASET_SPECS[dataset_result.dataset_name]
        for condition, observed in dataset_result.observed_results.items():
            experiment_id = (
                f"{dataset_result.dataset_name}_norm_{condition.norm}_opt_{int(condition.opt)}"
            )
            experiments_data[experiment_id] = {
                "configuration": {
                    "dataset": dataset_result.dataset_name,
                    "feature_count": dataset_result.feature_count,
                    "norm": int(condition.norm),
                    "opt": bool(condition.opt),
                    "method_alpha": spec.method_overrides.get("alpha", 50),
                    "method_beta": spec.method_overrides.get("beta", 0.5),
                    "method_gamma": spec.method_overrides.get("gamma", 0.1),
                },
                "metrics": {
                    metric_name: {
                        "original": spec.paper_targets.get(condition, {}).get(metric_name),
                        "reproduced": metric_value,
                    }
                    for metric_name, metric_value in observed.items()
                },
            }
    return experiments_data


@pytest.mark.slow
def test_reproduce() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--dataset",
        action="append",
        choices=[*PAPER_DATASET_SPECS.keys(), "all"],
        default=None,
    )
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    base_config = _load_config(config_path)
    dataset_names = _resolve_dataset_names(args, base_config)

    dataset_results: list[DatasetRunResult] = []
    for dataset_name in dataset_names:
        dataset_results.append(
            _run_single_dataset(
                base_config=base_config,
                spec=PAPER_DATASET_SPECS[dataset_name],
            )
        )

    report_path = write_reproduction_report(
        output_path=REPORT_PATH,
        paper_id="diverse_dist_paper_reproduction",
        reproduction_metadata={
            "timestamp": datetime.now(timezone.utc),
            "framework_version": "1.0.0",
            "source_script": Path(__file__).name,
            "config_path": str(config_path),
            "datasets": dataset_names,
            "local_data_root": str(LOCAL_DATA_ROOT),
        },
        experiments_data=_build_report_experiments(dataset_results),
    )
    print(f"reproduction_report_path: {report_path}")


if __name__ == "__main__":
    test_reproduce()
