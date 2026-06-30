from __future__ import annotations

import argparse
import copy
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from gurobipy import GRB, GurobiError
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier

from dataset.dataset_object import DatasetObject
from dataset.diabetes.diabetes import DiabetesDataset
from method.apas.apas import ApasMethod
from method.apas.support import (
    BinaryNetwork,
    create_silent_gurobi_model,
    extract_binary_target_networks,
)
from model.mlp.mlp import MlpModel


warnings.filterwarnings("ignore")

REFERENCE_EXP_DIR = PROJECT_ROOT / "experiment" / "apas" / "dataset"


@dataclass(frozen=True)
class SklearnSpec:
    hidden_layer_sizes: int | tuple[int, ...]
    learning_rate_init: float
    batch_size: int
    max_iter: int
    random_state: int = 0
    learning_rate: str = "adaptive"
    activation: str = "relu"
    solver: str = "adam"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    target_column: str
    feature_columns: tuple[str, ...]
    continuous_features: tuple[str, ...]
    sklearn: SklearnSpec
    gap: float
    num_test_instances: int
    num_sound_instances: int
    split_seed: int
    d1_size: int | None
    d1_train_test_seed: int
    d1_train_test_split: float
    loader: Callable[[], dict[str, object]]
    targets: dict[str, dict[str, float]]


class NotebookDataset(DatasetObject):
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        continuous_features: list[str],
        name: str = "notebook",
        raw_feature_type: dict[str, str] | None = None,
        raw_feature_mutability: dict[str, bool] | None = None,
        raw_feature_actionability: dict[str, str] | None = None,
        encoding: dict[str, list[str]] | None = None,
        encoded_feature_type: dict[str, str] | None = None,
        encoded_feature_mutability: dict[str, bool] | None = None,
        encoded_feature_actionability: dict[str, str] | None = None,
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
            column: column != target_column for column in self.raw_feature_type
        }
        self.raw_feature_actionability = raw_feature_actionability or {
            column: "none" if column == target_column else "any"
            for column in self.raw_feature_type
        }
        if encoding is not None:
            self.encoding = encoding
        if encoded_feature_type is not None:
            self.encoded_feature_type = encoded_feature_type
        if encoded_feature_mutability is not None:
            self.encoded_feature_mutability = encoded_feature_mutability
        if encoded_feature_actionability is not None:
            self.encoded_feature_actionability = encoded_feature_actionability
        self.continuous_features = list(continuous_features)

    def _read_df(self, path: str) -> pd.DataFrame:
        raise NotImplementedError("NotebookDataset is constructed from a DataFrame")


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _resolve_device(device: str) -> str:
    device = str(device).lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Configured device 'cuda' is unavailable")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    return device


def _resolve_runtime_device(config: dict) -> str:
    model_device = _resolve_device(config["model"].get("device", "cpu"))
    method_device = _resolve_device(config["method"].get("device", model_device))
    if model_device != method_device:
        raise ValueError("model.device must match method.device")
    return model_device


def _min_max_scale(
    df: pd.DataFrame,
    feature_columns: list[str],
    continuous_features: list[str],
    min_source: pd.DataFrame | None = None,
    max_source: pd.DataFrame | None = None,
    reset_index: bool = True,
) -> pd.DataFrame:
    scaled_df = df.copy(deep=True)
    min_df = df if min_source is None else min_source
    max_df = df if max_source is None else max_source
    min_vals = min_df.loc[:, continuous_features].min(axis=0)
    max_vals = max_df.loc[:, continuous_features].max(axis=0)
    denominators = (max_vals - min_vals).replace(0, 1.0)
    for column in continuous_features:
        scaled_df[column] = (
            scaled_df[column].astype("float64") - float(min_vals[column])
        ) / float(denominators[column])
    for column in feature_columns:
        scaled_df[column] = scaled_df[column].astype("float64")
    if reset_index:
        return scaled_df.reset_index(drop=True)
    return scaled_df


def _make_frozen_dataset(
    df: pd.DataFrame,
    target_column: str,
    continuous_features: list[str],
    name: str,
    metadata: dict[str, object] | None = None,
) -> NotebookDataset:
    metadata = metadata or {}
    dataset = NotebookDataset(
        df=df,
        target_column=target_column,
        continuous_features=continuous_features,
        name=name,
        raw_feature_type=metadata.get("raw_feature_type"),
        raw_feature_mutability=metadata.get("raw_feature_mutability"),
        raw_feature_actionability=metadata.get("raw_feature_actionability"),
        encoding=metadata.get("encoding"),
        encoded_feature_type=metadata.get("encoded_feature_type"),
        encoded_feature_mutability=metadata.get("encoded_feature_mutability"),
        encoded_feature_actionability=metadata.get("encoded_feature_actionability"),
    )
    dataset.freeze()
    return dataset


CREDIT_COLUMNS = [
    "checking-status",
    "duration",
    "credit-history",
    "purpose",
    "amount",
    "savings",
    "employment",
    "rate",
    "sex-status",
    "guarantors",
    "residence",
    "property",
    "age",
    "installment",
    "housing",
    "num-credits",
    "job",
    "liable",
    "phone",
    "foreign",
    "good-credit",
]
CREDIT_ORDINAL_FEATURES = {
    "checking-status": 4,
    "savings": 5,
    "employment": 5,
    "rate": 4,
    "guarantors": 3,
    "residence": 4,
    "property": 4,
    "housing": 3,
    "num-credits": 4,
    "job": 4,
    "liable": 2,
}
CREDIT_DISCRETE_FEATURES = {
    "credit-history": 5,
    "purpose": 11,
    "sex-status": 4,
    "installment": 3,
    "phone": 2,
    "foreign": 2,
}
CREDIT_CONTINUOUS_FEATURES = ["duration", "amount", "age"]
CREDIT_FEATURE_ENCODINGS_OLD = {
    "checking-status": {"A14": 0, "A11": 1, "A12": 2, "A13": 3},
    "credit-history": {"A30": 2, "A31": 4, "A32": 3, "A33": 0, "A34": 1},
    "purpose": {
        "A410": 0,
        "A40": 1,
        "A41": 2,
        "A42": 3,
        "A43": 4,
        "A44": 5,
        "A45": 6,
        "A46": 7,
        "A47": 8,
        "A48": 9,
        "A49": 10,
    },
    "savings": {"A65": 0, "A61": 1, "A62": 2, "A63": 3, "A64": 4},
    "employment": {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4},
    "sex-status": {"A91": 0, "A92": 1, "A93": 1, "A94": 2, "A95": 3},
    "guarantors": {"A101": 0, "A102": 1, "A103": 2},
    "property": {"A121": 3, "A122": 2, "A123": 1, "A124": 0},
    "installment": {"A141": 0, "A142": 1, "A143": 2},
    "housing": {"A151": 1, "A152": 2, "A153": 0},
    "job": {"A171": 0, "A172": 1, "A173": 2, "A174": 3},
    "phone": {"A191": 0, "A192": 1},
    "foreign": {"A201": 0, "A202": 1},
    "good-credit": {1: 1, 2: 0},
}
CREDIT_TO_SUBTRACT1_OLD = ["rate", "residence", "num-credits", "liable"]
CREDIT_TO_SUBTRACT1_NEW = [
    "checking-status",
    "savings",
    "employment",
    "rate",
    "sex-status",
    "guarantors",
    "residence",
    "property",
    "installment",
    "housing",
    "num-credits",
    "job",
    "liable",
    "phone",
    "foreign",
]


def _encode_credit_df(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, list[int]], dict[str, list[str]]]:
    encoded_df = copy.copy(df)
    encoded_columns_order = list(CREDIT_COLUMNS)
    feature_var_map: dict[int, list[int]] = {}
    encoding: dict[str, list[str]] = {}

    for feature_index, column in enumerate(CREDIT_COLUMNS):
        if column in CREDIT_ORDINAL_FEATURES:
            encoded_columns = [
                f"{column}_therm_{index}"
                for index in range(CREDIT_ORDINAL_FEATURES[column])
            ]
            values = df[column].to_numpy(dtype=float)
            encoded_values = np.zeros((len(values), len(encoded_columns)), dtype=float)
            for row_index, value in enumerate(values):
                encoded_values[row_index, : int(value + 1)] = 1.0
            insert_at = encoded_df.columns.get_loc(column)
            start = insert_at
            front = encoded_df.loc[:, encoded_columns_order[:insert_at]]
            back = encoded_df.loc[:, encoded_columns_order[insert_at + 1 :]]
            encoded_df = pd.concat(
                [front, pd.DataFrame(encoded_values, columns=encoded_columns), back],
                axis=1,
            )
            feature_var_map[feature_index] = list(range(start, start + len(encoded_columns)))
            encoding[column] = encoded_columns
            encoded_columns_order = list(encoded_df.columns)
        elif column in CREDIT_DISCRETE_FEATURES:
            encoded_columns = [
                f"{column}_cat_{index}"
                for index in range(CREDIT_DISCRETE_FEATURES[column])
            ]
            values = df[column].to_numpy(dtype=int)
            encoded_values = np.zeros((len(values), len(encoded_columns)), dtype=float)
            for row_index, value in enumerate(values):
                encoded_values[row_index, int(value)] = 1.0
            insert_at = encoded_df.columns.get_loc(column)
            start = insert_at
            front = encoded_df.loc[:, encoded_columns_order[:insert_at]]
            back = encoded_df.loc[:, encoded_columns_order[insert_at + 1 :]]
            encoded_df = pd.concat(
                [front, pd.DataFrame(encoded_values, columns=encoded_columns), back],
                axis=1,
            )
            feature_var_map[feature_index] = list(range(start, start + len(encoded_columns)))
            encoding[column] = encoded_columns
            encoded_columns_order = list(encoded_df.columns)
        else:
            feature_var_map[feature_index] = [encoded_df.columns.get_loc(column)]
            if column != "good-credit":
                encoding[column] = [column]

    return encoded_df, feature_var_map, encoding


def _credit_metadata(
    feature_columns: list[str],
    encoding: dict[str, list[str]],
) -> dict[str, object]:
    raw_feature_type = {}
    for column in CREDIT_COLUMNS:
        if column == "good-credit":
            raw_feature_type[column] = "binary"
        elif column in CREDIT_CONTINUOUS_FEATURES:
            raw_feature_type[column] = "numerical"
        else:
            raw_feature_type[column] = "categorical"

    raw_feature_mutability = {
        column: column != "good-credit" for column in CREDIT_COLUMNS
    }
    raw_feature_actionability = {
        column: "none" if column == "good-credit" else "any"
        for column in CREDIT_COLUMNS
    }
    encoded_feature_type = {
        column: "numerical"
        if column in CREDIT_CONTINUOUS_FEATURES
        else "binary"
        for column in feature_columns
    }
    encoded_feature_mutability = {column: True for column in feature_columns}
    encoded_feature_actionability = {column: "any" for column in feature_columns}
    return {
        "raw_feature_type": raw_feature_type,
        "raw_feature_mutability": raw_feature_mutability,
        "raw_feature_actionability": raw_feature_actionability,
        "encoding": encoding,
        "encoded_feature_type": encoded_feature_type,
        "encoded_feature_mutability": encoded_feature_mutability,
        "encoded_feature_actionability": encoded_feature_actionability,
    }


def _load_diabetes_reference() -> dict[str, object]:
    dataset = DiabetesDataset(path="./dataset/diabetes/")
    df = dataset.snapshot().dropna().reset_index(drop=True)
    target_column = dataset.target_column
    feature_columns = [column for column in df.columns if column != target_column]
    scaled_df = _min_max_scale(
        df=df,
        feature_columns=feature_columns,
        continuous_features=feature_columns,
    )

    np.random.seed(1)
    
    d1_indices = np.sort(np.random.choice(range(len(scaled_df)), 384))
    d2_indices = np.array(
        [index for index in range(len(scaled_df)) if index not in d1_indices]
    )
    d1_df = pd.DataFrame(scaled_df.values[d1_indices], columns=scaled_df.columns)
    d2_df = pd.DataFrame(scaled_df.values[d2_indices], columns=scaled_df.columns)
    return {
        "scaled_df": scaled_df,
        "d1_df": d1_df,
        "d2_df": d2_df,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "continuous_features": feature_columns,
    }


def _load_no2_reference() -> dict[str, object]:
    df = pd.read_csv(REFERENCE_EXP_DIR / "no2.csv")
    df = df.dropna().replace(to_replace={"N": 0, "P": 1}).reset_index(drop=True)
    target_column = "binaryClass"
    feature_columns = [column for column in df.columns if column != target_column]
    scaled_df = _min_max_scale(
        df=df,
        feature_columns=feature_columns,
        continuous_features=feature_columns,
    )

    np.random.seed(4)
    d1_indices = np.sort(np.random.choice(range(500), 250))
    d2_indices = np.array([index for index in range(500) if index not in d1_indices])
    d1_df = pd.DataFrame(scaled_df.values[d1_indices], columns=scaled_df.columns)
    d2_df = pd.DataFrame(scaled_df.values[d2_indices], columns=scaled_df.columns)
    return {
        "scaled_df": scaled_df,
        "d1_df": d1_df,
        "d2_df": d2_df,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "continuous_features": feature_columns,
    }


def _load_sba_reference() -> dict[str, object]:
    df = pd.read_csv(REFERENCE_EXP_DIR / "SBAcase.11.13.17.csv")
    df = df.dropna(axis=1)
    df = df.drop(
        columns=[
            "ApprovalDate",
            "LoanNr_ChkDgt",
            "Name",
            "Zip",
            "City",
            "State",
            "NAICS",
            "FranchiseCode",
            "BalanceGross",
            "MIS_Status",
            "Selected",
            "UrbanRural",
            "Recession",
            "New",
            "RealEstate",
            "Portion",
        ]
    )
    continuous_features = [
        "Term",
        "NoEmp",
        "CreateJob",
        "RetainedJob",
        "DisbursementGross",
        "ChgOffPrinGr",
        "GrAppv",
        "SBA_Appv",
        "daysterm",
    ]
    target_column = "Default"
    d1_raw = df[df["ApprovalFY"] < 2006].drop(columns="ApprovalFY").reset_index(drop=True)
    d2_raw = df[df["ApprovalFY"] >= 2006].drop(columns="ApprovalFY").reset_index(drop=True)
    scale_source = df.drop(columns="ApprovalFY").reset_index(drop=True)
    feature_columns = [column for column in d1_raw.columns if column != target_column]

    d1_df = _min_max_scale(
        df=d1_raw,
        feature_columns=feature_columns,
        continuous_features=continuous_features,
        min_source=scale_source,
        max_source=scale_source,
    )
    d2_df = _min_max_scale(
        df=d2_raw,
        feature_columns=feature_columns,
        continuous_features=continuous_features,
        min_source=scale_source,
        max_source=scale_source,
    )
    d1_df.loc[:, target_column] = 1 - d1_df[target_column].astype(int)
    d2_df.loc[:, target_column] = 1 - d2_df[target_column].astype(int)
    scaled_df = pd.concat([d1_df, d2_df], ignore_index=True)
    return {
        "scaled_df": scaled_df,
        "d1_df": d1_df,
        "d2_df": d2_df,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "continuous_features": continuous_features,
    }


def _load_credit_reference() -> dict[str, object]:
    old_path = REFERENCE_EXP_DIR / "credit" / "old" / "german.data"
    new_path = REFERENCE_EXP_DIR / "credit" / "new" / "train.csv"

    old_df = pd.read_csv(old_path, header=None, delimiter=",").dropna()
    old_df.columns = CREDIT_COLUMNS
    old_df = old_df.replace(to_replace=CREDIT_FEATURE_ENCODINGS_OLD)
    for column in CREDIT_TO_SUBTRACT1_OLD:
        old_df[column] = old_df[column] - 1
    old_scaled_df = _min_max_scale(
        df=old_df,
        feature_columns=[column for column in CREDIT_COLUMNS if column != "good-credit"],
        continuous_features=CREDIT_CONTINUOUS_FEATURES,
        reset_index=False,
    )
    old_encoded_df, feature_var_map, encoding = _encode_credit_df(old_scaled_df)

    new_df = pd.read_csv(new_path, header=None).dropna()
    new_df = new_df.drop(columns=[0])
    new_df = new_df.drop(labels=0)
    new_df.columns = CREDIT_COLUMNS
    for column in new_df.columns:
        new_df[column] = new_df[column].astype(int)
    for column in CREDIT_TO_SUBTRACT1_NEW:
        new_df[column] = new_df[column] - 1
    new_scaled_df = _min_max_scale(
        df=new_df,
        feature_columns=[column for column in CREDIT_COLUMNS if column != "good-credit"],
        continuous_features=CREDIT_CONTINUOUS_FEATURES,
        reset_index=False,
    )
    new_encoded_df, _, _ = _encode_credit_df(new_scaled_df)
    new_encoded_df = new_encoded_df.dropna().reset_index(drop=True)

    target_column = "good-credit"
    feature_columns = [column for column in old_encoded_df.columns if column != target_column]
    metadata = _credit_metadata(feature_columns=feature_columns, encoding=encoding)
    scaled_df = pd.concat([old_encoded_df, new_encoded_df], ignore_index=True)
    return {
        "scaled_df": scaled_df,
        "d1_df": old_encoded_df,
        "d2_df": new_encoded_df,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "continuous_features": feature_columns,
        "metadata": metadata,
        "feature_var_map": feature_var_map,
    }


DATASET_SPECS: dict[str, DatasetSpec] = {
    "diabetes": DatasetSpec(
        name="diabetes",
        target_column="Outcome",
        feature_columns=(),
        continuous_features=(),
        sklearn=SklearnSpec(
            hidden_layer_sizes=8,
            learning_rate_init=0.01,
            batch_size=8,
            max_iter=7000,
        ),
        gap=0.25,
        num_test_instances=500,
        num_sound_instances=50,
        split_seed=1,
        d1_size=384,
        d1_train_test_seed=0,
        d1_train_test_split=0.2,
        loader=_load_diabetes_reference,
        targets={
            "notebook": {
                "sound_fraction": 0.27380952380952384,
                "sound_count": 69,
                "delta_max": 0.3457906217784634,
                "delta_min": 0.1058395646526038,
                "delta_e": 0.27983832,
                "found": 1.0,
                "vm1": 1.0,
                "vm2": 1.0,
                "delta_validity": 0.0,
                "l1": 0.07657846284575133,
                "l0": 0.3075,
                "lof": 1.0,
            },
            "paper": {
                "delta": 0.11,
                "delta_e": 0.27,
                "vm1": 1.0,
                "vm2": 1.0,
                "l1": 0.077,
                "lof": 1.0,
            },
            "full_results": {
                "full_results_milp_mean": 0.108742,
                "full_results_wilks_mean": 0.27409900000000004,
            },
        },
    ),
    "no2": DatasetSpec(
        name="no2",
        target_column="binaryClass",
        feature_columns=(),
        continuous_features=(),
        sklearn=SklearnSpec(
            hidden_layer_sizes=16,
            learning_rate_init=0.005,
            batch_size=8,
            max_iter=2000,
        ),
        gap=0.15,
        num_test_instances=200,
        num_sound_instances=50,
        split_seed=4,
        d1_size=250,
        d1_train_test_seed=0,
        d1_train_test_split=0.2,
        loader=_load_no2_reference,
        targets={
            "notebook": {
                "sound_fraction": 0.4424778761061947,
                "sound_count": 50,
                "delta_max": 0.11449453866860582,
                "delta_min": 0.020924421229148754,
                "delta_e": 0.07160799,
                "found": 1.0,
                "vm1": 1.0,
                "vm2": 1.0,
                "delta_validity": 0.0,
                "l1": 0.04218988369320217,
                "l0": 0.2116399999999999,
                "lof": 1.0,
            },
            "full_results": {
                "full_results_milp_mean": 0.022574999999999998,
                "full_results_wilks_mean": 0.076827,
            },
        },
    ),
    "sba": DatasetSpec(
        name="sba",
        target_column="Default",
        feature_columns=(),
        continuous_features=(),
        sklearn=SklearnSpec(
            hidden_layer_sizes=18,
            learning_rate_init=0.005,
            batch_size=8,
            max_iter=9000,
        ),
        gap=0.25,
        num_test_instances=200,
        num_sound_instances=50,
        split_seed=0,
        d1_size=None,
        d1_train_test_seed=5,
        d1_train_test_split=0.2,
        loader=_load_sba_reference,
        targets={
            "notebook": {
                "sound_fraction": 0.6348314606741573,
                "sound_count": 113,
                "delta_max": 0.30498064647513345,
                "delta_min": 0.10821238236127151,
                "delta_e": 0.24813461,
                "found": 1.0,
                "vm1": 1.0,
                "vm2": 1.0,
                "delta_validity": 0.0,
                "l1": 0.008920690895365926,
                "l0": 0.1642800000000001,
                "lof": 0.4,
            },
            "full_results": {
                "full_results_milp_mean": 0.10966000000000001,
                "full_results_wilks_mean": 0.317258,
            },
        },
    ),
    "credit": DatasetSpec(
        name="credit",
        target_column="good-credit",
        feature_columns=(),
        continuous_features=(),
        sklearn=SklearnSpec(
            hidden_layer_sizes=5,
            learning_rate_init=0.02,
            batch_size=32,
            max_iter=7000,
        ),
        gap=0.04,
        num_test_instances=2000,
        num_sound_instances=50,
        split_seed=0,
        d1_size=None,
        d1_train_test_seed=0,
        d1_train_test_split=0.2,
        loader=_load_credit_reference,
        targets={
            "notebook": {
                "sound_fraction": 0.30633802816901406,
                "sound_count": 87,
                "delta_max": 1.5372448287620384,
                "delta_min": 0.054449288758308775,
                "delta_e": 1.2818451,
                "found": 1.0,
                "vm1": 1.0,
                "vm2": 0.94,
                "delta_validity": 0.34,
                "l1": 0.028256931400971,
                "l0": 0.09299999999999993,
                "lof": 1.0,
            },
            "paper": {
                "delta": 0.054,
                "delta_e": 1.28,
                "vm1": 1.0,
                "vm2": 0.94,
                "l1": 0.028,
                "lof": 1.0,
            },
            "full_results": {
                "full_results_milp_mean": 0.06997500000000001,
                "full_results_wilks_mean": 0.20390900000000006,
            },
        },
    ),
}


def _build_reference_classifier(spec: SklearnSpec) -> MLPClassifier:
    return MLPClassifier(
        learning_rate=spec.learning_rate,
        hidden_layer_sizes=spec.hidden_layer_sizes,
        learning_rate_init=float(spec.learning_rate_init),
        batch_size=int(spec.batch_size),
        max_iter=int(spec.max_iter),
        random_state=int(spec.random_state),
        activation=spec.activation,
        solver=spec.solver,
    )


def _flatten_weights_and_biases(
    clf: MLPClassifier,
    include_weights: bool = True,
    include_biases: bool = True,
) -> np.ndarray:
    values: list[np.ndarray] = []
    if include_weights:
        values.extend(np.asarray(weight).reshape(-1) for weight in clf.coefs_)
    if include_biases:
        values.extend(np.asarray(bias).reshape(-1) for bias in clf.intercepts_)
    if not values:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate(values).astype(np.float64, copy=False)


def _inf_norm(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.max(np.abs(x - y)))


def _partial_fit_clone(
    clf: MLPClassifier,
    X_update: pd.DataFrame,
    y_update: pd.Series,
) -> MLPClassifier:
    clone = copy.deepcopy(clf)
    clone.partial_fit(
        X_update.to_numpy(dtype=np.float64),
        y_update.to_numpy(dtype=np.int64),
    )
    return clone


def _compute_delta_min(
    clf: MLPClassifier,
    X2: pd.DataFrame,
    y2: pd.Series,
    gap: float,
) -> float:
    reference = _flatten_weights_and_biases(clf)
    delta_min = -1.0
    subset_size = int(gap * len(X2))
    for seed in range(5):
        np.random.seed(seed)
        indices = np.random.choice(range(len(X2)), subset_size)
        shifted = _partial_fit_clone(clf, X2.iloc[indices], y2.iloc[indices])
        delta_min = max(
            delta_min,
            _inf_norm(reference, _flatten_weights_and_biases(shifted)),
        )
    return float(delta_min)


def _compute_delta_max_reference(
    clf: MLPClassifier,
    X2: pd.DataFrame,
    y2: pd.Series,
) -> tuple[float, MLPClassifier]:
    reference = _flatten_weights_and_biases(clf)
    delta_max = -1.0
    max_model: MLPClassifier | None = None
    subset_size = int(0.99 * len(X2))
    for seed in range(5):
        np.random.seed(seed)
        indices = np.random.choice(range(len(X2)), subset_size)
        shifted = _partial_fit_clone(clf, X2.iloc[indices], y2.iloc[indices])
        shifted_delta = _inf_norm(reference, _flatten_weights_and_biases(shifted))
        if shifted_delta >= delta_max:
            delta_max = shifted_delta
            max_model = shifted
    if max_model is None:
        raise RuntimeError("Could not compute reference delta_max model")
    return float(delta_max), max_model


def _compute_delta_e_notebook_style(
    clf: MLPClassifier,
    retrained_clf: MLPClassifier,
) -> float:
    difference = np.abs(
        np.asarray(clf.coefs_[0], dtype=np.float64)
        - np.asarray(retrained_clf.coefs_[0], dtype=np.float64)
    )
    return float(difference.max())


def _sklearn_to_benchmark_mlp(
    clf: MLPClassifier,
    layers: list[int],
    device: str,
) -> MlpModel:
    model = MlpModel(
        seed=int(getattr(clf, "random_state", 0) or 0),
        device=device,
        epochs=1,
        learning_rate=float(getattr(clf, "learning_rate_init", 0.01)),
        batch_size=int(getattr(clf, "batch_size", 8)),
        layers=layers,
        optimizer="adam",
        criterion="bce",
        output_activation="sigmoid",
        save_name=None,
    )

    model._class_to_index = {
        int(class_value): index for index, class_value in enumerate(clf.classes_.tolist())
    }
    model._output_dim = 1
    model._model = model._build_model(int(clf.n_features_in_), 1).to(model._device)

    linear_layers = [
        layer for layer in model._model.modules() if isinstance(layer, torch.nn.Linear)
    ]
    for linear_layer, weights, bias in zip(linear_layers, clf.coefs_, clf.intercepts_):
        linear_layer.weight.data = torch.tensor(
            np.asarray(weights, dtype=np.float32).T,
            dtype=torch.float32,
            device=model._device,
        )
        linear_layer.bias.data = torch.tensor(
            np.asarray(bias, dtype=np.float32),
            dtype=torch.float32,
            device=model._device,
        )

    model._model.eval()
    model._is_trained = True
    return model


def _predict_label_indices(model: MlpModel, X: pd.DataFrame) -> np.ndarray:
    prediction = model.get_prediction(X, proba=True)
    return prediction.detach().cpu().numpy().argmax(axis=1)


def _linear_expr(coefficients: np.ndarray, terms: list[float | object]):
    expression = 0.0
    for coefficient, term in zip(coefficients, terms):
        value = float(coefficient)
        if isinstance(term, (int, float, np.floating)):
            expression += value * float(term)
        else:
            expression += value * term
    return expression


def _compute_interval_lower_bound(
    network: BinaryNetwork,
    point: np.ndarray,
    delta: float,
    big_m: float,
    use_biases: bool,
    seed: int | None = None,
) -> float | None:
    bias_delta = float(delta) if use_biases else 0.0
    try:
        model = create_silent_gurobi_model("apas_interval_lower", seed=seed)
        previous_layer: list[float | object] = [float(value) for value in point.reshape(-1)]

        for layer_index, (weights, bias) in enumerate(
            zip(network.hidden_weights, network.hidden_biases)
        ):
            current_layer: list[object] = []
            for node_index in range(weights.shape[0]):
                node = model.addVar(
                    lb=0.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"n_{layer_index}_{node_index}",
                )
                active = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"a_{layer_index}_{node_index}",
                )
                ub_expr = _linear_expr(
                    weights[node_index] + float(delta),
                    previous_layer,
                ) + float(bias[node_index]) + bias_delta
                lb_expr = _linear_expr(
                    weights[node_index] - float(delta),
                    previous_layer,
                ) + float(bias[node_index]) - bias_delta

                model.addConstr(
                    node <= float(big_m) * (1 - active),
                    name=f"relu_upper_{layer_index}_{node_index}",
                )
                model.addConstr(
                    ub_expr + float(big_m) * active >= node,
                    name=f"relu_active_{layer_index}_{node_index}",
                )
                model.addConstr(
                    lb_expr <= node,
                    name=f"relu_lower_{layer_index}_{node_index}",
                )
                current_layer.append(node)
            previous_layer = current_layer

        output_score = model.addVar(
            lb=-GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name="output_score",
        )
        output_weights = network.output_weight[0]
        output_ub_expr = _linear_expr(
            output_weights + float(delta),
            previous_layer,
        ) + float(network.output_bias) + bias_delta
        output_lb_expr = _linear_expr(
            output_weights - float(delta),
            previous_layer,
        ) + float(network.output_bias) - bias_delta

        model.addConstr(output_score <= output_ub_expr, name="output_upper")
        model.addConstr(output_score >= output_lb_expr, name="output_lower")
        model.setObjective(output_score, GRB.MINIMIZE)
        model.optimize()
    except GurobiError:
        return None

    if getattr(model, "SolCount", 0) < 1:
        return None
    return float(output_score.X)


def _is_interval_robust(
    network: BinaryNetwork,
    point: np.ndarray,
    delta: float,
    big_m: float,
    use_biases: bool,
    seed: int | None = None,
) -> bool:
    lower_bound = _compute_interval_lower_bound(
        network=network,
        point=point,
        delta=delta,
        big_m=big_m,
        use_biases=use_biases,
        seed=seed,
    )
    return lower_bound is not None and lower_bound >= 0.0


def _select_reference_test_instances(
    base_model: MlpModel,
    X1: pd.DataFrame,
    desired_class: int,
    num_test_instances: int,
) -> pd.DataFrame:
    np.random.seed(1)
    predictions = _predict_label_indices(base_model, X1)
    candidate_indices = np.where(predictions == (1 - desired_class))[0]
    sampled_indices = np.random.choice(
        candidate_indices,
        min(num_test_instances, len(candidate_indices)),
    )
    return pd.DataFrame(X1.values[sampled_indices], columns=X1.columns)


def _verify_soundness(
    base_model: MlpModel,
    candidate_factuals: pd.DataFrame,
    delta_min: float,
    big_m: float,
    use_biases: bool,
) -> tuple[float, list[int]]:
    target_networks = extract_binary_target_networks(base_model)
    predictions = _predict_label_indices(base_model, candidate_factuals)

    valid_positions: list[int] = []
    for position, (_, row) in enumerate(candidate_factuals.iterrows()):
        predicted_class = int(predictions[position])
        network = target_networks[predicted_class]
        if _is_interval_robust(
            network=network,
            point=row.to_numpy(dtype=np.float64),
            delta=delta_min,
            big_m=big_m,
            use_biases=use_biases,
            seed=position,
        ):
            valid_positions.append(position)

    sound_fraction = len(valid_positions) / float(max(len(candidate_factuals), 1))
    return sound_fraction, valid_positions


def _normalised_l1_all(counterfactual: np.ndarray, factual: np.ndarray) -> float:
    return float(np.sum(np.abs(counterfactual - factual)) / counterfactual.shape[0])


def _normalised_l0(
    counterfactual: np.ndarray,
    factual: np.ndarray,
    feature_groups: list[list[int]] | None = None,
) -> float:
    if feature_groups is None:
        return float(np.mean(np.abs(counterfactual - factual) >= 1e-4))

    changed = 0
    for group in feature_groups:
        group_indices = np.asarray(group, dtype=int)
        if np.abs(
            np.sum(counterfactual[group_indices]) - np.sum(factual[group_indices])
        ) >= 1e-4:
            changed += 1
    return float(changed / max(len(feature_groups), 1))


def _evaluate_counterfactuals(
    factuals: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    base_model: MlpModel,
    retrained_model: MlpModel,
    delta_min: float,
    desired_class: int,
    big_m: float,
    use_biases: bool,
    lof_model: LocalOutlierFactor,
    feature_groups: list[list[int]] | None = None,
) -> dict[str, float]:
    target_networks = extract_binary_target_networks(base_model)
    original_predictions = _predict_label_indices(base_model, factuals)

    found = 0
    vm1 = 0
    delta_validity = 0
    vm2 = 0
    l1_sum = 0.0
    l0_sum = 0.0
    lof_sum = 0.0

    for row_index in range(factuals.shape[0]):
        factual = factuals.iloc[row_index].to_numpy(dtype=np.float64)
        counterfactual = counterfactuals.iloc[row_index]

        if counterfactual.isna().any():
            continue

        found += 1
        counterfactual_array = counterfactual.to_numpy(dtype=np.float64)
        counterfactual_df = pd.DataFrame(
            [counterfactual_array], columns=factuals.columns
        )
        cf_prediction = int(_predict_label_indices(base_model, counterfactual_df)[0])
        original_prediction = int(original_predictions[row_index])

        if _is_interval_robust(
            network=target_networks[desired_class],
            point=counterfactual_array,
            delta=delta_min,
            big_m=big_m,
            use_biases=use_biases,
            seed=10_000 + row_index,
        ):
            delta_validity += 1

        if cf_prediction != original_prediction:
            vm1 += 1
            l1_sum += _normalised_l1_all(counterfactual_array, factual)
            l0_sum += _normalised_l0(
                counterfactual_array,
                factual,
                feature_groups=feature_groups,
            )
            lof_sum += float(lof_model.predict(counterfactual_array.reshape(1, -1))[0])

            retrained_prediction = int(
                _predict_label_indices(retrained_model, counterfactual_df)[0]
            )
            if retrained_prediction != original_prediction:
                vm2 += 1

    denominator = float(max(factuals.shape[0], 1))
    valid_denominator = float(max(vm1, 1))
    return {
        "found": found / denominator,
        "vm1": vm1 / denominator,
        "vm2": vm2 / denominator,
        "delta_validity": delta_validity / denominator,
        "l1": l1_sum / valid_denominator if vm1 > 0 else float("nan"),
        "l0": l0_sum / valid_denominator if vm1 > 0 else float("nan"),
        "lof": lof_sum / valid_denominator if vm1 > 0 else float("nan"),
        "num_found": float(found),
        "num_valid": float(vm1),
    }


def _resolve_layers(hidden_layer_sizes: int | tuple[int, ...]) -> list[int]:
    if isinstance(hidden_layer_sizes, tuple):
        return [int(value) for value in hidden_layer_sizes]
    return [int(hidden_layer_sizes)]


def _prepare_datasets(spec: DatasetSpec) -> dict[str, object]:
    loaded = spec.loader()
    feature_columns = list(loaded["feature_columns"])
    target_column = str(loaded["target_column"])
    d1_df = loaded["d1_df"].copy(deep=True).reset_index(drop=True)
    d2_df = loaded["d2_df"].copy(deep=True).reset_index(drop=True)
    scaled_df = loaded["scaled_df"].copy(deep=True).reset_index(drop=True)
    continuous_features = list(loaded["continuous_features"])
    metadata = loaded.get("metadata", {})
    feature_var_map = loaded.get("feature_var_map")

    X1 = d1_df.loc[:, feature_columns].copy(deep=True)
    y1 = d1_df.loc[:, target_column].astype(int).copy(deep=True)
    X2 = d2_df.loc[:, feature_columns].copy(deep=True)
    y2 = d2_df.loc[:, target_column].astype(int).copy(deep=True)

    X1_train, _, y1_train, _ = train_test_split(
        X1,
        y1,
        stratify=y1,
        test_size=float(spec.d1_train_test_split),
        shuffle=True,
        random_state=int(spec.d1_train_test_seed),
    )

    full_dataset = _make_frozen_dataset(
        df=scaled_df,
        target_column=target_column,
        continuous_features=continuous_features,
        name=spec.name,
        metadata=metadata,
    )
    feature_groups = None
    if isinstance(feature_var_map, dict):
        feature_groups = [
            list(feature_var_map[index])
            for index in sorted(feature_var_map)
            if index < len(CREDIT_COLUMNS) - 1
        ]
    return {
        "full_dataset": full_dataset,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "feature_groups": feature_groups,
        "X1": X1.reset_index(drop=True),
        "y1": y1.reset_index(drop=True),
        "X2": X2.reset_index(drop=True),
        "y2": y2.reset_index(drop=True),
        "X1_train": X1_train.reset_index(drop=True),
        "y1_train": y1_train.reset_index(drop=True),
    }


def _compare_against_targets(
    results: dict[str, float],
    targets: dict[str, float],
) -> list[tuple[str, float, float, float]]:
    rows: list[tuple[str, float, float, float]] = []
    for key, target_value in targets.items():
        if key not in results:
            continue
        reproduced = float(results[key])
        target = float(target_value)
        rows.append((key, target, reproduced, abs(reproduced - target)))
    return rows


def _method_config(config: dict, device: str, delta_min: float) -> dict[str, object]:
    method_cfg = copy.deepcopy(config.get("method", {}))
    method_cfg.pop("name", None)
    method_cfg["device"] = device
    method_cfg["desired_class"] = int(method_cfg.get("desired_class", 1))

    delta_source = str(config.get("reproduction", {}).get("delta_source", "delta_min")).lower()
    if delta_source == "delta_min":
        method_cfg["delta"] = float(delta_min)
    elif method_cfg.get("delta") is None:
        raise ValueError("method.delta must be set when reproduction.delta_source is not 'delta_min'")
    return method_cfg


def run_dataset_reproduction(
    spec: DatasetSpec,
    config: dict,
    limit_sound: int | None = None,
) -> dict[str, object]:
    device = _resolve_runtime_device(config)
    datasets = _prepare_datasets(spec)

    sklearn_base = _build_reference_classifier(spec.sklearn)
    sklearn_base.fit(
        datasets["X1_train"].to_numpy(dtype=np.float64),
        datasets["y1_train"].to_numpy(dtype=np.int64),
    )

    delta_min = _compute_delta_min(
        clf=sklearn_base,
        X2=datasets["X2"],
        y2=datasets["y2"],
        gap=float(spec.gap),
    )
    delta_max_reference, max_shifted_sklearn = _compute_delta_max_reference(
        clf=sklearn_base,
        X2=datasets["X2"],
        y2=datasets["y2"],
    )
    sklearn_retrained = _partial_fit_clone(
        sklearn_base,
        datasets["X2"],
        datasets["y2"],
    )
    delta_e = _compute_delta_e_notebook_style(sklearn_base, sklearn_retrained)

    layers = _resolve_layers(spec.sklearn.hidden_layer_sizes)
    base_model = _sklearn_to_benchmark_mlp(
        clf=sklearn_base,
        layers=layers,
        device=device,
    )
    retrained_model = _sklearn_to_benchmark_mlp(
        clf=sklearn_retrained,
        layers=layers,
        device=device,
    )

    method_cfg = _method_config(config=config, device=device, delta_min=delta_min)
    desired_class = int(method_cfg["desired_class"])
    candidate_factuals = _select_reference_test_instances(
        base_model=base_model,
        X1=datasets["X1"],
        desired_class=desired_class,
        num_test_instances=int(spec.num_test_instances),
    )
    sound_fraction, sound_positions = _verify_soundness(
        base_model=base_model,
        candidate_factuals=candidate_factuals,
        delta_min=delta_min,
        big_m=float(method_cfg.get("big_m", 10000.0)),
        use_biases=bool(method_cfg.get("use_biases", True)),
    )

    final_count = int(spec.num_sound_instances if limit_sound is None else limit_sound)
    final_positions = sound_positions[:final_count]
    if len(final_positions) < final_count:
        raise ValueError(
            f"{spec.name}: expected at least {final_count} sound factuals, "
            f"found {len(final_positions)}"
        )
    factuals = candidate_factuals.iloc[final_positions].reset_index(drop=True)

    apas_method = ApasMethod(target_model=base_model, **method_cfg)
    apas_method.fit(datasets["full_dataset"])
    counterfactuals = apas_method.get_counterfactuals(factuals)

    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof_model.fit(datasets["X1"].to_numpy(dtype=np.float64))

    metric_results = _evaluate_counterfactuals(
        factuals=factuals,
        counterfactuals=counterfactuals,
        base_model=base_model,
        retrained_model=retrained_model,
        delta_min=delta_min,
        desired_class=desired_class,
        big_m=float(method_cfg.get("big_m", 10000.0)),
        use_biases=bool(method_cfg.get("use_biases", True)),
        lof_model=lof_model,
        feature_groups=datasets.get("feature_groups"),
    )

    full_results_targets = spec.targets.get("full_results", {})
    results = {
        "dataset": spec.name,
        "device": device,
        "d1_size": int(datasets["X1"].shape[0]),
        "d2_size": int(datasets["X2"].shape[0]),
        "candidate_factuals": int(candidate_factuals.shape[0]),
        "sound_fraction": float(sound_fraction),
        "sound_count": int(len(sound_positions)),
        "evaluated_factuals": int(factuals.shape[0]),
        "delta": float(delta_min),
        "delta_min": float(delta_min),
        "delta_max": float(delta_max_reference),
        "delta_e": float(delta_e),
        "max_shifted_model_delta_check": _inf_norm(
            _flatten_weights_and_biases(sklearn_base),
            _flatten_weights_and_biases(max_shifted_sklearn),
        ),
        **full_results_targets,
        **metric_results,
    }

    return {
        "results": results,
        "notebook_comparison": _compare_against_targets(
            results, spec.targets.get("notebook", {})
        ),
        "paper_comparison": _compare_against_targets(
            results, spec.targets.get("paper", {})
        ),
        "full_results_comparison": _compare_against_targets(
            results, spec.targets.get("full_results", {})
        ),
    }


def _format_value(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.12f}".rstrip("0").rstrip(".")
    return str(value)


def _print_comparison_table(
    title: str,
    rows: list[tuple[str, float, float, float]],
) -> None:
    if not rows:
        return

    print(title)
    print(f"{'Metric':<28} {'Target':>16} {'Reproduced':>16} {'Abs Diff':>16}")
    for metric, target, reproduced, difference in rows:
        print(
            f"{metric:<28} "
            f"{_format_value(target):>16} "
            f"{_format_value(reproduced):>16} "
            f"{_format_value(difference):>16}"
        )
    print()


def _print_report(output: dict[str, object]) -> None:
    results = output["results"]

    print("=" * 88)
    print(f"Experiment: notebook-faithful APAS reproduction ({results['dataset']})")
    print(f"Device: {results['device']}")
    print(f"D1 size: {results['d1_size']}")
    print(f"D2 size: {results['d2_size']}")
    print(f"Candidate factuals: {results['candidate_factuals']}")
    print(
        "Sound factuals: "
        f"{results['sound_count']} / {results['candidate_factuals']} "
        f"({_format_value(results['sound_fraction'])})"
    )
    print(f"Evaluated factuals: {results['evaluated_factuals']}")
    print()

    print("Metrics")
    for label, key in [
        ("delta_min", "delta_min"),
        ("delta_max", "delta_max"),
        ("delta_e", "delta_e"),
        ("found", "found"),
        ("VM1", "vm1"),
        ("VM2", "vm2"),
        ("delta_validity", "delta_validity"),
        ("L1", "l1"),
        ("L0", "l0"),
        ("LOF", "lof"),
        ("full_results_milp_mean", "full_results_milp_mean"),
        ("full_results_wilks_mean", "full_results_wilks_mean"),
    ]:
        if key in results:
            print(f"  {label:<24} {_format_value(results[key])}")
    print()

    _print_comparison_table("Notebook Comparison", output["notebook_comparison"])
    _print_comparison_table("Paper Comparison", output["paper_comparison"])
    _print_comparison_table("Full Results Summary", output["full_results_comparison"])


def _selected_specs(selection: str) -> list[DatasetSpec]:
    if selection == "all":
        return [
            DATASET_SPECS["diabetes"],
            DATASET_SPECS["no2"],
            DATASET_SPECS["sba"],
            DATASET_SPECS["credit"],
        ]
    return [DATASET_SPECS[selection]]

@pytest.mark.slow
def test_reproduce() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./experiment/apas/config.yaml",
        help="Shared framework/APAS config path.",
    )
    parser.add_argument(
        "--dataset",
        choices=["diabetes", "no2", "sba", "credit", "all"],
        default="all",
        help="Notebook dataset to reproduce.",
    )
    parser.add_argument(
        "--limit-sound",
        type=int,
        default=None,
        help="Optional smoke-test cap for sound factuals; defaults to notebook-faithful 50.",
    )
    args = parser.parse_args()

    if args.limit_sound is not None and args.limit_sound < 1:
        raise ValueError("--limit-sound must be >= 1 when provided")

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = _load_config(config_path)
    for spec in _selected_specs(args.dataset):
        output = run_dataset_reproduction(
            spec=spec,
            config=config,
            limit_sound=args.limit_sound,
        )
        _print_report(output)


if __name__ == "__main__":
    test_reproduce()
