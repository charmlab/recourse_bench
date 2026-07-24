from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset.dataset_object import DatasetObject
from model.model_object import ModelObject, process_nan
from preprocess.preprocess_utils import resolve_feature_metadata
from utils.seed import seed_context


def _read_reference_csv(data_dir: Path, filename: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(data_dir / filename, **kwargs)


def _plausibility_to_metadata(
    feature_names: list[str],
    categorical_feature_names: list[str],
    plausibility_constraints: dict[str, str | None],
    target_column: str,
) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
    raw_feature_type: dict[str, str] = {}
    raw_feature_mutability: dict[str, bool] = {}
    raw_feature_actionability: dict[str, str] = {}

    for feature_name in feature_names:
        constraint = plausibility_constraints[feature_name]
        if feature_name in categorical_feature_names:
            raw_feature_type[feature_name] = "categorical"
        else:
            raw_feature_type[feature_name] = "numerical"

        if constraint == "=":
            raw_feature_mutability[feature_name] = False
            raw_feature_actionability[feature_name] = "none"
        elif constraint == ">=":
            raw_feature_mutability[feature_name] = True
            raw_feature_actionability[feature_name] = "same-or-increase"
        elif constraint == "<=":
            raw_feature_mutability[feature_name] = True
            raw_feature_actionability[feature_name] = "same-or-decrease"
        else:
            raw_feature_mutability[feature_name] = True
            raw_feature_actionability[feature_name] = "any"

    raw_feature_type[target_column] = "categorical"
    raw_feature_mutability[target_column] = False
    raw_feature_actionability[target_column] = "none"
    return raw_feature_type, raw_feature_mutability, raw_feature_actionability


def _boston_spec(data_dir: Path) -> dict[str, object]:
    xy = np.genfromtxt(data_dir / "boston_housing.csv", delimiter=",")
    X = xy[:, :-1]
    y = np.array([1 if value > 26.0 else 0 for value in xy[:, -1]], dtype=np.int64)

    feature_names = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
    ]
    categorical_feature_names = ["CHAS"]
    df = pd.DataFrame({feature_name: X[:, index] for index, feature_name in enumerate(feature_names)})
    df["LABEL"] = y

    intervals = {
        "CRIM": (0.0, 100.0),
        "ZN": (0.0, 100.0),
        "INDUS": (0.0, 30.0),
        "CHAS": [0, 1],
        "NOX": (0.3, 1.0),
        "RM": (2.0, 10.0),
        "AGE": (2.0, 100.0),
        "DIS": (0.5, 15.0),
        "RAD": (1.0, 24.0),
        "TAX": (150.0, 800.0),
        "PTRATIO": (10.0, 25.0),
        "B": (0.0, 400.0),
        "LSTAT": (1.0, 40.0),
    }
    perturbations = {
        "CRIM": {"type": "relative", "increase": 0.05, "decrease": 0.01},
        "ZN": {"type": "relative", "increase": 0.05, "decrease": 0.01},
        "INDUS": {"type": "relative", "increase": 0.01, "decrease": 0.02},
        "CHAS": None,
        "NOX": {"type": "relative", "increase": 0.05, "decrease": 0.01},
        "RM": {"type": "absolute", "increase": 1.0, "decrease": 1.0},
        "AGE": {"type": "relative", "increase": 0.0, "decrease": 0.05},
        "DIS": None,
        "RAD": {"type": "absolute", "increase": 3.0, "decrease": 0.0},
        "TAX": {"type": "relative", "increase": 0.05, "decrease": 0.05},
        "PTRATIO": {"type": "relative", "increase": 0.05, "decrease": 0.03},
        "B": {"type": "relative", "increase": 0.05, "decrease": 0.05},
        "LSTAT": {"type": "absolute", "increase": 5.0, "decrease": 5.0},
    }
    plausibility = {
        "CRIM": None,
        "ZN": None,
        "INDUS": None,
        "CHAS": "=",
        "NOX": None,
        "RM": None,
        "AGE": "<=",
        "DIS": "=",
        "RAD": None,
        "TAX": None,
        "PTRATIO": None,
        "B": "=",
        "LSTAT": None,
    }

    return {
        "name": "boston",
        "target_column": "LABEL",
        "feature_names": feature_names,
        "categorical_feature_names": categorical_feature_names,
        "best_class": 0,
        "df": df.loc[:, [*feature_names, "LABEL"]].copy(deep=True),
        "feature_intervals": [intervals[name] for name in feature_names],
        "perturbations": [perturbations[name] for name in feature_names],
        "plausibility_constraints": [plausibility[name] for name in feature_names],
    }


def _compas_spec(data_dir: Path) -> dict[str, object]:
    df = _read_reference_csv(data_dir, "compas-scores-two-years.csv")
    columns = [
        "age",
        "sex",
        "race",
        "priors_count",
        "days_b_screening_arrest",
        "c_jail_in",
        "c_jail_out",
        "c_charge_degree",
        "is_recid",
        "is_violent_recid",
        "two_year_recid",
        "decile_score",
        "score_text",
    ]
    df = df.loc[:, columns].copy(deep=True)
    df = df.drop(columns=["score_text"]).copy(deep=True)
    df["days_b_screening_arrest"] = np.abs(df["days_b_screening_arrest"])
    df["c_jail_out"] = pd.to_datetime(df["c_jail_out"])
    df["c_jail_in"] = pd.to_datetime(df["c_jail_in"])
    df["length_of_stay"] = (df["c_jail_out"] - df["c_jail_in"]).dt.days.abs()
    df = df.dropna(axis=0).copy(deep=True)
    df = df.drop(columns=["c_jail_in", "c_jail_out"]).copy(deep=True)
    df["length_of_stay"] = df["length_of_stay"].astype(int)
    df["days_b_screening_arrest"] = df["days_b_screening_arrest"].astype(int)
    df["LABEL"] = (df["decile_score"] >= 7).astype(int)
    df = df.drop(columns=["decile_score"]).copy(deep=True)
    df = df.iloc[:2000].copy(deep=True)

    feature_names = [column for column in df.columns if column != "LABEL"]
    categorical_feature_names = [
        "sex",
        "race",
        "c_charge_degree",
        "is_recid",
        "is_violent_recid",
        "two_year_recid",
    ]
    for feature_name in categorical_feature_names:
        df[feature_name] = pd.Categorical(df[feature_name]).codes

    intervals = {feature_name: sorted(df[feature_name].unique().tolist()) for feature_name in categorical_feature_names}
    intervals.update(
        {
            "age": (18.0, 100.0),
            "priors_count": (0.0, 100.0),
            "days_b_screening_arrest": (0.0, 1200.0),
            "length_of_stay": (0.0, 1200.0),
        }
    )
    perturbations = {feature_name: None for feature_name in feature_names}
    plausibility = {feature_name: None for feature_name in feature_names}
    perturbations["age"] = {"type": "absolute", "increase": 2.0, "decrease": 0.0}
    plausibility["age"] = ">="
    perturbations["length_of_stay"] = {"type": "absolute", "increase": 730.0, "decrease": 0.0}
    plausibility["length_of_stay"] = ">="
    perturbations["priors_count"] = {"type": "absolute", "increase": 3.0, "decrease": 2.0}
    perturbations["is_recid"] = {"type": "absolute", "categories": sorted(df["is_recid"].unique().tolist())}
    perturbations["two_year_recid"] = {
        "type": "absolute",
        "categories": sorted(df["two_year_recid"].unique().tolist()),
    }
    plausibility["is_violent_recid"] = "="
    plausibility["sex"] = "="

    return {
        "name": "compas",
        "target_column": "LABEL",
        "feature_names": feature_names,
        "categorical_feature_names": categorical_feature_names,
        "best_class": 0,
        "df": df.loc[:, [*feature_names, "LABEL"]].copy(deep=True),
        "feature_intervals": [intervals[name] for name in feature_names],
        "perturbations": [perturbations[name] for name in feature_names],
        "plausibility_constraints": [plausibility[name] for name in feature_names],
    }


def _adult_spec(data_dir: Path) -> dict[str, object]:
    df = _read_reference_csv(data_dir, "adult.csv", delimiter=",", skipinitialspace=True)
    df = df.drop(columns=["fnlwgt", "education"]).copy(deep=True)
    df = df.replace("?", np.nan).dropna(axis=0).copy(deep=True)

    categorical_feature_names = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    for feature_name in categorical_feature_names:
        df[feature_name] = pd.Categorical(df[feature_name]).codes

    df["LABEL"] = df["income"].astype(int)
    df = df.drop(columns=["income"]).copy(deep=True)
    feature_names = [column for column in df.columns if column != "LABEL"]

    intervals = {feature_name: sorted(df[feature_name].unique().tolist()) for feature_name in categorical_feature_names}
    intervals.update(
        {
            "age": (17.0, 90.0),
            "education-num": (2.0, 16.0),
            "capital-gain": (0.0, 99999.0),
            "capital-loss": (0.0, 4500.0),
            "hours-per-week": (1.0, 99.0),
        }
    )
    perturbations: dict[str, dict[str, object] | None] = {}
    plausibility: dict[str, str | None] = {}
    for feature_name in feature_names:
        if feature_name in {"workclass", "marital-status", "occupation", "relationship"}:
            perturbations[feature_name] = {
                "type": "absolute",
                "categories": sorted(df[feature_name].unique().tolist()),
            }
            plausibility[feature_name] = None
        elif feature_name in {"race", "native-country", "sex"}:
            perturbations[feature_name] = None
            plausibility[feature_name] = "="
        elif feature_name == "age":
            perturbations[feature_name] = {"type": "absolute", "increase": 2.0, "decrease": 0.0}
            plausibility[feature_name] = ">="
        elif feature_name == "capital-gain":
            perturbations[feature_name] = {"type": "relative", "increase": 0.2, "decrease": 0.2}
            plausibility[feature_name] = None
        elif feature_name == "capital-loss":
            perturbations[feature_name] = {"type": "absolute", "increase": 1000.0, "decrease": 1000.0}
            plausibility[feature_name] = None
        elif feature_name == "education-num":
            perturbations[feature_name] = {"type": "absolute", "increase": 1.0, "decrease": 0.0}
            plausibility[feature_name] = ">="
        elif feature_name == "hours-per-week":
            perturbations[feature_name] = {"type": "absolute", "increase": 5.0, "decrease": 5.0}
            plausibility[feature_name] = None
        else:
            raise KeyError(f"Unexpected adult feature: {feature_name}")

    return {
        "name": "adult",
        "target_column": "LABEL",
        "feature_names": feature_names,
        "categorical_feature_names": categorical_feature_names,
        "best_class": 1,
        "df": df.loc[:, [*feature_names, "LABEL"]].copy(deep=True),
        "feature_intervals": [intervals[name] for name in feature_names],
        "perturbations": [perturbations[name] for name in feature_names],
        "plausibility_constraints": [plausibility[name] for name in feature_names],
    }


def _credit_spec(data_dir: Path) -> dict[str, object]:
    df = _read_reference_csv(data_dir, "south_german_credit.csv")
    categorical_feature_names = [
        "purpose",
        "personal_status_sex",
        "other_debtors",
        "other_installment_plans",
        "telephone",
        "foreign_worker",
    ]
    for feature_name in categorical_feature_names:
        df[feature_name] = pd.Categorical(df[feature_name]).codes

    df["LABEL"] = df["credit_risk"]
    df = df.drop(columns=["credit_risk"]).copy(deep=True)
    feature_names = [column for column in df.columns if column != "LABEL"]

    intervals = {feature_name: sorted(df[feature_name].unique().tolist()) for feature_name in categorical_feature_names}
    for feature_name in feature_names:
        if feature_name not in categorical_feature_names:
            intervals[feature_name] = (
                float(df[feature_name].min()),
                float(df[feature_name].max()),
            )
    intervals["age"] = (18.0, 75.0)
    intervals["duration_in_month"] = (3.0, 75.0)
    intervals["credit_amount"] = (250.0, 20000.0)

    perturbations: dict[str, dict[str, object] | None] = {}
    plausibility: dict[str, str | None] = {}
    perturbations["age"] = {"type": "absolute", "increase": 0.5, "decrease": 0.0}
    plausibility["age"] = ">="
    perturbations["installment_as_income_perc"] = {"type": "relative", "increase": 0.10, "decrease": 0.10}
    plausibility["installment_as_income_perc"] = None
    perturbations["present_res_since"] = None
    plausibility["present_res_since"] = ">="
    perturbations["duration_in_month"] = {"type": "relative", "increase": 0.25, "decrease": 0.05}
    plausibility["duration_in_month"] = None
    perturbations["purpose"] = None
    plausibility["purpose"] = "="
    perturbations["housing"] = None
    plausibility["housing"] = None
    for feature_name in ["account_check_status", "credit_this_bank"]:
        perturbations[feature_name] = {"type": "absolute", "increase": 1.0, "decrease": 1.0}
        plausibility[feature_name] = None
    perturbations["credit_amount"] = {"type": "relative", "increase": 0.1, "decrease": 0.1}
    plausibility["credit_amount"] = None
    perturbations["savings"] = {"type": "relative", "increase": 0.1, "decrease": 0.1}
    plausibility["savings"] = None
    perturbations["present_emp_since"] = None
    plausibility["present_emp_since"] = ">="
    for feature_name in ["other_installment_plans", "credits_this_bank", "job"]:
        perturbations[feature_name] = None
        plausibility[feature_name] = None
    for feature_name in [
        "credit_history",
        "personal_status_sex",
        "other_debtors",
        "property",
        "telephone",
        "foreign_worker",
        "people_under_maintenance",
    ]:
        perturbations[feature_name] = None
        plausibility[feature_name] = "="

    for feature_name in feature_names:
        perturbations.setdefault(feature_name, None)
        plausibility.setdefault(feature_name, None)

    return {
        "name": "credit",
        "target_column": "LABEL",
        "feature_names": feature_names,
        "categorical_feature_names": categorical_feature_names,
        "best_class": 1,
        "df": df.loc[:, [*feature_names, "LABEL"]].copy(deep=True),
        "feature_intervals": [intervals[name] for name in feature_names],
        "perturbations": [perturbations[name] for name in feature_names],
        "plausibility_constraints": [plausibility[name] for name in feature_names],
    }


def _garments_spec(data_dir: Path) -> dict[str, object]:
    df = _read_reference_csv(data_dir, "garments_worker_productivity.csv")
    df = df.drop(columns=["date", "wip"]).copy(deep=True)
    mask_unique = ~df.drop(columns=["actual_productivity"]).duplicated()
    df = df.loc[mask_unique].copy(deep=True)
    df["LABEL"] = 0
    df.loc[df["actual_productivity"] > 0.7, "LABEL"] = 1
    df.loc[df["actual_productivity"] > 0.81, "LABEL"] = 2
    df = df.drop(columns=["actual_productivity"]).copy(deep=True)

    feature_names = [column for column in df.columns if column != "LABEL"]
    categorical_feature_names = [
        "quarter",
        "department",
        "day",
        "team",
        "no_of_style_change",
    ]
    for feature_name in categorical_feature_names:
        df[feature_name] = pd.Categorical(df[feature_name]).codes

    intervals = {feature_name: sorted(df[feature_name].unique().tolist()) for feature_name in categorical_feature_names}
    intervals.update(
        {
            "smv": (2.5, 60.0),
            "over_time": (0.0, 25920.0),
            "incentive": (0.0, 3600.0),
            "idle_time": (0.0, 300.0),
            "idle_men": (0.0, 50.0),
            "no_of_workers": (1.0, 100.0),
            "targeted_productivity": (0.05, 0.8),
        }
    )
    perturbations = {
        "team": None,
        "day": {"type": "absolute", "categories": sorted(df["day"].unique().tolist())},
        "quarter": {"type": "absolute", "categories": sorted(df["quarter"].unique().tolist())},
        "department": None,
        "no_of_workers": {"type": "relative", "increase": 0.0, "decrease": 0.1},
        "no_of_style_change": None,
        "targeted_productivity": None,
        "smv": {"type": "relative", "increase": 0.1, "decrease": 0.1},
        "over_time": {"type": "absolute", "increase": 4320.0, "decrease": 4320.0},
        "incentive": None,
        "idle_time": {"type": "relative", "increase": 0.1, "decrease": 0.05},
        "idle_men": {"type": "relative", "increase": 0.1, "decrease": 0.05},
    }
    plausibility = {
        "team": None,
        "day": None,
        "quarter": None,
        "department": None,
        "no_of_workers": None,
        "no_of_style_change": None,
        "targeted_productivity": None,
        "smv": None,
        "over_time": None,
        "incentive": None,
        "idle_time": None,
        "idle_men": None,
    }

    return {
        "name": "garments",
        "target_column": "LABEL",
        "feature_names": feature_names,
        "categorical_feature_names": categorical_feature_names,
        "best_class": 2,
        "df": df.loc[:, [*feature_names, "LABEL"]].copy(deep=True),
        "feature_intervals": [intervals[name] for name in feature_names],
        "perturbations": [perturbations[name] for name in feature_names],
        "plausibility_constraints": [plausibility[name] for name in feature_names],
    }


DATASET_SPECS = {
    "adult": _adult_spec,
    "boston": _boston_spec,
    "compas": _compas_spec,
    "credit": _credit_spec,
    "garments": _garments_spec,
}


class CoGSReferenceDataset(DatasetObject):
    def __init__(self, name: str, path: str | Path | None = None, **kwargs):
        dataset_name = str(name).lower()
        if dataset_name not in DATASET_SPECS:
            raise KeyError(f"Unsupported CoGS reference dataset: {dataset_name}")

        data_dir = Path(path) if path is not None else Path(__file__).with_name("data")
        spec = DATASET_SPECS[dataset_name](data_dir)

        self._rawdf = spec["df"].copy(deep=True)
        self._freeze = False
        self.name = spec["name"]
        self.target_column = spec["target_column"]
        self.best_class = spec["best_class"]
        self.feature_order = [*spec["feature_names"], self.target_column]
        self.feature_intervals = deepcopy(spec["feature_intervals"])
        self.perturbations = deepcopy(spec["perturbations"])
        self.reference_plausibility_constraints = deepcopy(spec["plausibility_constraints"])
        (
            self.raw_feature_type,
            self.raw_feature_mutability,
            self.raw_feature_actionability,
        ) = _plausibility_to_metadata(
            feature_names=spec["feature_names"],
            categorical_feature_names=spec["categorical_feature_names"],
            plausibility_constraints={
                feature_name: constraint
                for feature_name, constraint in zip(
                    spec["feature_names"],
                    spec["plausibility_constraints"],
                    strict=True,
                )
            },
            target_column=self.target_column,
        )
        self._rawdf = self._rawdf.loc[:, self.feature_order].copy(deep=True)

    def _read_df(self, path: str) -> pd.DataFrame:
        raise NotImplementedError(
            "CoGSReferenceDataset does not use DatasetObject._read_df()"
        )


class _ReferenceBlackboxPreprocessor:
    def __init__(
        self,
        indices_categorical_features: list[int],
        preprocs: list[str],
    ):
        self.indices_categorical_features = list(indices_categorical_features)
        self.preprocs = list(preprocs)
        self.nonbinary_cat_features: list[int] = []
        self.ohe: OneHotEncoder | None = None
        self.scaler: StandardScaler | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self._apply(X, fit=True)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._apply(X, fit=False)

    def _apply(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X_prime = np.asarray(X, dtype=np.float64).copy()
        for preproc in self.preprocs:
            if preproc == "onehot":
                if fit:
                    self.nonbinary_cat_features = [
                        index
                        for index in self.indices_categorical_features
                        if len(np.unique(X_prime[:, index])) > 2
                    ]
                if len(self.nonbinary_cat_features) == 0:
                    continue
                X_cat = X_prime[:, self.nonbinary_cat_features]
                if fit:
                    self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                    X_ohe = self.ohe.fit_transform(X_cat)
                else:
                    if self.ohe is None:
                        raise RuntimeError("OneHotEncoder is not fitted")
                    X_ohe = self.ohe.transform(X_cat)
                X_prime = np.delete(X_prime, self.nonbinary_cat_features, axis=1)
                X_prime = np.concatenate((X_prime, X_ohe), axis=1)
            elif preproc == "standard_scale":
                num_feature_indices = [
                    index for index in range(X.shape[1]) if index not in self.indices_categorical_features
                ]
                X_num = X_prime[:, num_feature_indices]
                if fit:
                    self.scaler = StandardScaler()
                    X_scaled = self.scaler.fit_transform(X_num)
                else:
                    if self.scaler is None:
                        raise RuntimeError("StandardScaler is not fitted")
                    X_scaled = self.scaler.transform(X_num)
                X_prime = np.delete(X_prime, num_feature_indices, axis=1)
                X_prime = np.concatenate((X_prime, X_scaled), axis=1)
            else:
                raise ValueError(f"Unknown preprocessing step: {preproc}")
        return X_prime


class ReferenceBlackboxModel(ModelObject):
    def __init__(
        self,
        kind: str,
        seed: int = 42,
        device: str = "cpu",
        cv_folds: int = 5,
        n_jobs: int = 1,
        rf_n_estimators: list[int] | None = None,
        rf_min_samples_split: list[int] | None = None,
        rf_max_features: list[str | None] | None = None,
        nn_learning_rate_init: list[float] | None = None,
        nn_solver: list[str] | None = None,
        nn_max_iter: list[int] | None = None,
        **kwargs,
    ):
        self._seed = int(seed)
        self._device = str(device).lower()
        self._need_grad = False
        self._is_trained = False
        self._kind = str(kind).lower()
        self._cv_folds = int(cv_folds)
        self._n_jobs = int(n_jobs)
        self._rf_n_estimators = list(rf_n_estimators or [50, 500])
        self._rf_min_samples_split = list(rf_min_samples_split or [2, 8])
        self._rf_max_features = list(rf_max_features or ["sqrt", None])
        self._nn_learning_rate_init = list(nn_learning_rate_init or [1e-2, 1e-4])
        self._nn_solver = list(nn_solver or ["adam", "sgd"])
        self._nn_max_iter = list(nn_max_iter or [200, 1000])
        self._preprocessor: _ReferenceBlackboxPreprocessor | None = None
        self._model = None
        self._best_params: dict[str, object] | None = None

        if self._device != "cpu":
            raise ValueError("ReferenceBlackboxModel only supports cpu")
        if self._kind not in {"rf", "nn"}:
            raise ValueError("kind must be 'rf' or 'nn'")

    def fit(self, trainset: DatasetObject | None):
        if trainset is None:
            raise ValueError("trainset is required for ReferenceBlackboxModel.fit()")

        with seed_context(self._seed):
            X_df, labels, _ = self.extract_training_data(trainset)
            X = X_df.to_numpy(dtype=np.float64)
            y = labels.cpu().numpy()
            feature_type, _, _ = resolve_feature_metadata(trainset)
            categorical_indices = [
                index
                for index, feature_name in enumerate(X_df.columns.tolist())
                if str(feature_type[feature_name]).lower() != "numerical"
            ]
            preprocs = ["onehot"] if self._kind == "rf" else ["standard_scale", "onehot"]
            self._preprocessor = _ReferenceBlackboxPreprocessor(categorical_indices, preprocs)
            X_prepared = self._preprocessor.fit_transform(X)

            if self._kind == "rf":
                estimator = RandomForestClassifier(random_state=self._seed)
                param_grid = {
                    "n_estimators": self._rf_n_estimators,
                    "min_samples_split": self._rf_min_samples_split,
                    "max_features": self._rf_max_features,
                }
            else:
                estimator = MLPClassifier(random_state=self._seed)
                param_grid = {
                    "learning_rate_init": self._nn_learning_rate_init,
                    "solver": self._nn_solver,
                    "max_iter": self._nn_max_iter,
                }

            cv = StratifiedKFold(
                n_splits=self._cv_folds,
                shuffle=True,
                random_state=self._seed,
            )
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                refit=True,
                cv=cv,
                n_jobs=self._n_jobs,
            )
            search.fit(X_prepared, y)
            self._model = search.best_estimator_
            self._best_params = dict(search.best_params_)
            self._is_trained = True

    @process_nan()
    def get_prediction(self, X: pd.DataFrame, proba: bool = True) -> torch.Tensor:
        if not self._is_trained or self._model is None or self._preprocessor is None:
            raise RuntimeError("Target model is not trained")
        features = X.to_numpy(dtype=np.float64)
        prepared = self._preprocessor.transform(features)
        if proba:
            probabilities = self._model.predict_proba(prepared)
            return torch.tensor(probabilities, dtype=torch.float32)

        predictions = self._model.predict(prepared)
        encoded = torch.tensor(
            [self.get_class_to_index()[int(value)] for value in predictions.tolist()],
            dtype=torch.long,
        )
        return torch.nn.functional.one_hot(
            encoded,
            num_classes=len(self.get_class_to_index()),
        ).to(dtype=torch.float32)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise TypeError(
            "ReferenceBlackboxModel.forward() is unavailable because the underlying model is sklearn-based"
        )

    def get_best_params(self) -> dict[str, object] | None:
        return None if self._best_params is None else dict(self._best_params)
