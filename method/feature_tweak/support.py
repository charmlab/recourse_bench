from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from model.model_object import ModelObject
from model.randomforest.randomforest import RandomForestModel
from preprocess.preprocess_utils import dataset_has_attr, resolve_feature_metadata


def ensure_supported_target_model(
    target_model: ModelObject,
    method_name: str,
) -> None:
    if isinstance(target_model, RandomForestModel):
        return
    raise TypeError(
        f"{method_name} supports target models [RandomForestModel] only, "
        f"received {target_model.__class__.__name__}"
    )


def ensure_binary_classifier(target_model: ModelObject, method_name: str) -> None:
    class_to_index = target_model.get_class_to_index()
    if len(class_to_index) != 2:
        raise ValueError(f"{method_name} supports binary classification only")


def to_feature_dataframe(
    values: pd.DataFrame | np.ndarray | torch.Tensor,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.loc[:, list(feature_names)].copy(deep=True)

    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    if array.ndim == 1:
        array = array.reshape(1, -1)
    return pd.DataFrame(array, columns=list(feature_names))


def _is_integer_valued(series: pd.Series) -> bool:
    values = series.dropna().to_numpy(dtype="float64")
    if values.size == 0:
        return False
    return bool(np.allclose(values, np.round(values)))


def _resolve_encoding_kind(
    source_name: str,
    columns: tuple[str, ...],
    encoded_value_mapping: dict[str, dict[int, object]],
) -> str:
    if all("_therm_" in column for column in columns):
        return "thermometer"
    if all("_cat_" in column for column in columns):
        return "onehot"
    if source_name in encoded_value_mapping and columns == (source_name,):
        return "mapping"
    return "scalar"


@dataclass(frozen=True)
class FeatureTweakGroup:
    source_name: str
    columns: tuple[str, ...]
    indices: tuple[int, ...]
    encoding_kind: str
    feature_kind: str
    mutable: bool
    actionability: str
    valid_codes: tuple[float, ...] | None
    lower_bound: float | None
    upper_bound: float | None
    integer_valued: bool


@dataclass(frozen=True)
class FeatureTweakContext:
    feature_names: tuple[str, ...]
    target_column: str
    groups: tuple[FeatureTweakGroup, ...]


def build_feature_tweak_context(trainset: DatasetObject) -> FeatureTweakContext:
    feature_df = trainset.get(target=False)
    feature_names = list(feature_df.columns)
    encoded_feature_type, encoded_feature_mutability, encoded_feature_actionability = (
        resolve_feature_metadata(trainset)
    )

    raw_feature_type = (
        trainset.attr("raw_feature_type")
        if dataset_has_attr(trainset, "raw_feature_type")
        else encoded_feature_type
    )
    raw_feature_mutability = (
        trainset.attr("raw_feature_mutability")
        if dataset_has_attr(trainset, "raw_feature_mutability")
        else encoded_feature_mutability
    )
    raw_feature_actionability = (
        trainset.attr("raw_feature_actionability")
        if dataset_has_attr(trainset, "raw_feature_actionability")
        else encoded_feature_actionability
    )
    encoding_map = (
        trainset.attr("encoding") if dataset_has_attr(trainset, "encoding") else {}
    )
    encoded_value_mapping = (
        trainset.attr("encoded_value_mapping")
        if dataset_has_attr(trainset, "encoded_value_mapping")
        else {}
    )

    feature_name_set = set(feature_names)
    processed_to_source: dict[str, str] = {}
    for source_name, encoded_columns in encoding_map.items():
        present_columns = tuple(
            str(column) for column in encoded_columns if column in feature_name_set
        )
        if present_columns and len(present_columns) != len(tuple(encoded_columns)):
            raise ValueError(
                f"Incomplete encoded feature group for '{source_name}': {present_columns}"
            )
        for column in present_columns:
            processed_to_source[column] = str(source_name)

    grouped_columns: dict[str, list[str]] = {}
    ordered_sources: list[str] = []
    for feature_name in feature_names:
        source_name = processed_to_source.get(feature_name, feature_name)
        if source_name not in grouped_columns:
            grouped_columns[source_name] = []
            ordered_sources.append(source_name)
        grouped_columns[source_name].append(feature_name)

    groups: list[FeatureTweakGroup] = []
    for source_name in ordered_sources:
        columns = tuple(grouped_columns[source_name])
        reference_column = columns[0]
        raw_kind = str(
            raw_feature_type.get(source_name, encoded_feature_type[reference_column])
        ).lower()
        mutable = bool(
            raw_feature_mutability.get(
                source_name,
                encoded_feature_mutability[reference_column],
            )
        )
        actionability = str(
            raw_feature_actionability.get(
                source_name,
                encoded_feature_actionability[reference_column],
            )
        ).lower()
        encoding_kind = _resolve_encoding_kind(
            source_name=source_name,
            columns=columns,
            encoded_value_mapping=encoded_value_mapping,
        )

        valid_codes: tuple[float, ...] | None = None
        lower_bound: float | None = None
        upper_bound: float | None = None
        integer_valued = False

        if encoding_kind == "mapping":
            valid_codes = tuple(
                float(code)
                for code in sorted(encoded_value_mapping[source_name].keys())
            )
        elif encoding_kind == "scalar" and raw_kind == "categorical":
            valid_codes = tuple(
                float(value)
                for value in sorted(
                    pd.Index(feature_df[reference_column].dropna().unique()).tolist()
                )
            )
        elif encoding_kind == "scalar" and raw_kind == "numerical":
            series = feature_df[reference_column]
            lower_bound = float(series.min())
            upper_bound = float(series.max())
            integer_valued = _is_integer_valued(series)

        groups.append(
            FeatureTweakGroup(
                source_name=str(source_name),
                columns=columns,
                indices=tuple(feature_names.index(column) for column in columns),
                encoding_kind=encoding_kind,
                feature_kind=raw_kind,
                mutable=mutable,
                actionability=actionability,
                valid_codes=valid_codes,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                integer_valued=integer_valued,
            )
        )

    return FeatureTweakContext(
        feature_names=tuple(feature_names),
        target_column=str(trainset.target_column),
        groups=tuple(groups),
    )


def _project_onehot(values: np.ndarray) -> np.ndarray:
    finite_values = np.nan_to_num(values.astype("float64"), nan=-np.inf)
    best_index = int(np.argmax(finite_values))
    projected = np.zeros(values.shape[0], dtype="float64")
    projected[best_index] = 1.0
    return projected


def _project_thermometer(values: np.ndarray) -> np.ndarray:
    size = int(values.shape[0])
    legal_patterns = np.tril(np.ones((size, size), dtype="float64"))
    distances = np.linalg.norm(legal_patterns - values.reshape(1, -1), axis=1)
    best_index = int(np.argmin(distances))
    return legal_patterns[best_index].copy()


def _project_binary_scalar(value: float) -> float:
    return float(np.clip(np.rint(value), 0.0, 1.0))


def _project_discrete_scalar(
    value: float,
    valid_codes: tuple[float, ...],
    fallback_value: float,
) -> float:
    if not valid_codes:
        return float(fallback_value)
    if not np.isfinite(value):
        return float(fallback_value)
    valid_array = np.asarray(valid_codes, dtype="float64")
    best_index = int(np.argmin(np.abs(valid_array - float(value))))
    return float(valid_array[best_index])


def _project_numerical_scalar(
    value: float,
    lower_bound: float | None,
    upper_bound: float | None,
    integer_valued: bool,
    fallback_value: float,
) -> float:
    if not np.isfinite(value):
        value = float(fallback_value)
    if integer_valued:
        value = float(np.rint(value))
    if lower_bound is not None:
        value = max(value, float(lower_bound))
    if upper_bound is not None:
        value = min(value, float(upper_bound))
    return float(value)


def project_candidate_features(
    candidate: np.ndarray,
    factual: np.ndarray,
    context: FeatureTweakContext,
) -> np.ndarray:
    candidate_values = np.asarray(candidate, dtype="float64").reshape(-1)
    factual_values = np.asarray(factual, dtype="float64").reshape(-1)
    if candidate_values.shape != factual_values.shape:
        raise ValueError("candidate and factual must share the same feature shape")
    if candidate_values.shape[0] != len(context.feature_names):
        raise ValueError("candidate shape does not match feature context")

    projected = candidate_values.copy()
    for group in context.groups:
        group_indices = np.asarray(group.indices, dtype=np.int64)
        if (not group.mutable) or group.actionability in {"none", "same"}:
            projected[group_indices] = factual_values[group_indices]
            continue

        group_values = projected[group_indices]
        factual_group_values = factual_values[group_indices]
        if group.encoding_kind == "onehot":
            projected[group_indices] = _project_onehot(group_values)
            continue
        if group.encoding_kind == "thermometer":
            projected[group_indices] = _project_thermometer(group_values)
            continue

        value = float(group_values[0])
        factual_value = float(factual_group_values[0])
        if group.valid_codes is not None:
            projected[group_indices] = _project_discrete_scalar(
                value=value,
                valid_codes=group.valid_codes,
                fallback_value=factual_value,
            )
        elif group.feature_kind == "binary":
            projected[group_indices] = _project_binary_scalar(value)
        elif group.feature_kind == "numerical":
            projected[group_indices] = _project_numerical_scalar(
                value=value,
                lower_bound=group.lower_bound,
                upper_bound=group.upper_bound,
                integer_valued=group.integer_valued,
                fallback_value=factual_value,
            )
        else:
            projected[group_indices] = factual_value

    return projected


class FeatureTweakTargetModelAdapter:
    def __init__(
        self,
        target_model: RandomForestModel,
        feature_context: FeatureTweakContext,
    ):
        self._target_model = target_model
        self._model = target_model._model
        self.feature_input_order = list(feature_context.feature_names)
        self.tree_iterator = tuple(self._model.estimators_)
        self.classes_ = np.arange(
            len(target_model.get_class_to_index()), dtype=np.int64
        )

    def _to_feature_array(
        self,
        X: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            array = X.loc[:, self.feature_input_order].to_numpy(dtype="float64")
        elif isinstance(X, torch.Tensor):
            array = X.detach().cpu().numpy()
        else:
            array = np.asarray(X, dtype="float64")

        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    def get_ordered_features(
        self,
        X: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> pd.DataFrame:
        return to_feature_dataframe(X, self.feature_input_order)

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names",
                category=UserWarning,
            )
            return np.asarray(self._model.predict_proba(self._to_feature_array(X)))

    def predict(
        self,
        X: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names",
                category=UserWarning,
            )
            return np.asarray(self._model.predict(self._to_feature_array(X))).astype(
                np.int64,
                copy=False,
            )


def check_counterfactuals(
    mlmodel: FeatureTweakTargetModelAdapter,
    counterfactuals: list[np.ndarray] | pd.DataFrame,
    factuals: pd.DataFrame,
    desired_class: int,
) -> pd.DataFrame:
    feature_names = list(mlmodel.feature_input_order)
    if isinstance(counterfactuals, list):
        if counterfactuals:
            array = np.asarray(counterfactuals, dtype="float64")
            df_counterfactuals = pd.DataFrame(
                array,
                index=factuals.index.copy(),
                columns=feature_names,
            )
        else:
            df_counterfactuals = factuals.loc[:, feature_names].copy(deep=True)
    else:
        df_counterfactuals = mlmodel.get_ordered_features(counterfactuals)
        df_counterfactuals = df_counterfactuals.reindex(
            index=factuals.index,
            columns=feature_names,
        )

    if df_counterfactuals.shape[0] != factuals.shape[0]:
        raise ValueError("Counterfactuals must preserve factual row count")

    valid_rows = ~df_counterfactuals.isna().any(axis=1)
    if bool(valid_rows.any()):
        valid_index = df_counterfactuals.index[valid_rows.to_numpy()]
        predicted_labels = mlmodel.predict(df_counterfactuals.loc[valid_index])
        success_mask = np.asarray(predicted_labels, dtype=np.int64) == int(
            desired_class
        )
        failed_index = valid_index[~success_mask]
        if len(failed_index) > 0:
            df_counterfactuals.loc[failed_index, :] = np.nan

    return df_counterfactuals.reindex(index=factuals.index, columns=feature_names)
