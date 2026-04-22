from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from dataset.dataset_object import DatasetObject
from model.model_object import ModelObject
from preprocess.preprocess_utils import dataset_has_attr

_MONOTONE_ACTIONABILITY = {
    "same-or-increase",
    "increase",
    "same-or-decrease",
    "decrease",
}


def _normalize_category_key(value: object) -> object:
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return int(numeric_value)
        return numeric_value
    return str(value)


def _parse_category_suffix(raw_value: str) -> object:
    try:
        numeric_value = float(raw_value)
    except ValueError:
        return raw_value
    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value


def _infer_python_type(series: pd.Series) -> type:
    values = pd.Series(series).dropna().to_numpy()
    if values.size == 0:
        return float
    if all(
        isinstance(value, (int, np.integer))
        or (isinstance(value, (float, np.floating)) and float(value).is_integer())
        for value in values
    ):
        return int
    return float


@dataclass(frozen=True)
class CfrlFeature:
    name: str
    output_columns: tuple[str, ...]
    raw_feature_kind: str
    encoding_kind: str
    mutable: bool
    actionability: str
    categories: tuple[object, ...]
    lower_bound: float
    upper_bound: float
    output_python_type: type
    round_output: bool


@dataclass(frozen=True)
class CfrlSchema:
    feature_names: list[str]
    output_columns: list[str]
    features: list[CfrlFeature]
    category_map: dict[int, list[str]]
    raw_to_idx: dict[str, dict[object, int]]
    idx_to_raw: dict[str, dict[int, object]]
    attr_bounds: dict[str, list[float]]
    categorical_indices: list[int]
    numerical_indices: list[int]
    feature_types: dict[str, type]
    output_to_source: dict[str, str]

    def source_name_for_alias(self, name: str) -> str:
        if name in self.feature_names:
            return name
        if name in self.output_to_source:
            return self.output_to_source[name]
        raise KeyError(f"Unknown CFRL feature alias: {name}")


def build_cfrl_schema(trainset: DatasetObject) -> CfrlSchema:
    feature_df = trainset.get(target=False)
    output_columns = list(feature_df.columns)
    raw_feature_type = trainset.attr("raw_feature_type")
    raw_feature_mutability = trainset.attr("raw_feature_mutability")
    raw_feature_actionability = trainset.attr("raw_feature_actionability")
    encoding_map = (
        trainset.attr("encoding") if dataset_has_attr(trainset, "encoding") else {}
    )
    encoded_value_mapping = (
        trainset.attr("encoded_value_mapping")
        if dataset_has_attr(trainset, "encoded_value_mapping")
        else {}
    )
    scaling_map = (
        trainset.attr("scaling") if dataset_has_attr(trainset, "scaling") else {}
    )

    output_to_source: dict[str, str] = {}
    for source_name, encoded_columns in encoding_map.items():
        for encoded_column in encoded_columns:
            if encoded_column in output_columns:
                output_to_source[str(encoded_column)] = str(source_name)

    features: list[CfrlFeature] = []
    feature_names: list[str] = []
    category_map: dict[int, list[str]] = {}
    raw_to_idx: dict[str, dict[object, int]] = {}
    idx_to_raw: dict[str, dict[int, object]] = {}
    attr_bounds: dict[str, list[float]] = {}
    categorical_indices: list[int] = []
    numerical_indices: list[int] = []
    feature_types: dict[str, type] = {}
    seen_sources: set[str] = set()

    for column in output_columns:
        source_name = output_to_source.get(column, column)
        if source_name in seen_sources:
            continue

        if source_name in encoding_map:
            configured_columns = tuple(
                str(encoded_column) for encoded_column in encoding_map[source_name]
            )
            present_columns = tuple(
                encoded_column
                for encoded_column in configured_columns
                if encoded_column in output_columns
            )
            if present_columns and len(present_columns) != len(configured_columns):
                raise ValueError(
                    f"Incomplete encoded feature group for '{source_name}': {present_columns}"
                )
            group_columns = present_columns if present_columns else (column,)
        else:
            group_columns = (column,)

        raw_kind = str(raw_feature_type[source_name]).lower()
        mutable = bool(raw_feature_mutability[source_name])
        actionability = str(raw_feature_actionability[source_name]).lower()

        if len(group_columns) == 1 and group_columns[0] == source_name:
            if source_name in encoded_value_mapping:
                encoding_kind = "mapping"
                categories = tuple(
                    encoded_value_mapping[source_name][key]
                    for key in sorted(encoded_value_mapping[source_name].keys())
                )
            else:
                encoding_kind = "scalar"
                if raw_kind == "categorical":
                    categories = tuple(
                        sorted(
                            {
                                _normalize_category_key(value)
                                for value in feature_df[source_name].dropna().tolist()
                            }
                        )
                    )
                else:
                    categories = tuple()
        else:
            if all("_cat_" in group_column for group_column in group_columns):
                encoding_kind = "onehot"
                prefix = f"{source_name}_cat_"
                categories = tuple(
                    _parse_category_suffix(group_column[len(prefix) :])
                    for group_column in group_columns
                )
            elif all("_therm_" in group_column for group_column in group_columns):
                raise ValueError(
                    "CfrlMethod does not support thermometer-encoded features"
                )
            else:
                raise ValueError(
                    f"Unsupported encoded feature group for '{source_name}': {group_columns}"
                )

        if raw_kind == "categorical":
            categorical_indices.append(len(features))
            raw_to_idx[source_name] = {
                _normalize_category_key(category): idx
                for idx, category in enumerate(categories)
            }
            idx_to_raw[source_name] = {
                idx: category for idx, category in enumerate(categories)
            }
            category_map[len(features)] = [str(category) for category in categories]
            attr_bounds[source_name] = [0.0, float(max(0, len(categories) - 1))]
            feature_types[source_name] = int
            output_python_type = int if encoding_kind == "mapping" else object
            round_output = encoding_kind == "mapping"
        else:
            numerical_indices.append(len(features))
            series = feature_df[group_columns[0]].astype("float64")
            scaled_key = (
                group_columns[0] if group_columns[0] in scaling_map else source_name
            )
            is_normalized = (
                str(scaling_map.get(scaled_key, "none")).lower() == "normalize"
            )
            lower_bound = 0.0 if is_normalized else float(series.min())
            upper_bound = 1.0 if is_normalized else float(series.max())
            attr_bounds[source_name] = [lower_bound, upper_bound]
            output_python_type = _infer_python_type(series)
            round_output = raw_kind == "binary" or (
                output_python_type is int and not is_normalized
            )
            feature_types[source_name] = output_python_type

        features.append(
            CfrlFeature(
                name=source_name,
                output_columns=tuple(group_columns),
                raw_feature_kind=raw_kind,
                encoding_kind=encoding_kind,
                mutable=mutable,
                actionability=actionability,
                categories=tuple(categories),
                lower_bound=attr_bounds[source_name][0],
                upper_bound=attr_bounds[source_name][1],
                output_python_type=output_python_type,
                round_output=round_output,
            )
        )
        feature_names.append(source_name)
        seen_sources.add(source_name)

    return CfrlSchema(
        feature_names=feature_names,
        output_columns=output_columns,
        features=features,
        category_map=category_map,
        raw_to_idx=raw_to_idx,
        idx_to_raw=idx_to_raw,
        attr_bounds=attr_bounds,
        categorical_indices=categorical_indices,
        numerical_indices=numerical_indices,
        feature_types=feature_types,
        output_to_source=output_to_source,
    )


def frame_to_cfrl_array(frame: pd.DataFrame, schema: CfrlSchema) -> np.ndarray:
    ordered = frame.reindex(columns=schema.output_columns, fill_value=0.0)
    arr = np.zeros((ordered.shape[0], len(schema.features)), dtype=np.float32)

    for index, feature in enumerate(schema.features):
        if feature.raw_feature_kind == "categorical":
            if feature.encoding_kind == "scalar":
                values = [
                    schema.raw_to_idx[feature.name][_normalize_category_key(value)]
                    for value in ordered[feature.output_columns[0]].tolist()
                ]
                arr[:, index] = np.asarray(values, dtype=np.float32)
            elif feature.encoding_kind == "mapping":
                values = (
                    np.rint(
                        ordered[feature.output_columns[0]].to_numpy(dtype=np.float32)
                    )
                    .astype(np.int64)
                    .clip(0, len(feature.categories) - 1)
                )
                arr[:, index] = values.astype(np.float32)
            else:
                block = ordered.loc[:, list(feature.output_columns)].to_numpy(
                    dtype=np.float32
                )
                arr[:, index] = np.argmax(block, axis=1).astype(np.float32)
        else:
            arr[:, index] = ordered[feature.output_columns[0]].to_numpy(
                dtype=np.float32
            )

    return arr


def cfrl_to_benchmark_frame(
    arr_zero: np.ndarray,
    schema: CfrlSchema,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    arr = np.atleast_2d(arr_zero).astype(np.float32, copy=False)
    output = pd.DataFrame(index=index if index is not None else range(arr.shape[0]))

    for feature_index, feature in enumerate(schema.features):
        values = arr[:, feature_index]
        if feature.raw_feature_kind == "categorical":
            idxs = np.rint(values).astype(np.int64).clip(0, len(feature.categories) - 1)
            if feature.encoding_kind == "scalar":
                output[feature.output_columns[0]] = [
                    schema.idx_to_raw[feature.name][int(idx)] for idx in idxs
                ]
            elif feature.encoding_kind == "mapping":
                output[feature.output_columns[0]] = idxs.astype(np.float32)
            else:
                block = np.zeros(
                    (arr.shape[0], len(feature.output_columns)), dtype=np.float32
                )
                block[np.arange(arr.shape[0]), idxs] = 1.0
                for block_index, column in enumerate(feature.output_columns):
                    output[column] = block[:, block_index]
            continue

        clipped = np.clip(values, feature.lower_bound, feature.upper_bound)
        if feature.round_output:
            clipped = np.rint(clipped)
        if feature.output_python_type is int:
            output[feature.output_columns[0]] = clipped.astype(np.int64)
        else:
            output[feature.output_columns[0]] = clipped.astype(np.float32)

    output = output.reindex(columns=schema.output_columns)
    return output


def build_predictor(
    target_model: ModelObject,
    schema: CfrlSchema,
) -> Callable[[np.ndarray], np.ndarray]:
    def predictor(x: np.ndarray) -> np.ndarray:
        model_input = cfrl_to_benchmark_frame(x, schema=schema)
        preds = target_model.get_prediction(model_input, proba=True)
        if isinstance(preds, torch.Tensor):
            return preds.detach().cpu().numpy()
        if isinstance(preds, pd.DataFrame):
            return preds.to_numpy()
        return np.asarray(preds)

    return predictor


def resolve_immutable_features_and_ranges(
    schema: CfrlSchema,
    immutable_features: list[str] | None,
    constrained_ranges: dict[str, list[float]] | None,
) -> tuple[list[str], dict[str, list[float]]]:
    immutable: set[str] = set()
    ranges: dict[str, list[float]] = {}

    if immutable_features is not None:
        immutable.update(
            schema.source_name_for_alias(name) for name in immutable_features
        )

    if constrained_ranges is not None:
        for name, bounds in constrained_ranges.items():
            source_name = schema.source_name_for_alias(name)
            if len(bounds) != 2:
                raise ValueError(
                    "constrained_ranges entries must contain [lower, upper]"
                )
            ranges[source_name] = [float(bounds[0]), float(bounds[1])]

    if immutable_features is not None or constrained_ranges is not None:
        return sorted(immutable), ranges

    for feature in schema.features:
        actionability = feature.actionability
        if (not feature.mutable) or actionability in {"none", "same"}:
            immutable.add(feature.name)
            continue

        if feature.raw_feature_kind == "categorical":
            if actionability in _MONOTONE_ACTIONABILITY:
                raise ValueError(
                    f"CfrlMethod does not support monotone categorical actionability for feature '{feature.name}'"
                )
            continue

        if actionability in {"same-or-increase", "increase"}:
            ranges[feature.name] = [0.0, 1.0]
        elif actionability in {"same-or-decrease", "decrease"}:
            ranges[feature.name] = [-1.0, 0.0]

    return sorted(immutable), ranges


def predict_label_indices(
    target_model: ModelObject,
    factuals: pd.DataFrame,
) -> np.ndarray:
    prediction = target_model.get_prediction(factuals, proba=True)
    if isinstance(prediction, torch.Tensor):
        return prediction.detach().cpu().numpy().argmax(axis=1)
    return np.asarray(prediction).argmax(axis=1)


def resolve_target_indices(
    target_model: ModelObject,
    factuals: pd.DataFrame,
    desired_class: int | str | None,
) -> np.ndarray:
    original_prediction = predict_label_indices(target_model, factuals)
    class_to_index = target_model.get_class_to_index()
    if desired_class is not None:
        if desired_class not in class_to_index:
            raise ValueError("desired_class is invalid for the trained target model")
        return np.full_like(
            original_prediction,
            fill_value=int(class_to_index[desired_class]),
            dtype=np.int64,
        )

    if len(class_to_index) != 2:
        raise ValueError(
            "desired_class=None is supported for binary classification only"
        )
    return 1 - original_prediction.astype(np.int64, copy=False)


def validate_counterfactuals(
    target_model: ModelObject,
    factuals: pd.DataFrame,
    candidates: pd.DataFrame,
    desired_class: int | str | None = None,
) -> pd.DataFrame:
    if list(candidates.columns) != list(factuals.columns):
        candidates = candidates.reindex(columns=factuals.columns)
    candidates = candidates.copy(deep=True)

    if candidates.shape[0] != factuals.shape[0]:
        raise ValueError("Candidates must preserve the number of factual rows")

    valid_rows = ~candidates.isna().any(axis=1)
    if not bool(valid_rows.any()):
        return candidates

    target_indices = resolve_target_indices(target_model, factuals, desired_class)
    candidate_prediction = predict_label_indices(
        target_model, candidates.loc[valid_rows]
    )

    success_mask = pd.Series(False, index=candidates.index, dtype=bool)
    success_mask.loc[valid_rows] = (
        candidate_prediction.astype(np.int64, copy=False)
        == target_indices[valid_rows.to_numpy()]
    )
    candidates.loc[~success_mask, :] = np.nan
    return candidates
