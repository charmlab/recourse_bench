from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataset.dataset_object import DatasetObject
from evaluation.evaluation_object import EvaluationObject
from evaluation.evaluation_utils import dataset_has_attr, resolve_evaluation_inputs
from utils.registry import register

_MONOTONE_ACTIONABILITY = {
    "same-or-increase",
    "increase",
    "same-or-decrease",
    "decrease",
}


@dataclass(frozen=True)
class FeatureUnit:
    source_name: str
    columns: tuple[str, ...]
    encoding_kind: str
    feature_kind: str
    mutable: bool
    actionability: str
    valid_codes: tuple[float, ...] | None = None


def _compare_dataset_attr(
    factuals: DatasetObject,
    counterfactuals: DatasetObject,
    flag: str,
) -> None:
    factual_has_attr = dataset_has_attr(factuals, flag)
    counterfactual_has_attr = dataset_has_attr(counterfactuals, flag)
    if factual_has_attr != counterfactual_has_attr:
        raise ValueError(
            f"factuals and counterfactuals disagree on metadata availability: {flag}"
        )
    if not factual_has_attr:
        return
    if factuals.attr(flag) != counterfactuals.attr(flag):
        raise ValueError(f"factuals and counterfactuals metadata conflict: {flag}")


def _resolve_metadata_space(
    dataset: DatasetObject,
    feature_names: list[str],
) -> str:
    has_encoded = all(
        dataset_has_attr(dataset, flag)
        for flag in (
            "encoded_feature_type",
            "encoded_feature_mutability",
            "encoded_feature_actionability",
        )
    )
    has_raw = all(
        dataset_has_attr(dataset, flag)
        for flag in (
            "raw_feature_type",
            "raw_feature_mutability",
            "raw_feature_actionability",
        )
    )

    encoded_matches = False
    if has_encoded:
        encoded_feature_type = dataset.attr("encoded_feature_type")
        encoded_matches = all(
            feature_name in encoded_feature_type for feature_name in feature_names
        )

    raw_matches = False
    if has_raw:
        raw_feature_type = dataset.attr("raw_feature_type")
        raw_matches = all(
            feature_name in raw_feature_type for feature_name in feature_names
        )

    if encoded_matches and dataset_has_attr(dataset, "encoding"):
        return "encoded"
    if raw_matches:
        return "raw"
    if encoded_matches:
        return "encoded"
    raise ValueError("Could not resolve metadata space for current feature columns")


def _validate_metadata_consistency(
    factuals: DatasetObject,
    counterfactuals: DatasetObject,
) -> None:
    for flag in (
        "raw_feature_type",
        "raw_feature_mutability",
        "raw_feature_actionability",
        "encoded_feature_type",
        "encoded_feature_mutability",
        "encoded_feature_actionability",
        "encoding",
        "encoded_value_mapping",
    ):
        _compare_dataset_attr(factuals, counterfactuals, flag)


def _resolve_encoding_kind(
    source_name: str,
    columns: tuple[str, ...],
    encoded_value_mapping: dict[str, dict[int, object]],
) -> str:
    if len(columns) > 1:
        if all("_therm_" in column for column in columns):
            return "thermometer"
        if all("_cat_" in column for column in columns):
            return "onehot"
        raise ValueError(
            f"Unsupported grouped encoding for feature '{source_name}': {columns}"
        )

    if source_name in encoded_value_mapping and columns[0] == source_name:
        return "mapping"
    return "scalar"


def _build_feature_units(
    dataset: DatasetObject,
    feature_names: list[str],
    metadata_space: str,
) -> list[FeatureUnit]:
    if metadata_space == "encoded":
        feature_type = dataset.attr("encoded_feature_type")
        feature_mutability = dataset.attr("encoded_feature_mutability")
        feature_actionability = dataset.attr("encoded_feature_actionability")
    else:
        feature_type = dataset.attr("raw_feature_type")
        feature_mutability = dataset.attr("raw_feature_mutability")
        feature_actionability = dataset.attr("raw_feature_actionability")

    encoding_map = (
        dataset.attr("encoding") if dataset_has_attr(dataset, "encoding") else {}
    )
    encoded_value_mapping = (
        dataset.attr("encoded_value_mapping")
        if dataset_has_attr(dataset, "encoded_value_mapping")
        else {}
    )
    feature_name_set = set(feature_names)

    processed_to_source: dict[str, str] = {}
    for source_name, encoded_columns in encoding_map.items():
        for encoded_column in encoded_columns:
            if encoded_column in feature_name_set:
                processed_to_source[str(encoded_column)] = str(source_name)

    seen_sources: set[str] = set()
    units: list[FeatureUnit] = []
    for feature_name in feature_names:
        source_name = processed_to_source.get(feature_name, feature_name)
        if source_name in seen_sources:
            continue

        if source_name in encoding_map:
            configured_columns = tuple(
                str(column) for column in encoding_map[source_name]
            )
            present_columns = tuple(
                column for column in configured_columns if column in feature_name_set
            )
            if present_columns and len(present_columns) != len(configured_columns):
                raise ValueError(
                    f"Incomplete encoded feature group for '{source_name}': {present_columns}"
                )
            columns = present_columns if present_columns else (feature_name,)
        else:
            columns = (feature_name,)

        encoding_kind = _resolve_encoding_kind(
            source_name=source_name,
            columns=columns,
            encoded_value_mapping=encoded_value_mapping,
        )

        reference_column = columns[0]
        mutable_values = {bool(feature_mutability[column]) for column in columns}
        actionability_values = {
            str(feature_actionability[column]).lower() for column in columns
        }
        if len(mutable_values) != 1 or len(actionability_values) != 1:
            raise ValueError(
                f"Inconsistent metadata within feature group '{source_name}'"
            )

        if encoding_kind in {"onehot", "thermometer"}:
            feature_kind = "categorical"
        else:
            feature_kind = str(feature_type[reference_column]).lower()

        actionability = next(iter(actionability_values))
        if actionability in _MONOTONE_ACTIONABILITY and encoding_kind in {
            "onehot",
        }:
            raise ValueError(
                f"Unsupported monotone actionability for unordered encoded feature '{source_name}'"
            )
        if (
            actionability in _MONOTONE_ACTIONABILITY
            and encoding_kind == "scalar"
            and feature_kind == "categorical"
        ):
            raise ValueError(
                f"Unsupported monotone actionability for categorical feature '{source_name}'"
            )

        valid_codes = None
        if encoding_kind == "mapping":
            if source_name not in encoded_value_mapping:
                raise ValueError(
                    f"Missing encoded_value_mapping for mapped feature '{source_name}'"
                )
            valid_codes = tuple(
                float(code)
                for code in sorted(encoded_value_mapping[source_name].keys())
            )

        units.append(
            FeatureUnit(
                source_name=source_name,
                columns=columns,
                encoding_kind=encoding_kind,
                feature_kind=feature_kind,
                mutable=next(iter(mutable_values)),
                actionability=actionability,
                valid_codes=valid_codes,
            )
        )
        seen_sources.add(source_name)

    return units


def _is_binary_scalar_valid(value: object, tolerance: float) -> bool:
    if pd.isna(value):
        return False
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isclose(numeric_value, 0.0, atol=tolerance)
        or np.isclose(numeric_value, 1.0, atol=tolerance)
    )


def _is_onehot_valid(values: np.ndarray, tolerance: float) -> bool:
    if values.ndim != 1 or values.shape[0] < 2:
        return False
    binary_mask = np.logical_or(
        np.isclose(values, 0.0, atol=tolerance),
        np.isclose(values, 1.0, atol=tolerance),
    )
    if not bool(np.all(binary_mask)):
        return False
    rounded = np.where(np.isclose(values, 1.0, atol=tolerance), 1.0, 0.0)
    return bool(np.isclose(float(rounded.sum()), 1.0, atol=tolerance))


def _is_thermometer_valid(values: np.ndarray, tolerance: float) -> bool:
    if values.ndim != 1 or values.shape[0] < 2:
        return False
    binary_mask = np.logical_or(
        np.isclose(values, 0.0, atol=tolerance),
        np.isclose(values, 1.0, atol=tolerance),
    )
    if not bool(np.all(binary_mask)):
        return False
    rounded = np.where(np.isclose(values, 1.0, atol=tolerance), 1, 0)
    if rounded[0] != 1:
        return False
    return bool(np.all(rounded[:-1] >= rounded[1:]))


def _decode_thermometer_level(values: np.ndarray, tolerance: float) -> int | None:
    if not _is_thermometer_valid(values, tolerance):
        return None
    rounded = np.where(np.isclose(values, 1.0, atol=tolerance), 1, 0)
    return int(rounded.sum() - 1)


def _scalar_equal(left: object, right: object, tolerance: float) -> bool:
    if pd.isna(left) or pd.isna(right):
        return False
    try:
        return bool(np.isclose(float(left), float(right), atol=tolerance))
    except (TypeError, ValueError):
        return bool(left == right)


def _unit_equal(
    factual_values: np.ndarray,
    counterfactual_values: np.ndarray,
    tolerance: float,
) -> bool:
    return all(
        _scalar_equal(left, right, tolerance)
        for left, right in zip(factual_values.tolist(), counterfactual_values.tolist())
    )


def _is_type_violation(
    unit: FeatureUnit,
    counterfactual_values: np.ndarray,
    tolerance: float,
) -> bool:
    if unit.encoding_kind == "onehot":
        return not _is_onehot_valid(counterfactual_values.astype(float), tolerance)
    if unit.encoding_kind == "thermometer":
        return not _is_thermometer_valid(counterfactual_values.astype(float), tolerance)
    if unit.encoding_kind == "mapping":
        if unit.valid_codes is None:
            raise ValueError(
                f"Missing valid codes for mapped feature '{unit.source_name}'"
            )
        if counterfactual_values.shape[0] != 1:
            raise ValueError(f"Mapped feature '{unit.source_name}' must be scalar")
        value = counterfactual_values[0]
        return not any(
            np.isclose(float(value), valid_code, atol=tolerance)
            for valid_code in unit.valid_codes
        )
    if counterfactual_values.shape[0] != 1:
        raise ValueError(f"Scalar feature '{unit.source_name}' must contain one column")

    value = counterfactual_values[0]
    if unit.feature_kind == "binary":
        return not _is_binary_scalar_valid(value, tolerance)
    if unit.feature_kind == "numerical":
        try:
            return not bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return True
    if unit.feature_kind == "categorical":
        return bool(pd.isna(value))

    if pd.isna(value):
        return True
    try:
        return not bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _is_mutability_violation(
    unit: FeatureUnit,
    factual_values: np.ndarray,
    counterfactual_values: np.ndarray,
    tolerance: float,
) -> bool:
    if unit.mutable:
        return False
    return not _unit_equal(factual_values, counterfactual_values, tolerance)


def _is_actionability_violation(
    unit: FeatureUnit,
    factual_values: np.ndarray,
    counterfactual_values: np.ndarray,
    tolerance: float,
) -> bool:
    actionability = unit.actionability
    if actionability == "any":
        return False
    if actionability in {"none", "same"}:
        return not _unit_equal(factual_values, counterfactual_values, tolerance)

    if actionability not in _MONOTONE_ACTIONABILITY:
        raise ValueError(
            f"Unsupported actionability for feature '{unit.source_name}': {actionability}"
        )

    if unit.encoding_kind == "thermometer":
        factual_level = _decode_thermometer_level(
            factual_values.astype(float), tolerance
        )
        counterfactual_level = _decode_thermometer_level(
            counterfactual_values.astype(float), tolerance
        )
        if factual_level is None or counterfactual_level is None:
            return True
        factual_value = float(factual_level)
        counterfactual_value = float(counterfactual_level)
    else:
        if factual_values.shape[0] != 1 or counterfactual_values.shape[0] != 1:
            raise ValueError(
                f"Unsupported actionability comparison for feature '{unit.source_name}'"
            )
        try:
            factual_value = float(factual_values[0])
            counterfactual_value = float(counterfactual_values[0])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Actionability comparison requires numeric values for '{unit.source_name}'"
            ) from error

    if actionability == "same-or-increase":
        return counterfactual_value < (factual_value - tolerance)
    if actionability == "increase":
        return counterfactual_value <= (factual_value + tolerance)
    if actionability == "same-or-decrease":
        return counterfactual_value > (factual_value + tolerance)
    return counterfactual_value >= (factual_value - tolerance)


@register("constraints")
class ConstraintsEvaluation(EvaluationObject):
    def __init__(self, tolerance: float = 1e-6, **kwargs):
        self._tolerance = float(tolerance)
        if self._tolerance < 0:
            raise ValueError("tolerance must be >= 0")

    def evaluate(
        self, factuals: DatasetObject, counterfactuals: DatasetObject
    ) -> pd.DataFrame:
        (
            factual_features,
            counterfactual_features,
            evaluation_mask,
            success_mask,
        ) = resolve_evaluation_inputs(factuals, counterfactuals)

        _validate_metadata_consistency(factuals, counterfactuals)

        feature_names = list(factual_features.columns)
        factual_space = _resolve_metadata_space(factuals, feature_names)
        counterfactual_space = _resolve_metadata_space(counterfactuals, feature_names)
        if factual_space != counterfactual_space:
            raise ValueError(
                "factuals and counterfactuals use conflicting metadata spaces"
            )

        units = _build_feature_units(factuals, feature_names, factual_space)
        selected_mask = evaluation_mask & success_mask
        if selected_mask.sum() == 0:
            return pd.DataFrame(
                [
                    {
                        "type_violation_rates": float("nan"),
                        "mutability_violation_rates": float("nan"),
                        "actionability_violation_rates": float("nan"),
                    }
                ]
            )

        factual_selected = factual_features.loc[selected_mask.to_numpy()]
        counterfactual_selected = counterfactual_features.loc[selected_mask.to_numpy()]

        type_violations: list[float] = []
        mutability_violations: list[float] = []
        actionability_violations: list[float] = []
        for row_index in factual_selected.index:
            factual_row = factual_selected.loc[row_index]
            counterfactual_row = counterfactual_selected.loc[row_index]

            type_violation = False
            mutability_violation = False
            actionability_violation = False
            for unit in units:
                factual_values = factual_row.loc[list(unit.columns)].to_numpy()
                counterfactual_values = counterfactual_row.loc[
                    list(unit.columns)
                ].to_numpy()

                type_violation |= _is_type_violation(
                    unit=unit,
                    counterfactual_values=counterfactual_values,
                    tolerance=self._tolerance,
                )
                mutability_violation |= _is_mutability_violation(
                    unit=unit,
                    factual_values=factual_values,
                    counterfactual_values=counterfactual_values,
                    tolerance=self._tolerance,
                )
                actionability_violation |= _is_actionability_violation(
                    unit=unit,
                    factual_values=factual_values,
                    counterfactual_values=counterfactual_values,
                    tolerance=self._tolerance,
                )

            type_violations.append(float(type_violation))
            mutability_violations.append(float(mutability_violation))
            actionability_violations.append(float(actionability_violation))

        return pd.DataFrame(
            [
                {
                    "type_violation_rates": float(np.mean(type_violations)),
                    "mutability_violation_rates": float(np.mean(mutability_violations)),
                    "actionability_violation_rates": float(
                        np.mean(actionability_violations)
                    ),
                }
            ]
        )
